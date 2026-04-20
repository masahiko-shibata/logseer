# LogSeer

Predicting enterprise system operation failures from server log state, before the operation begins.

## Overview

Large enterprise systems — ERP platforms, middleware, batch processing infrastructure — run on multiple servers that continuously emit logs reflecting the current system state. When a critical operation such as a deployment or batch job is initiated, it sometimes fails — not because the system is broken, but because its current state is incompatible with that particular operation succeeding. These failures are often costly: the operation may run for a long time before hitting an error or timeout, and recovery requires intervention and a re-attempt.

LogSeer collects logs from all relevant servers, reads the current system state, and predicts whether a planned operation will succeed or fail — **before it is initiated**. If failure is predicted, a preventive action can be taken before the operation starts — such as holding the operation, rescheduling, or restarting the system — avoiding the costly failure entirely.

It is designed to generalize to any system and any operation type. Oracle JD Edwards (JDE) EnterpriseOne serves as the primary implementation and validation environment.

## Problem

In large enterprise environments, a failed operation typically follows this pattern:

1. An operation (deployment, batch job, etc.) is initiated on a system that is running normally
2. An underlying state issue causes it to fail — either by hitting an error partway through, or by running until a timeout
3. Recovery is required — the specific action depends on the system and operation
4. The operation is re-attempted after recovery

If the system state can be assessed from logs **before the operation starts**, the failure can be avoided entirely.

### Prediction, not detection

Most log-based ML systems perform anomaly *detection* — they analyze logs produced during or after an event to identify that something has already gone wrong. LogSeer is fundamentally different: it performs *prediction*. The logs it reads do not contain the failure. They contain the latent system state that will cause a future operation to fail. The label is the outcome of an operation that has not yet started.

This makes the problem harder — there is no explicit error signal in the input — but also more valuable, as the failure can be prevented entirely rather than just identified after the fact.

| Outcome | Cost |
|---|---|
| False Negative (missed failure) | Full timeout + review + restart + re-attempt |
| False Positive (false alarm) | Unnecessary restart and service disruption |

The relative cost ratio between FN and FP depends on the system. For the JDE reference environment, FN costs approximately **3× more** than FP, so the system is tuned to favor recall.

## Approach

LogSeer ingests log files collected from all relevant servers, combines them into a single representation of the current system state, preprocessed, and classified as predicted success or predicted failure.

The pipeline includes:

- **Multi-file log ingestion** — log files from multiple server processes are combined per operation into a single text representation
- **Domain-aware preprocessing** — timestamps, IDs, IP addresses, and other high-cardinality tokens are normalized to reduce noise while preserving meaningful signal
- **LogCNNLite** — a dilated 1D CNN that performed best in comparison against RNN-based models (LSTM, GRU, biLSTM, biGRU). Notably, this is a convolutional architecture more commonly associated with image processing, applied here to log token sequences — sequential models underperformed, suggesting the signal in these logs is not strongly order-dependent at the token level
- **XGBoost** — a TF-IDF based classifier that captures anomalous token occurrence patterns
- **Ensemble** — union of both model predictions, optimized for recall given the asymmetric cost structure
- **Repeated evaluation** — 100 randomized train/test splits with Fisher's exact significance testing for statistically reliable performance estimates

> Note: log collection from live servers is handled by a separate pipeline outside this repository. This codebase assumes logs have already been collected and organized into the data directory structure described below.

## JDE Reference Implementation

The current implementation and training data are based on Oracle JD Edwards EnterpriseOne, a large-scale ERP platform widely used in enterprise environments. This was validated in a large JDE development environment at the global headquarters — with hundreds of concurrent developers and QA engineers.

- Package deployments happen multiple times per day
- A failed deployment times out after ~30 minutes, forcing all users offline
- Recovery (restart + re-deploy) takes approximately 15 minutes of additional downtime

Before each deployment, LogSeer collects the current JDE server logs from all running kernel processes and predicts whether the deployment will succeed. If failure is predicted, preventive system restarts are initiated to avoid the timeout entirely.

## Results

Evaluated on the JDE reference environment over 100 repeated random train/test splits:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| LogCNNLite | 0.396 | 0.292 | 0.336 |
| XGBoost | 0.590 | 0.296 | 0.394 |
| Ensemble (CNN \| XGB) | 0.415 | 0.404 | 0.409 |

## Project Structure

```
logseer/
├── logseer/
│   ├── loader.py        # Log file loading, preprocessing, and train/test split
│   ├── models.py        # CNN, RNN, and attention-based model architectures
│   ├── trainer.py       # Training loop, sklearn models, ensemble reporting
│   ├── tester.py        # Model evaluation and result accumulation
│   ├── checkpoints.py   # Custom Keras checkpoint callbacks
│   └── __init__.py
├── notebooks/
│   ├── train.ipynb      # Training pipeline (Google Colab)
│   └── predict.ipynb    # Inference on new operations
└── requirements.txt
```

## Data Format

Training data is organized by operation outcome:

```
data/
├── error/
│   ├── 1001/   ← failed operation (one or more log files)
│   ├── 1002/
│   └── ...
└── success/
    ├── 2001/   ← successful operation (one or more log files)
    ├── 2002/
    └── ...
```

Each subdirectory contains the log files collected before a single operation run. Multiple log files per run are supported and combined during preprocessing.

## Training

Training is designed to run on Google Colab with a GPU. Open `notebooks/train.ipynb`, configure the parameters in the Configuration cell, and run all cells.

### Two-phase workflow

**Phase 1 — Model tuning** (`REPETITION = 100`, `VALIDATE_ON_TEST_DATA = False`)

Run many repeated train/test splits to evaluate model architecture and hyperparameters under varied data conditions. Results are aggregated across all repetitions to produce statistically stable performance estimates. Use this phase to tune `MODEL_NAME`, `EPOCHS`, `LEARNING_RATE`, etc.

**Phase 2 — Production model generation** (`REPETITION = 1`, `VALIDATE_ON_TEST_DATA = True`)

Once tuning is complete, run a single pass with the full training set and validate on held-out test data. This produces the model file (`logseer.keras`) and tokenizer (`tokenizer.pickle`) used in production inference.

### Why repeated splits?

Repetition is particularly important when the number of failure examples in the training data is small. With only a few dozen to a few hundred failure cases, a single train/test split is heavily influenced by which examples land where by chance. Repeating with different random splits averages out this variance and gives a more reliable estimate of true model performance.

With very large failure datasets (e.g. 10,000+ examples), a single split would likely suffice — but in practice, failure logs are expensive to collect and label, making this regime common.

### Key configuration options

```python
MODEL_NAME            = 'LogCNNLite'  # model architecture to train
REPETITION            = 100           # number of repeated train/test splits (set to 1 for production)
VALIDATE_ON_TEST_DATA = False         # set to True for production model generation
EPOCHS                = 60
NUM_CHAR              = 3000          # characters read from tail of each log file
TO_ID                 = 6000          # upper bound of operation IDs to include
```

## Requirements

```
tensorflow>=2.12
keras>=2.12
scikit-learn
xgboost
pandas
matplotlib
seaborn
scipy
numpy
```

## Status

The core modeling approach and evaluation methodology are complete and validated.

- Validated in a real enterprise environment using production logs
- Predictions evaluated against actual outcomes in shadow mode
- Architecture and results are stable across repeated evaluation

The repository focuses on the prediction layer. Integration with live systems (log collection, orchestration, automated decision-making) is environment-specific and handled separately.
