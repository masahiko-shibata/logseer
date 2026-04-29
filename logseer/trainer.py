import pickle
import sys
import numpy as np
import keras
from keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from scipy.stats import fisher_exact
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from .models import getModel, getEmbeddingLayer
from .checkpoints import MultiMetricCheckpoint, BestF1Checkpoint, F1Logger
from .loader import Loader
from .tester import Tester


def split_data(texts, labels, test_texts, test_labels,
               validation_split=0.1, validate_on_test_data=False):
    if validate_on_test_data:
        return (np.array(texts), np.array(labels),
                np.array(test_texts), np.array(test_labels))
    nb_val = int(validation_split * len(texts))
    return (np.array(texts[:-nb_val]), np.array(labels[:-nb_val]),
            np.array(texts[-nb_val:]), np.array(labels[-nb_val:]))


def setup_tokenizer(train_texts, tokenizer_path, max_nb_words, retrain=False):
    if retrain:
        with open(tokenizer_path, 'rb') as f:
            return pickle.load(f)
    tokenizer = Tokenizer(num_words=max_nb_words, filters='', lower=False)
    tokenizer.fit_on_texts(train_texts)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def prepare_sequences(tokenizer, train_texts, val_texts, test_texts, max_sequence_length):
    train_seqs = tokenizer.texts_to_sequences(train_texts)
    val_seqs   = tokenizer.texts_to_sequences(val_texts)
    test_seqs  = tokenizer.texts_to_sequences(test_texts)
    train_data = np.array(pad_sequences(train_seqs, maxlen=max_sequence_length), dtype=np.int32)
    val_data   = np.array(pad_sequences(val_seqs,   maxlen=max_sequence_length), dtype=np.int32)
    test_data  = np.array(pad_sequences(test_seqs,  maxlen=max_sequence_length), dtype=np.int32)
    max_seq_len = max(len(s) for s in train_seqs + val_seqs)
    print('Longest data seq in train/val %s tokens' % max_seq_len)
    print('Shape of train data:', train_data.shape)
    return train_data, val_data, test_data


def train_nn(model_name, embedding_layer, train_data, train_labels, val_data, val_labels,
             test_data, test_labels, tester, *,
             model_save_path, epochs, batch_size, learning_rate, max_loss, retrain=False,
             checkpoint_type='multi_metric', start_from_epoch=0, es_start_from_epoch=0,
             use_early_stopping=False, patience=None, monitor='val_recall', mode='max',
             restore_best_weights=False, error_weight=1, threshold=0.5, label_smoothing=0.0):
    train_labels = np.array(train_labels, dtype=np.int32)
    val_labels   = np.array(val_labels,   dtype=np.int32)

    nb_val_error = np.count_nonzero(val_labels)
    if nb_val_error < 1:
        print('Oops. Validation contains no error. Skipping this repetition.')
        return False

    print('Number of validation data sets: %s' % len(val_labels))
    print('Number of errors in validation data: %s' % nb_val_error)

    if retrain:
        model = load_model(model_save_path)
    else:
        model = getModel(model_name, embedding_layer)

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
                  optimizer=optimizer,
                  metrics=[keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall'),
                           ])

    # Build checkpoint callback
    if checkpoint_type == 'best_f1':
        checkpoint_cb = BestF1Checkpoint(filepath=model_save_path,
                                         start_from_epoch=start_from_epoch,
                                         max_loss=max_loss)
    elif checkpoint_type == 'standard':
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=model_save_path, monitor=monitor, mode=mode,
            save_best_only=True, verbose=1)
    else:  # 'multi_metric' (default)
        checkpoint_cb = MultiMetricCheckpoint(filepath=model_save_path,
                                              max_loss=max_loss,
                                              start_from_epoch=start_from_epoch)

    callbacks = [F1Logger(), checkpoint_cb]

    # Optional early stopping
    if use_early_stopping and patience is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=patience,
            start_from_epoch=es_start_from_epoch,
            restore_best_weights=restore_best_weights, verbose=1))

    model.fit(train_data, train_labels,
              validation_data=(val_data, val_labels),
              epochs=epochs,
              verbose=1,
              batch_size=batch_size,
              callbacks=callbacks,
              class_weight={0: 1, 1: error_weight} if error_weight != 1 else None)

    model = load_model(model_save_path)
    tester.testModel(model, test_data, test_labels, threshold=threshold)
    return True


SKLEARN_MODELS = frozenset({'xgb', 'svm', 'rf'})


def train_sklearn(tokenizer, train_texts, test_texts, train_labels, test_labels, tester, *,
                  sklearn_model='xgb', sklearn_weight=6, sklearn_threshold=0.5):
    """Train a single sklearn/tree-based model and evaluate it.

    sklearn_model: 'xgb' | 'svm' | 'rf' | 'none'
    """
    if not sklearn_model or sklearn_model == 'none':
        return

    train_mat = tokenizer.texts_to_matrix(train_texts, mode='tfidf')
    test_mat  = tokenizer.texts_to_matrix(test_texts,  mode='tfidf')

    if sklearn_model == 'xgb':
        model = xgb.XGBClassifier(scale_pos_weight=sklearn_weight)
    elif sklearn_model == 'svm':
        model = svm.SVC(probability=True, class_weight={1: sklearn_weight})
    elif sklearn_model == 'rf':
        model = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=50,
                                       class_weight={1: sklearn_weight})
    else:
        raise ValueError(f'Unknown sklearn_model: {sklearn_model!r}. Choose from: xgb, svm, rf, none')

    model.name = sklearn_model
    model.fit(train_mat, train_labels)
    pickle.dump(model, open(f'{sklearn_model}.pkl', 'wb'))
    tester.testModel(model, test_mat, test_labels, threshold=sklearn_threshold)


def print_ensemble(tester, ensemble_model=None, sweep_start=0.50, sweep_end=0.85, sweep_step=0.05):
    nn_row  = next((r for r in tester.stored if r[0] not in SKLEARN_MODELS), None)
    if ensemble_model:
        skl_row = next((r for r in tester.stored if r[0] == ensemble_model), None)
    else:
        skl_row = next((r for r in tester.stored if r[0] in SKLEARN_MODELS), None)
    if not (nn_row and skl_row):
        return

    nn_name  = nn_row[0]
    skl_name = skl_row[0]

    y        = np.array(nn_row[1])
    cnn      = np.array(nn_row[2])
    skl      = np.array(skl_row[2])
    cnn_prob = np.array(nn_row[3])
    skl_prob = np.array(skl_row[3])
    errors       = y == 1
    cnn_tp       = np.sum(errors & (cnn == 1))
    skl_tp       = np.sum(errors & (skl == 1))
    both_tp      = np.sum(errors & (cnn == 1) & (skl == 1))
    both_fp      = np.sum(~errors & (cnn == 1) & (skl == 1))
    either_tp    = np.sum(errors & ((cnn == 1) | (skl == 1)))
    either_fp    = np.sum(~errors & ((cnn == 1) | (skl == 1)))
    total_errors = np.sum(errors)
    or_p  = either_tp / (either_tp + either_fp) if (either_tp + either_fp) > 0 else 0.0
    or_r  = either_tp / total_errors if total_errors > 0 else 0.0
    or_f1 = 2 * or_p * or_r / (or_p + or_r) if (or_p + or_r) > 0 else 0.0
    and_p  = both_tp / (both_tp + both_fp) if (both_tp + both_fp) > 0 else 0.0
    and_r  = both_tp / total_errors if total_errors > 0 else 0.0
    and_f1 = 2 * and_p * and_r / (and_p + and_r) if (and_p + and_r) > 0 else 0.0
    w = max(len(nn_name), len(skl_name)) + len('-only TP') + 2
    def lbl(s):
        return f'  {s:<{w}}'

    print()
    print(f'### Ensemble ({nn_name} | {skl_name}) ###')
    print()
    print(lbl('Total errors')    + f': {total_errors}')
    print(lbl(f'{nn_name} TP')   + f': {cnn_tp}  (recall {cnn_tp/total_errors:.3f})')
    print(lbl(f'{skl_name} TP')  + f': {skl_tp}  (recall {skl_tp/total_errors:.3f})')
    print(lbl('Overlap (both)')  + f': {both_tp}')
    print(lbl(f'{nn_name}-only TP')  + f': {cnn_tp - both_tp}')
    print(lbl(f'{skl_name}-only TP') + f': {skl_tp - both_tp}')
    print(f'  Union TP          : {either_tp}')
    print(f'  Union FP          : {either_fp}')
    print(f'  OR  ensemble      : precision {or_p:.3f}  recall {or_r:.3f}  F1 {or_f1:.3f}')
    print(f'  AND ensemble      : precision {and_p:.3f}  recall {and_r:.3f}  F1 {and_f1:.3f}  (TP={both_tp}  FP={both_fp})')

    def prob_stats(label, probs, mask):
        p = probs[mask]
        if len(p) == 0:
            print(f'  {label}  (no samples)')
            return
        print(f'  {label}  mean={p.mean():.3f}  median={np.median(p):.3f}  min={p.min():.3f}  max={p.max():.3f}  n={len(p)}')

    print()
    print('  -- Threshold tuning --')
    prob_stats(f'{nn_name} TP probs',  cnn_prob, errors & (cnn == 1))
    prob_stats(f'{nn_name} FP probs',  cnn_prob, ~errors & (cnn == 1))
    prob_stats(f'{skl_name} TP probs', skl_prob, errors & (skl == 1))
    prob_stats(f'{skl_name} FP probs', skl_prob, ~errors & (skl == 1))

    thresholds = np.arange(sweep_start, sweep_end + sweep_step / 2, sweep_step)
    print()
    print('  -- AND threshold sweep --')
    print(f'  {"NN_t":>5}  {"SKL_t":>5}  {"TP":>5}  {"FP":>5}  {"precision":>9}  {"recall":>6}  {"F1":>6}')
    for cnn_t in thresholds:
        for skl_t in thresholds:
            m    = (cnn_prob >= cnn_t) & (skl_prob >= skl_t)
            s_tp = int(np.sum(errors & m))
            s_fp = int(np.sum(~errors & m))
            s_p  = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
            s_r  = s_tp / total_errors if total_errors > 0 else 0.0
            s_f1 = 2 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
            print(f'  {cnn_t:>5.2f}  {skl_t:>5.2f}  {s_tp:>5}  {s_fp:>5}  {s_p:>9.3f}  {s_r:>6.3f}  {s_f1:>6.3f}')

    print()
    print('  -- OR threshold sweep --')
    print(f'  {"NN_t":>5}  {"SKL_t":>5}  {"TP":>5}  {"FP":>5}  {"precision":>9}  {"recall":>6}  {"F1":>6}')
    for cnn_t in thresholds:
        for skl_t in thresholds:
            m    = (cnn_prob >= cnn_t) | (skl_prob >= skl_t)
            s_tp = int(np.sum(errors & m))
            s_fp = int(np.sum(~errors & m))
            s_p  = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
            s_r  = s_tp / total_errors if total_errors > 0 else 0.0
            s_f1 = 2 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
            print(f'  {cnn_t:>5.2f}  {skl_t:>5.2f}  {s_tp:>5}  {s_fp:>5}  {s_p:>9.3f}  {s_r:>6.3f}  {s_f1:>6.3f}')



def significance_test(tester):
    print()
    print('### Error Detection ###')
    for row in tester.stored:
        name, y, pred = row[0], row[1], row[2]
        tp = sum(1 for t, p in zip(y, pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y, pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y, pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y, pred) if t == 0 and p == 0)
        _, p_value = fisher_exact([[tn, fp], [fn, tp]], alternative='greater')
        n_errors = tp + fn
        print(f'{name}: TP={tp}/{n_errors}, p-value={p_value:.6f}',
              '*** significant' if p_value < 0.05 else '')


def run_training(
    data_dir,
    *,
    loader_class=None,
    max_nb_words=24000,
    max_sequence_length=26000,
    embedding_dim=32,
    validation_split=0.1,
    validate_on_test_data=False,
    epochs=60,
    batch_size=32,
    model_save_path='logseer.keras',
    tokenizer_path='tokenizer.pickle',
    embedding_layer_type='vanilla',
    model_name='LogCNNLite',
    repetition=100,
    nn_error_weight=1,
    learning_rate=0.0003,
    max_loss=0.7,
    retrain=False,
    numchar=3000,
    toid=6000,
    checkpoint_type='multi_metric',
    start_from_epoch=0,
    es_start_from_epoch=0,
    use_early_stopping=False,
    patience=None,
    monitor='val_recall',
    mode='max',
    restore_best_weights=False,
    test_nn=True,
    sklearn_models=None,
    ensemble_model=None,
    nn_threshold=0.5,
    sklearn_threshold=0.5,
    sklearn_weight=6,
    sweep_start=0.50,
    sweep_end=0.85,
    sweep_step=0.05,
    success_log_ratio=99,
    success_log_ratio_test=12.4,
    dump_proba=False,
    label_smoothing=0.0,
):
    """Run the full training loop and return the Tester instance.

    loader_class: Loader subclass to use (default: JDELoader). Pass Loader for generic use.
    checkpoint_type: 'multi_metric' (default), 'best_f1', or 'standard'
    sklearn_models: list of sklearn models to train, e.g. ['xgb', 'svm', 'rf']. None or [] = skip all.
    ensemble_model: which sklearn model to use in print_ensemble. Defaults to first in sklearn_models.
    use_early_stopping: set to True to enable EarlyStopping (also requires patience to be set)
    patience: number of epochs with no improvement before stopping (only used when use_early_stopping=True)
    """
    from .jde_loader import JDELoader
    if loader_class is None:
        loader_class = JDELoader

    # Normalise sklearn_models to a list
    if sklearn_models is None:
        _sklearn_models = []
    elif isinstance(sklearn_models, str):
        _sklearn_models = [m.strip() for m in sklearn_models.replace(',', ' ').split() if m.strip()]
    else:
        _sklearn_models = list(sklearn_models)
    # Filter out 'none'
    _sklearn_models = [m for m in _sklearn_models if m != 'none']

    # Ensemble partner: first sklearn model by default
    _ensemble_model = ensemble_model or (_sklearn_models[0] if _sklearn_models else None)
    ld = loader_class()
    tester = Tester()

    for i in range(repetition):
        print()
        print('Repetition %s' % str(i + 1))
        texts, labels, test_texts, test_labels = ld.getdata(
            data_dir, numchar=numchar, toid=toid,
            SUCCESS_LOG_RATIO=success_log_ratio,
            SUCCESS_LOG_RATIO_TEST=success_log_ratio_test,
        )
        print('Found %s texts.' % len(texts))
        print('Found %s test texts.' % len(test_texts))

        train_texts, train_labels, val_texts, val_labels = split_data(
            texts, labels, test_texts, test_labels,
            validation_split=validation_split,
            validate_on_test_data=validate_on_test_data,
        )

        tokenizer = setup_tokenizer(train_texts, tokenizer_path, max_nb_words, retrain=retrain)
        print('Found %s unique tokens in the tokenizer.' % len(tokenizer.word_index))

        if test_nn:
            train_data, val_data, test_data = prepare_sequences(
                tokenizer, train_texts, val_texts, test_texts, max_sequence_length)
            emb_layer = getEmbeddingLayer(
                embedding_layer_type, max_nb_words, embedding_dim,
                max_sequence_length, word_index=tokenizer.word_index)
            ok = train_nn(
                model_name, emb_layer,
                train_data, train_labels, val_data, val_labels, test_data, test_labels,
                tester,
                model_save_path=model_save_path, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, max_loss=max_loss, retrain=retrain,
                checkpoint_type=checkpoint_type, start_from_epoch=start_from_epoch,
                es_start_from_epoch=es_start_from_epoch,
                use_early_stopping=use_early_stopping, patience=patience,
                monitor=monitor, mode=mode,
                restore_best_weights=restore_best_weights,
                error_weight=nn_error_weight,
                threshold=nn_threshold,
                label_smoothing=label_smoothing,
            )
            if not ok:
                continue

        for skl_m in _sklearn_models:
            train_sklearn(tokenizer, train_texts, test_texts, train_labels, test_labels, tester,
                          sklearn_model=skl_m,
                          sklearn_weight=sklearn_weight,
                          sklearn_threshold=sklearn_threshold)

        if i % 10 == 9:
            tester.total(heatmap=False)

        print_ensemble(tester, ensemble_model=_ensemble_model,
                       sweep_start=sweep_start, sweep_end=sweep_end, sweep_step=sweep_step)

    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, s):
            for f in self.files: f.write(s)
        def flush(self):
            for f in self.files: f.flush()

    with open('test_result.txt', 'w', encoding='utf-8') as rf:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, rf)
        try:
            tester.total(heatmap=True)
            print_ensemble(tester, ensemble_model=_ensemble_model,
                           sweep_start=sweep_start, sweep_end=sweep_end, sweep_step=sweep_step)
            significance_test(tester)
        finally:
            sys.stdout = old_stdout
    if dump_proba:
        tester.dump_proba()
    return tester

