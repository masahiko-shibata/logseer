import keras


class F1Score(keras.metrics.Metric):
    """Computes F1 from precision and recall so val_f1 is available for EarlyStopping."""

    def __init__(self, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self._precision = keras.metrics.Precision()
        self._recall = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._precision.update_state(y_true, y_pred, sample_weight)
        self._recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self._precision.result()
        r = self._recall.result()
        return 2 * p * r / (p + r + 1e-7)

    def reset_state(self):
        self._precision.reset_state()
        self._recall.reset_state()


class MultiMetricCheckpoint(keras.callbacks.Callback):
    """Recall-gated checkpoint: saves only when val_recall >= best_recall.
    Within a recall tier, also saves on precision or loss improvement.
    max_loss guards against degenerate saves.
    """

    def __init__(self, filepath, start_from_epoch=0, max_loss=0.5):
        super().__init__()
        self.filepath = filepath
        self.start_from_epoch = start_from_epoch
        self.max_loss = max_loss
        self.best_recall = 0.0
        self.best_precision = 0.0
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        if epoch == self.start_from_epoch:
            self.model.save(self.filepath)
            return

        val_recall    = logs.get('val_recall', 0.0)
        val_precision = logs.get('val_precision', 0.0)
        val_loss      = logs.get('val_loss', float('inf'))

        if val_recall < self.best_recall:
            return

        if val_recall > self.best_recall and val_loss < self.max_loss:
            self.model.save(self.filepath)
            self.best_recall = val_recall
            self.best_loss = val_loss
            print()
            print(f'val_recall improved. val_recall:{val_recall:.4f}, saved')
            return

        if val_precision < self.best_precision:
            return

        if val_precision > self.best_precision and val_loss < self.max_loss:
            self.model.save(self.filepath)
            self.best_precision = val_precision
            self.best_loss = val_loss
            print()
            print(f'val_precision improved. val_precision:{val_precision:.4f}, saved')
            return

        if val_loss < self.best_loss:
            self.model.save(self.filepath)
            self.best_loss = val_loss
            print()
            print(f'val_loss improved. val_loss:{val_loss:.4f}, saved')


class BestF1Checkpoint(keras.callbacks.Callback):
    """Saves model when val F1 improves. Optionally stops training after `patience` epochs without improvement."""

    def __init__(self, filepath, start_from_epoch=0, patience=None):
        super().__init__()
        self.filepath = filepath
        self.start_from_epoch = start_from_epoch
        self.patience = patience
        self.best_f1 = 0.0
        self._wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        if epoch == self.start_from_epoch:
            self.model.save(self.filepath)
            return

        p  = logs.get('val_precision', 0.0)
        r  = logs.get('val_recall', 0.0)
        f1 = 2 * p * r / (p + r + 1e-7)
        if f1 > self.best_f1:
            self.model.save(self.filepath)
            self.best_f1 = f1
            self._wait = 0
            print()
            print(f'val_f1 improved to {f1:.4f} (p={p:.3f}, r={r:.3f}), saved')
        else:
            if self.patience is not None:
                self._wait += 1
                if self._wait >= self.patience:
                    print(f'\nEarly stopping: no val_f1 improvement for {self.patience} epochs.')
                    self.model.stop_training = True
