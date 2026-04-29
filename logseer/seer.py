import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from .jde_loader import JDELoader

OUTCOME_OK      = 'OK'
OUTCOME_ALERT   = 'ALERT'
OUTCOME_RESTART = 'RESTART'


class Seer:

    @classmethod
    def from_files(cls, nn_model_path, tokenizer_path, xgb_path=None, **kwargs):
        """Load models from file paths and return a Seer instance."""
        nn_model = load_model(nn_model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        xgb_model = None
        if xgb_path and os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                xgb_model = pickle.load(f)
        return cls(nn_model, tokenizer, xgb_model, **kwargs)

    def __init__(self, nn_model, tokenizer, xgb_model=None, *,
                 numchar=3000, max_sequence_length=26000,
                 nn_threshold=0.5, xgb_threshold=0.5,
                 loader_class=None):
        self.nn_model            = nn_model
        self.tokenizer           = tokenizer
        self.xgb_model           = xgb_model
        self.numchar             = numchar
        self.max_sequence_length = max_sequence_length
        self.nn_threshold        = nn_threshold
        self.xgb_threshold       = xgb_threshold
        self.loader              = (loader_class or JDELoader)()

    def predict(self, data_dir):
        """Run prediction on all log sets in data_dir.

        data_dir can contain:
          - subdirectories, one per log set (e.g. 0001/, 0002/, ...)
          - log files directly (treated as a single log set)

        Returns a list of dicts with keys:
          name, nn_prob, xgb_prob, nn_pred, xgb_pred, or_pred, and_pred, outcome
        """
        file_groups = self._collect_file_groups(data_dir)
        if not file_groups:
            return []

        texts_with_names = self.loader.gen_texts(file_groups, numchar=self.numchar)
        names = [t[1] for t in texts_with_names]
        texts = [t[0] for t in texts_with_names]

        seqs     = self.tokenizer.texts_to_sequences(texts)
        data     = np.array(pad_sequences(seqs, maxlen=self.max_sequence_length), dtype=np.int32)
        nn_probs = self.nn_model.predict(data, verbose=0).flatten().tolist()
        nn_preds = [1 if p >= self.nn_threshold else 0 for p in nn_probs]

        xgb_probs = [None] * len(texts)
        xgb_preds = [0]    * len(texts)
        if self.xgb_model is not None:
            mat       = self.tokenizer.texts_to_matrix(texts, mode='tfidf')
            xgb_probs = list(self.xgb_model.predict_proba(mat)[:, 1])
            xgb_preds = [1 if p >= self.xgb_threshold else 0 for p in xgb_probs]

        results = []
        for name, nn_p, nn_pred, xgb_p, xgb_pred in zip(names, nn_probs, nn_preds, xgb_probs, xgb_preds):
            or_pred  = int(bool(nn_pred or xgb_pred))
            and_pred = int(bool(nn_pred and xgb_pred))
            if and_pred:
                outcome = OUTCOME_RESTART
            elif or_pred:
                outcome = OUTCOME_ALERT
            else:
                outcome = OUTCOME_OK
            results.append({
                'name':     name,
                'nn_prob':  nn_p,
                'xgb_prob': xgb_p,
                'nn_pred':  nn_pred,
                'xgb_pred': xgb_pred,
                'or_pred':  or_pred,
                'and_pred': and_pred,
                'outcome':  outcome,
            })
        return results

    def _collect_file_groups(self, data_dir):
        entries = sorted(e for e in os.listdir(data_dir) if not e.startswith('.'))
        has_subdirs = any(os.path.isdir(os.path.join(data_dir, e)) for e in entries)
        file_groups = []
        if has_subdirs:
            for name in entries:
                path = os.path.join(data_dir, name)
                if not os.path.isdir(path):
                    continue
                files = self._read_dir(path)
                if files:
                    file_groups.append([files, name])
        else:
            files = self._read_dir(data_dir)
            if files:
                file_groups.append([files, os.path.basename(os.path.abspath(data_dir))])
        return file_groups

    def _read_dir(self, path):
        files = []
        for fname in sorted(os.listdir(path)):
            if fname.startswith('.'):
                continue
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                with open(fpath, encoding='utf-8', errors='ignore') as f:
                    files.append(f.read())
        return files


def print_results(results):
    if not results:
        print('No results.')
        return
    w = max(len(r['name']) for r in results)
    has_xgb = results[0]['xgb_prob'] is not None
    if has_xgb:
        print(f'  {"Set":<{w}}  {"NN_prob":>8}  {"XGB_prob":>8}  {"OR":>5}  {"AND":>5}  Note')
        print('  ' + '-' * (w + 42))
        for r in results:
            print(f'  {r["name"]:<{w}}  {r["nn_prob"]:>8.4f}  {r["xgb_prob"]:>8.4f}'
                  f'  {"ERR" if r["or_pred"] else "ok":>5}  {"ERR" if r["and_pred"] else "ok":>5}'
                  f'  {r["outcome"]}')
    else:
        print(f'  {"Set":<{w}}  {"NN_prob":>8}  {"NN":>5}  Note')
        print('  ' + '-' * (w + 26))
        for r in results:
            print(f'  {r["name"]:<{w}}  {r["nn_prob"]:>8.4f}  {"ERR" if r["nn_pred"] else "ok":>5}  {r["outcome"]}')