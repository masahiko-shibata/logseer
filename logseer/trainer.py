import pickle
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

from .models import getModel
from .checkpoints import MultiMetricCheckpoint


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
             model_save_path, epochs, batch_size, learning_rate, max_loss, retrain=False):
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
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall')])

    model.fit(train_data, train_labels,
              validation_data=(val_data, val_labels),
              epochs=epochs,
              verbose=1,
              batch_size=batch_size,
              callbacks=[MultiMetricCheckpoint(filepath=model_save_path, max_loss=max_loss)])

    model = load_model(model_save_path)
    tester.testModel(model, test_data, test_labels, threshold=0.5)
    return True


def train_sklearn(tokenizer, train_texts, test_texts, train_labels, test_labels, tester, *,
                  test_xgb=True, test_svm=False, test_rf=False, error_weight=2):
    train_mat = tokenizer.texts_to_matrix(train_texts, mode='tfidf')
    test_mat  = tokenizer.texts_to_matrix(test_texts,  mode='tfidf')

    if test_xgb:
        model = xgb.XGBClassifier(scale_pos_weight=error_weight)
        model.name = 'xgbModel'
        model.fit(train_mat, train_labels)
        pickle.dump(model, open('xgboost.pkl', 'wb'))
        tester.testModel(model, test_mat, test_labels)

    if test_svm:
        model = svm.SVC(probability=True)
        model.name = 'svmModel'
        model.fit(train_mat, train_labels)
        pickle.dump(model, open('svmModel.pkl', 'wb'))
        tester.testModel(model, test_mat, test_labels)

    if test_rf:
        model = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=50)
        model.name = 'rfModel'
        model.fit(train_mat, train_labels)
        pickle.dump(model, open('rfModel.pkl', 'wb'))
        tester.testModel(model, test_mat, test_labels)


def print_ensemble(tester):
    nn_row  = next((r for r in tester.stored if r[0] not in ('xgbModel', 'svmModel', 'rfModel')), None)
    xgb_row = next((r for r in tester.stored if r[0] == 'xgbModel'), None)
    if not (nn_row and xgb_row):
        return

    y    = np.array(nn_row[1])
    cnn  = np.array(nn_row[2])
    xgb_ = np.array(xgb_row[2])
    errors       = y == 1
    cnn_tp       = np.sum(errors & (cnn  == 1))
    xgb_tp       = np.sum(errors & (xgb_ == 1))
    both_tp      = np.sum(errors & (cnn  == 1) & (xgb_ == 1))
    either_tp    = np.sum(errors & ((cnn == 1) | (xgb_ == 1)))
    either_fp    = np.sum(~errors & ((cnn == 1) | (xgb_ == 1)))
    total_errors = np.sum(errors)
    ens_p  = either_tp / (either_tp + either_fp) if (either_tp + either_fp) > 0 else 0.0
    ens_r  = either_tp / total_errors if total_errors > 0 else 0.0
    ens_f1 = 2 * ens_p * ens_r / (ens_p + ens_r) if (ens_p + ens_r) > 0 else 0.0
    print()
    print('### Ensemble (CNN | XGB) ###')
    print()
    print(f'  Total errors   : {total_errors}')
    print(f'  CNN TP         : {cnn_tp}  (recall {cnn_tp/total_errors:.3f})')
    print(f'  XGB TP         : {xgb_tp}  (recall {xgb_tp/total_errors:.3f})')
    print(f'  Overlap (both) : {both_tp}')
    print(f'  CNN-only TP    : {cnn_tp - both_tp}')
    print(f'  XGB-only TP    : {xgb_tp - both_tp}')
    print(f'  Union TP       : {either_tp}')
    print(f'  Union FP       : {either_fp}')
    print(f'  Ensemble       : precision {ens_p:.3f}  recall {ens_r:.3f}  F1 {ens_f1:.3f}')


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