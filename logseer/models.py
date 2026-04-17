import numpy as np
import keras
from keras.layers import (Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                          Dense, Dropout, Flatten, LSTM, Bidirectional,
                          Embedding, GRU, MultiHeadAttention, LayerNormalization)
from keras.models import Sequential
from keras import regularizers


class SelfAttention(keras.layers.Layer):
    """Multi-head self-attention with post-attention layer normalization.
    Placed after conv layers so attention operates on extracted features.
    """

    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm = LayerNormalization()

    def call(self, x):
        x = self.attention(x, x)
        return self.norm(x)


def getEmbeddingLayer(name, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, word_index=None):
    if name == 'vanilla':
        return Embedding(MAX_NB_WORDS, EMBEDDING_DIM)

    # Pre-trained embedding (GloVe / Word2Vec file path)
    embeddings_index = {}
    with open(name, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < MAX_NB_WORDS + 1:
            embedding_matrix[i] = embedding_vector

    return Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     trainable=False)


def getModel(model_name, embedding_layer=None, MAX_SEQUENCE_LENGTH=25000, EMBEDDING_DIM=300):
    models = {
        'simple':      lambda: simpleNN(MAX_SEQUENCE_LENGTH),
        'conv':        lambda: convNet(embedding_layer),
        'vgg':         lambda: vgglite(embedding_layer),
        'LogCNN':      lambda: LogCNN(embedding_layer),
        'LogCNNLite':  lambda: LogCNNLite(embedding_layer),
        'LogCNNattn':  lambda: LogCNNattn(embedding_layer),
        'LSTM':        lambda: LSTMModel(embedding_layer),
        'biLSTM':      lambda: biLSTMModel(embedding_layer),
        'biGRU':       lambda: biGRU(embedding_layer),
        'GRU':         lambda: plainGRU(embedding_layer),
    }
    if model_name not in models:
        raise ValueError(f'Unknown model: {model_name}. Options: {list(models.keys())}')
    return models[model_name]()


def convNet(embedding_layer):
    model = Sequential(name='ConvNet')
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=32))
    model.add(Conv1D(filters=256, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def vgglite(embedding_layer):
    model = Sequential(name='VGG16')
    model.add(embedding_layer)
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def LogCNN(embedding_layer):
    dr, ks = 2, 3
    model = Sequential(name='LogCNN')
    model.add(embedding_layer)
    for filters in [64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 64]:
        model.add(Conv1D(filters=filters, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
        if filters == 128:
            model.add(MaxPooling1D(pool_size=3))
            model.add(Dropout(0.1 if filters <= 128 else 0.2))
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def LogCNNLite(embedding_layer):
    """Primary model. Dilated 1D CNN optimized for recall on imbalanced log data.
    Dilated convolutions (rate=2) expand receptive field without extra parameters.
    GlobalAveragePooling makes detection position-invariant across the log sequence.
    """
    dr, ks = 2, 3
    model = Sequential(name='LogCNNLite')
    model.add(embedding_layer)
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Conv1D(filters=64, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def LogCNNattn(embedding_layer):
    """Dilated CNN with self-attention after conv layers."""
    dr, ks = 2, 3
    model = Sequential(name='LogCNNattn')
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv1D(filters=128, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv1D(filters=64, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv1D(filters=32, kernel_size=ks, dilation_rate=dr, activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(SelfAttention(num_heads=4, key_dim=32))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def simpleNN(MAX_SEQUENCE_LENGTH):
    model = Sequential(name='SimpleNN')
    model.add(Dense(256, activation='relu', input_shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(Dropout(0.2))
    for units in [256, 256, 256, 256, 128]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def biLSTMModel(embedding_layer):
    model = Sequential(name='biLSTM')
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def LSTMModel(embedding_layer):
    model = Sequential(name='LSTM')
    model.add(embedding_layer)
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def biGRU(embedding_layer):
    model = Sequential(name='biGRU')
    model.add(embedding_layer)
    model.add(Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def plainGRU(embedding_layer):
    model = Sequential(name='GRU')
    model.add(embedding_layer)
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
