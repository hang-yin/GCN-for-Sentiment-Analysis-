import tensorflow as tf
import numpy as np
from tfRecordTools import *
from dataLoader import *
from bertEmbeddings import *
from train import *

def make_feed_forward_model(HPARAMS):
    """
    Builds a simple 2 layer feed forward neural network.
    """

    inputs = tf.keras.Input(
        shape=(HPARAMS.max_seq_length,),
        dtype='int64',
        name='words')

    embedding_layer = tf.keras.layers.Embedding(HPARAMS.vocab_size, 16)(inputs)
    pooling_layer = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
    dense_layer = tf.keras.layers.Dense(16, activation='relu')(pooling_layer)
    outputs = tf.keras.layers.Dense(1)(dense_layer)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_bilstm_model(HPARAMS):
    """
    Builds a bi-directional LSTM model.
    """

    inputs = tf.keras.Input(
        shape=(HPARAMS.max_seq_length,),
        dtype='int64',
        name='words')

    embedding_layer = tf.keras.layers.Embedding(HPARAMS.vocab_size,
                                                HPARAMS.num_embedding_dims)(
                                                    inputs)
    lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(HPARAMS.num_lstm_dims))(
        embedding_layer)

    dense_layer = tf.keras.layers.Dense(
        HPARAMS.num_fc_units,
        activation='relu')(
        lstm_layer)

    outputs = tf.keras.layers.Dense(1)(dense_layer)
    return tf.keras.Model(inputs=inputs, outputs=outputs)