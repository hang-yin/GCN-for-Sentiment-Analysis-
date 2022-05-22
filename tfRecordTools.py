import tensorflow as tf
import numpy as np
from tensorflow import keras


def int64Feature(value):
    """
    Returns int64 tf.train.Feature.

    Args:
        value - (np.ndarray) array of ints
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def bytesFeature(value):
    """
    Returns bytes tf.train.Feature.

    Args:
        value - (string) string
    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def floatFeature(value):
    """
    Returns float tf.train.Feature.

    Args:
        value - (np.ndarray) array of floats

    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))
