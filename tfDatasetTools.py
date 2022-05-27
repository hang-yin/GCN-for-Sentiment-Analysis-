import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


def decodeZeroDimTensor(yourTensor):
    """
    Takes a tensor of zero dim and returns the string stored inside
    Args:
        yourTensor - (tf.Tensor) The input tensor you want to convert
    Returns:
        yourString - (string) The string from the decoded tensor
    """
    noSpecial = tf.strings.regex_replace(
        yourTensor,
        "<[^>]+>",
        " ")
    return np.array(noSpecial).reshape((1,))[0].decode("utf-8")


def convertTakeDataset(takeDataset):
    """
    Converts your takeDataset into features and labels
    Args:
        takeDataset - (TakeDataset) the TakeDataset that contains some number of examples from your initial dataset
    Returns:
        features - (list) the list of features
        labels - (list) the list of labels
    """
    labels = []
    features = []
    for text, label in takeDataset:
        labels.append(int(np.array(label)))
        features.append(decodeZeroDimTensor(text))

    return features, labels
