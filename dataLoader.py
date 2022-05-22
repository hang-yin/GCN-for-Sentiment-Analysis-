import tensorflow as tf
import numpy as np
from tfRecordTools import *

def sampleFunction(inputFeature):
    """
    This is a description of the function
    
    Args:
        inputFeature - (np.ndarray) This is what the feature is
    Returns:
        result - (int) This is what is returned
    """

    result = 55
    return result

def build_reverse_word_index(dataset):
    """
    Convert the index back to words with proper accounting for 
    the special characters reserved at the beginning of the dictionary

    Args: 
        dataset - (keras.dataset) The dataset to use
    Returns:
        reverse_word_index - (dict) A dictionary mapping words to an integer index
    """
    word_index = dataset.get_word_index()

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2  # unknown
    word_index['<UNUSED>'] = 3
    return dict((value, key) for (key, value) in word_index.items())

def decode_review(text, dataset):
    """
    Uses build_reverse_word_index to decode original data format into text
    
    Args:
        text - (np.ndarray) The text to decode
        dataset - (keras.dataset) The dataset to use
    Returns:
        decoded_review - (string) The decoded review
    """
    return ' '.join([build_reverse_word_index(dataset).get(i, '?') for i in text])
