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

def buildReverseWordIndex(dataset):
    """
    Convert the index back to words with proper accounting for 
    the special characters reserved at the beginning of the dictionary

    Args: 
        dataset - (keras.dataset) The dataset to use
    Returns:
        buildReverseWordIndex - (dict) A dictionary mapping words to an integer index
    """
    wordIndex = dataset.get_word_index()

    # The first indices are reserved
    wordIndex = {k: (v + 3) for k, v in wordIndex.items()}
    wordIndex['<PAD>'] = 0
    wordIndex['<START>'] = 1
    wordIndex['<UNK>'] = 2  # unknown
    wordIndex['<UNUSED>'] = 3
    return dict((value, key) for (key, value) in wordIndex.items())

def decodeReview(text, reverseWordIndex):
    """
    Uses build_reverse_word_index to decode original data format into text
    
    Args:
        text - (np.ndarray) The text to decode
        reverseWordIndex - (dict) The reverse word index to use
    Returns:
        decodedReview - (string) The decoded review
    """
    return ' '.join([reverseWordIndex.get(i, '?') for i in text])
