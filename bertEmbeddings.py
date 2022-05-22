import tensorflow as tf
import numpy as np
from tfRecordTools import *
from dataLoader import *

def createBertEmbeddingExample(wordVector, recordID, reverseWordIndex, encoder, preprocessor):
    """
    Create tf.Example containing the sample's embedding and its ID.
    
    Args:
        wordVector - (np.ndarray) the text to decode
        recordId - (int) ID of the sample
        reverseWordIndex - (dict) The reverse word index to use
        encoder - (string) encoder name
        preprocessor - (string) preprocessor name
    Returns:
        example - (tf.Example) tf.Example containing the sample's embedding and its ID
    """

    text = decodeReview(wordVector, reverseWordIndex)

    # Shape = [batch_size,].
    sentenceEmbedding = encoder(preprocessor(tf.reshape(text, shape=[-1, ])))['pooled_output']
    
    # Flatten the sentence embedding back to 1-D.
    sentenceEmbedding = tf.reshape(sentenceEmbedding, shape=[-1])
    
    features = {
        'id': bytesFeature(str(recordID)),
        'embedding': floatFeature(sentenceEmbedding.numpy())
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def createBertEmbedding(wordVectors, outputPath, startingRecordId, reverseWordIndex, encoder, preprocessor):
    """
    Create full set of BERT embeddings

    Args:
        wordVectors - (np.ndarray) all text to decode
        outputPath - (string) path to output file
        startingRecordId - (int) ID of the first sample
        reverseWordIndex - (dict) The reverse word index to use
        encoder - (string) encoder name
        preprocessor - (string) preprocessor name
    Returns:
        recordID - (int) ID of the last sample
    """
    recordID = int(startingRecordId)
    with tf.io.TFRecordWriter(outputPath) as writer:
        for word_vector in wordVectors:
            example = createBertEmbeddingExample(word_vector, recordID, reverseWordIndex, encoder, preprocessor)
            recordID = recordID + 1
            writer.write(example.SerializeToString())
    return recordID
