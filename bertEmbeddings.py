import tensorflow as tf
import numpy as np
from tfRecordTools import *
from dataLoader import *

def createBertEmbeddingExample(word_vector, record_id, dataset, encoder, preprocessor):
    """
    Create tf.Example containing the sample's embedding and its ID.
    
    
    """

    text = decode_review(word_vector, dataset)

    # Shape = [batch_size,].
    sentence_embedding = encoder(preprocessor(tf.reshape(text, shape=[-1, ])))['pooled_output']
    
    # Flatten the sentence embedding back to 1-D.
    sentence_embedding = tf.reshape(sentence_embedding, shape=[-1])
    
    features = {
        'id': _bytes_feature(str(record_id)),
        'embedding': _float_feature(sentence_embedding.numpy())
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def createBertEmbedding(word_vectors, output_path, starting_record_id, dataset, encoder, preprocessor):
    record_id = int(starting_record_id)
    with tf.io.TFRecordWriter(output_path) as writer:
        for word_vector in word_vectors:
            example = createBertEmbeddingExample(word_vector, record_id, dataset, encoder, preprocessor)
            record_id = record_id + 1
            writer.write(example.SerializeToString())
    return record_id
