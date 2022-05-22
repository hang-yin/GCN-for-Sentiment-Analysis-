import tensorflow as tf
import numpy as np
import neural_strcutured_learning as nsl
from tfRecordTools import *
from dataLoader import *
from bertEmbeddings import *

class HParams(object):
    """
    Hyperparameters used for training.
    """

    def __init__(self):
        # dataset parameters
        self.num_classes = 2
        self.max_seq_length = 256
        self.vocab_size = 10000
        # neural graph learning parameters
        self.distance_type = nsl.configs.DistanceType.L2
        self.graph_regularization_multiplier = 0.1
        self.num_neighbors = 2
        # model architecture
        self.num_embedding_dims = 16
        self.num_lstm_dims = 64
        self.num_fc_units = 64
        # training parameters
        self.train_epochs = 20
        self.batch_size = 128
        # eval parameters
        self.eval_steps = None  # All instances in the test set are evaluated.

def create_example(word_vector, label, record_id):
    """
    Create tf.Example containing the sample's word vector, label, and ID.

    Args:
        word_vector: A `tf.Tensor` containing the word vector.
        label: A `tf.Tensor` containing the label.
        record_id: A `tf.Tensor` containing the record ID.
    Returns:
        An instance of `tf.train.Example`.
    """
    features = {
        'id': _bytes_feature(str(record_id)),
        'words': _int64_feature(np.asarray(word_vector)),
        'label': _int64_feature(np.asarray([label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_records(word_vectors, labels, record_path, starting_record_id):
    """
    Creates tf.Record files containing the word vectors and labels.
    
    Args:
        word_vectors: A `np.array` containing the word vectors.
        labels: A `np.array` containing the labels.
        record_path: The path to the tf.Record file to be created.
        starting_record_id: The ID of the first sample.
    Returns:
        The ID of the last sample.
    """
    record_id = int(starting_record_id)
    with tf.io.TFRecordWriter(record_path) as writer:
        for word_vector, label in zip(word_vectors, labels):
            example = create_example(word_vector, label, record_id)
            record_id = record_id + 1
            writer.write(example.SerializeToString())
    return record_id



# Data must be converted from integer to tensors before input
# Create an "max_length x num_reviews" matrix containing integers for the reviews
def make_dataset(file_path, HPARAMS, NBR_FEATURE_PREFIX, NBR_WEIGHT_SUFFIX, training=False):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """

  def pad_sequence(sequence, max_seq_length):
    """Pads the input sequence (a `tf.SparseTensor`) to `max_seq_length`."""
    pad_size = tf.maximum([0], max_seq_length - tf.shape(sequence)[0])
    padded = tf.concat(
        [sequence.values,
         tf.fill((pad_size), tf.cast(0, sequence.dtype))],
        axis=0)
    # The input sequence may be larger than max_seq_length. Truncate down if
    # necessary.
    return tf.slice(padded, [0], [max_seq_length])

  def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    # The 'words' feature is a variable length word ID vector.
    feature_spec = {
        'words': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }
    # We also extract corresponding neighbor features in a similar manner to
    # the features above during training.
    if training:
      for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i,
                                         NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.VarLenFeature(tf.int64)

        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, default_value=tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto, feature_spec)

    # Since the 'words' feature is a variable length word vector, we pad it to a
    # constant maximum length based on HPARAMS.max_seq_length
    features['words'] = pad_sequence(features['words'], HPARAMS.max_seq_length)
    if training:
      for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        features[nbr_feature_key] = pad_sequence(features[nbr_feature_key],
                                                 HPARAMS.max_seq_length)

    labels = features.pop('label')
    return features, labels

  dataset = tf.data.TFRecordDataset([file_path])
  if training:
    dataset = dataset.shuffle(10000)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(HPARAMS.batch_size)
  return dataset