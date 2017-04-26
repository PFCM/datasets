"""Addition task as used in the IRNN paper:

https://arxiv.org/pdf/1504.00941.pdf


At every time step, the input is a random signal and a mask signal.
Mask is zero at all timesteps except for two, when it has value one.
The task is to add the two random numbers present when the mask is
one.

We generate a fixed number of examples for a training and test set.
It is rather a lot, so this might get a bit gross, memory wise.

We will also try an online version with new random data every time.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_online_sequences(sequence_length, batch_size):
    """Gets tensor which constantly produce new random examples.

    Args:
        sequence_length: total length of the sequences.
        batch_size: how many at a time.

    Returns:
        (data, targets): data is `[sequence_length, batch_size, 2]` and targets
            are `[batch_size]`.
    """
    # getting the random channel is easy
    random_data = tf.random_uniform([sequence_length, batch_size, 1],
                                    minval=0.0, maxval=1.0)
    # now we need a random marker in each half of the data
    random_index_1 = tf.random_uniform([1, batch_size], minval=0,
                                       maxval=sequence_length//2,
                                       dtype=tf.int32)
    random_index_2 = tf.random_uniform([1, batch_size], minval=0,
                                       maxval=sequence_length//2,
                                       dtype=tf.int32)
    markers = tf.concat(axis=2, values=[tf.one_hot(random_index_1, sequence_length//2),
                            tf.one_hot(random_index_2, sequence_length//2)])
    markers = tf.transpose(markers)
    targets = tf.reduce_sum(random_data * markers,
                            axis=0)
    return tf.concat(axis=2, values=[random_data, markers]), tf.squeeze(targets)


def _gen_numpy_data(sequence_length, num_examples, seed=1991):
    """Make a rather large batch of examples. Results will be
    shape `[number, time_step, 2]` (2 because we have two input
    lines: data and mask)."""
    np.random.seed(seed)
    # as per Le and Hinton, the random data is in [0,1]
    random_data = np.random.sample((num_examples, sequence_length))
    random_data = random_data.astype(np.float32)
    # now we have to figure out where to put the marks
    mask = np.zeros((num_examples, sequence_length), dtype=np.float32)
    # get the first one, between 0 and sequence_length / 2
    first_marks = np.random.randint(sequence_length/2, size=(num_examples,))
    # and the second
    second_marks = np.random.randint(sequence_length/2, sequence_length,
                                     size=(num_examples,))
    eg_idx = np.arange(num_examples)
    mask[eg_idx, first_marks] = 1.0
    mask[eg_idx, second_marks] = 1.0

    # row wise dot them to get the targets
    targets = np.einsum('ij, ij->i', random_data, mask)

    return np.stack((random_data, mask), axis=-1), targets


def get_data_batches(sequence_length, batch_size, train_size, test_size,
                     num_epochs=None):
    """Gets tensors which batch up the data.
    Probably will have to unpack them to feed into the rnn.
    The test data is shuffled at each epoch.

    Args:
        sequence_length: the length of the sequences you want.
        batch_size: precisely how to batch it up.
        train_size: size of the training set
        test_size: size of the test set.

    Returns:
        train, test: each is a tuple of data, targets. Data is
        [sequence_length, batch_size, 2], targets are [batch_size].
    """
    all_data, all_targets = _gen_numpy_data(sequence_length,
                                            train_size+test_size)
    # now we have the data
    # put it into variables
    train_data = tf.Variable(all_data[:train_size, ...], trainable=False)
    train_targets = tf.Variable(all_targets[:train_size], trainable=False)
    test_data = tf.Variable(all_data[train_size:, ...], trainable=False)
    test_targets = tf.Variable(all_targets[train_size:], trainable=False)

    td_slice, tt_slice = tf.train.slice_input_producer(
        [train_data, train_targets], num_epochs=num_epochs)

    vd_slice, vt_slice = tf.train.slice_input_producer(
        [test_data, test_targets], num_epochs=None)

    td_batch, tt_batch = tf.train.batch([td_slice, tt_slice],
                                        batch_size=batch_size)
    # get the data into the right order
    td_batch = tf.transpose(td_batch, [1, 0, 2])

    vd_batch, vt_batch = tf.train.batch([vd_slice, vt_slice],
                                        batch_size=batch_size)
    vd_batch = tf.transpose(vd_batch, [1, 0, 2])

    return (td_batch, tt_batch), (vd_batch, vt_batch)


if __name__ == '__main__':
    """not putting real tests in because can't be bothered getting tensorflow
    on travis"""
    targets, data = get_online_sequences(10, 3)

    sess = tf.Session()

    with sess.as_default():
        print(sess.run([data, targets]))
