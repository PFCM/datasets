"""Copy task -- people cite Hochreiter and Schmidhuber, although it seems
to have changed somewhat since then. We provide the following, as
per Henaff's Orthogonal RNN and Arjovsky's Unitary Evolution RNN.

Input is T+20 vector of ints:
    - first 10 sampled uniformly at random from [0-7]
    - next T-1 are 8 (blank symbol)
    - next is 9 (the go symbol)
    - final 10 are 8

Target is:
    - T+10 8s
    - the first 10 of the input, in the same order.

Goal is to minimise the average cross entropy across the sequence.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_online_sequences(sequence_length, batch_size,
                         pattern_length=10):
    """Gets tensors which produce new random examples every time
    they are evaluated.

    Args:
        sequence_length: the length of the time-lag the model has to
            remember the sequence for.
        batch_size: how many at once.
        pattern_length: the length of the pattern that has to be
            remembered and regurgitated.

    Returns:
        (data, targets): data is
            `[sequence_length + 2*pattern_length, batch_size, 1]`, targets
            are also `[sequence_length + 2*pattern_length, batch_size, 1]`.
    """
    # first we need a pattern to remember
    pattern = tf.random_uniform([pattern_length, batch_size, 1], maxval=8,
                                dtype=tf.int32)
    central_fillers = tf.fill([sequence_length-1, batch_size, 1], 8)
    go = tf.fill([1, batch_size, 1], 9)
    final_fillers = tf.fill([pattern_length, batch_size, 1], 8)
    inputs = tf.concat(axis=0, values=[pattern, central_fillers, go, final_fillers])

    fillers = tf.fill([sequence_length+pattern_length, batch_size, 1], 8)
    targets = tf.concat(axis=0, values=[fillers, pattern])

    return inputs, targets


if __name__ == '__main__':
    # just a quick look, before we write real tests
    data, targets = get_online_sequences(10, 2)

    sess = tf.Session()

    with sess.as_default():
        print(sess.run([data, targets]))
