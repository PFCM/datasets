"""Addition task as used in the IRNN paper:

https://arxiv.org/pdf/1504.00941.pdf


At every time step, the input is a random signal and a mask signal.
Mask is zero at all timesteps except for two, when it has value one.
The task is to add the two random numbers present when the mask is
one.

We generate a fixed number of examples for a training and test set.
It is rather a lot, so this might get a bit gross, memory wise.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _gen_numpy_data(sequence_length, num_examples, seed=1001):
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
    train, test = get_data_batches(100, 50, 1000, 100, 1)
    # we would then expect to get 20 batches of data before an exception.
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    data, targets = sess.run(test)
    assert data.shape == (100, 50, 2)
    assert targets.shape == (50,)

    try:
        step = 0
        while not coord.should_stop():
            data, targets = sess.run(train)
            assert data.shape == (100, 50, 2)
            assert targets.shape == (50,)
            step += 1
    except tf.errors.OutOfRangeError:
        assert step == 20
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    print('done')
