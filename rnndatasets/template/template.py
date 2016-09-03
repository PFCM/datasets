"""Some kind of template matching -- looking for temporal shapes in a
sequence"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf


def online_saw_tensors(batch_size, sequence_length, block_size, stddev=1.0):
    """Returns tensors for the following binary classification problem on 1D
    sequences of reals:
        - class 0:
            - gaussian noise
        - class 1:
            - gaussian noise
            - at a random location, a saw from -stddev to stddev over
                `block_size` steps.

    So the job is to look for the presence or absence of a predictable shape.
    A smaller shape should be harder to predict by calculating statistics over
    the sequences (ie. won't throw off the standard dev). One way to solve the
    problem is to try and predict the current input + the saw step every time
    and check how close the prediction is.

    Args:
        batch_size: how many at once.
        sequence_length: the total length of all the sequences.
        block_size: the length of the saw.
    """
    with tf.variable_scope('saw_data'):
        sequences = tf.random_normal([batch_size, sequence_length, 1],
                                     stddev=stddev, name='noise')
        sawtooth = tf.linspace(-stddev, stddev, block_size, name='saw')
        # now we have to choose some starting positions
        starts = tf.random_uniform(
            [batch_size//2], minval=0, maxval=sequence_length-block_size,
            dtype=tf.int32, name='starts')
        # let's pad the saws with zeros to make them the right shape
        # there is a tf function for this, but we have to be a little bit
        # tricky to get the shapes right
        pads = [tf.pack([start, sequence_length - start - block_size])
                for start in tf.unpack(starts)]

        sawtooth = tf.pack([tf.pad(sawtooth, tf.expand_dims(pad, 0))
                            for pad in pads], name='saws')
        sawtooth = tf.concat(0, [sawtooth, tf.zeros_like(sawtooth)])
        # and now figure out how to combine the two
        # looks like we have to do some weird packing and unpacking again
        # because it only does one range at a time
        positions = [tf.range(start, limit=start + block_size)
                     for start in tf.unpack(starts)]

        mask = [tf.sparse_to_dense(position, [sequence_length],
                                   True, default_value=False)
                for position in positions]
        mask = tf.pack(mask)
        falses = tf.cast(tf.zeros([batch_size//2, sequence_length]), tf.bool)
        mask = tf.concat(0, [mask, falses])

        # make sure it all lines up
        sawtooth = tf.expand_dims(sawtooth, -1)
        mask = tf.expand_dims(mask, -1)

        data = tf.transpose(tf.select(mask, sawtooth, sequences), [1, 0, 2])

        # and labels are easy
        labels = tf.concat(0, [tf.ones_like(starts), tf.zeros_like(starts)])
        labels = tf.cast(labels, tf.float32)

        return data, labels


def online_block_tensors(batch_size, sequence_length, block_size,
                         one_chance=0.3, variable_sequence_length=True):
    """Returns tensors for the block problem, classifying tensors as either
    having or not having large blocks of ones in them. Otherwise the sequences
    are just random ones or zeros (with probability one_chance of being 1).

    The integrity of the sequences depends on sequence_length, block_size and
    one_chance -- the the block size is small and one_chance is high there
    is a good chance that there will be false negatives in the labels.

    At this stage only handles even batch sizes.

    Args:
        batch_size: the size of the tensors returned along the batch dimension.
        sequence_length: maximum length of the resulting sequences.
        block_size: size of the block that needs to be found.
        one_chance (optional): the likelihood of an element of the sequence
            that is not part of a block being a 1.
        variable_sequence_length (optional): if True (default) then sequences
            have lengths randomly chosen between sequence_length/2 and
            sequence_length.

    Return:
        Two, possibly three tensors:
            - data, `[sequence_length, batch_size, 1]` the input sequences.
            - targets, `[batch_size]` the labels, either 0 or 1.
            - lengths, `[batch_size]` tensor of ints, only returned if
                variable_sequence_length is True, contains the lengths of the
                sequences.
    """
    with tf.variable_scope('input'):
        # first just make some random zeros and ones
        sequences = tf.convert_to_tensor(one_chance)
        sequences += tf.random_uniform([sequence_length, batch_size, 1])
        sequences = tf.floor(sequences)
        # now we have to decide where to put in some blocks of ones
        if variable_sequence_length:
            sequence_lengths = tf.random_uniform(
                [batch_size], minval=sequence_length // 2,
                maxval=sequence_length, dtype=tf.int32)
        else:
            sequence_lengths = tf.convert_to_tensor(
                [sequence_length] * batch_size, dtype=tf.int32)

        # generate positions for our 'objects' for half the sequence
        start_positions = [
            tf.random_uniform(
                [], minval=0, maxval=seq_len-block_size,
                dtype=tf.int32)
            for seq_len in tf.unpack(sequence_lengths)[:batch_size//2]]

        ones = tf.ones_like(sequences)

        # now we want to select between the random or the ones
        # tf.select seems like the appropriate candiate, we just have to
        # come up with some kind of binary mask tensor as well
        indices = [
            tf.range(tf.squeeze(start), limit=tf.squeeze(start)+block_size)
            for start in start_positions]
        mask = [
            tf.sparse_to_dense(index, [sequence_length], True,
                               default_value=False)
            for index in indices]
        mask = [tf.expand_dims(seq, 1) for seq in mask]
        mask = tf.pack(mask)
        mask = tf.transpose(mask, [1, 0, 2])

        # add in some falses so we leave half the sequences untouched
        mask = tf.concat(1, [mask, tf.constant(False, shape=mask.get_shape())])

        # the labels are easy
        labels = tf.concat(0, [tf.ones([batch_size//2]),
                               tf.zeros([batch_size//2])])

        return tf.select(mask, ones, sequences), sequence_lengths, labels


if __name__ == '__main__':
    sess = tf.Session()
    print(sess.run(online_saw_tensors(4, 10, 5)))
