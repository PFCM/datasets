"""Some kind of template matching -- looking for temporal shapes in a
sequence"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf


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
            sequence_lengths = tf.convert_to_tensor([sequence_length] * batch_size
                                                    , dtype=tf.int32)

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
        
