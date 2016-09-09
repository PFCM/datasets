"""
Tasks that require something close to variable binding.

At least, the ability to store things in a particular part of memory and
compare new inputs to them.

More or less generalisations of the temporal order task from the LSTM paper.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_recognition_tensors(batch_size, sequence_length, num_items=1,
                            dimensionality=8, task='recall', offset=1,
                            inbetween_noise=True, real=False):
    """Produces tensors for the following task:

        - inputs are sequences of length `sequence_length` with
          `dimensionality` + `num_items` features.
        - the first `dimensionality` bits are almost always random noise.
        - periodically we insert one of a number of special primer symbols,
          these indicate that the following value is one that needs to be
          remembered.
        - if `task` == 'recall', at the end we show it a primer symbol again
          and it has to output the symbol. May get weird if num_items does not
          divide batch_size.
        - if `task` == 'order', we keep showing it random patterns but the
          last one is one of the patterns it has seen before, and the goal is
          to output the (one-hot vector) indicating which symbol it was. Might
          get strange if num_items+1 does not divide batch size (because there
          need also to be negative examples).

        Also keep in mind that the shape of the targets depends on the task.
        If task is recall, the target tensor will be
        `[batch_size, dimensionality]` floats
        If task is order, the target tensor will be `[batch_size]` ints (and
        the total number of possible targets will be num_items + 1).
    """
    num_features = dimensionality + num_items

    noise = tf.random_uniform([sequence_length, batch_size, dimensionality])
    noise = tf.concat(
        2, [noise, tf.zeros([sequence_length, batch_size, num_items])])
    if not real:
        binary_patterns = tf.round(noise)
    else:
        binary_patterns = noise

    primer_patterns = [
        tf.tile(
            tf.reshape(
                tf.one_hot(num_features-i-1, num_features),
                [1, 1, num_features]),
            [sequence_length, batch_size, 1]) + binary_patterns
        for i in range(num_items)]

    if num_items == 1:
        # then just choose a single random
        primer_positions = [tf.random_uniform(
            [batch_size], 0, sequence_length-2, dtype=tf.int32)]
    else:
        # make sure we choose without replacement
        # this is kind of horrifying
        all_positions = tf.range(sequence_length-offset-1)
        primer_positions = [
            tf.slice(tf.random_shuffle(all_positions), [0], [num_items])
            for _ in range(batch_size)]
        primer_positions = tf.unpack(tf.transpose(tf.pack(primer_positions)))
    # one way or another we have positions given by
    # a list of `[batch_size]` int tensors.
    # in order to do a tf.select we need to turn them into
    # `[sequence_length, batch_size, num_features]` bool tensors
    primer_masks = []
    for idces in primer_positions:
        locations = tf.pack([tf.range(batch_size), idces])
        # we do this so that the indices are ordered
        locations = tf.transpose(locations)
        batch_mask = tf.sparse_to_dense(
            locations, [batch_size, sequence_length],
            True, default_value=False)
        primer_masks.append(tf.transpose(batch_mask))
    # now we have a list of `[sequence_length, batch_size]` masks for each
    # primer, so we will tile it to fill out the features
    primer_masks = [
        tf.tile(tf.expand_dims(mask, -1), [1, 1, num_features])
        for mask in primer_masks]

    # and now we should be in a position to nest a whole bunch of selects
    current_sequence = binary_patterns
    for mask, pattern in zip(primer_masks, primer_patterns):
        current_sequence = tf.select(mask, pattern, current_sequence)

    # now we have to figure out how to go ahead and put the appropriate
    # pattern into the last positions and come up with the required targets
    if task == 'recall':
        # we want the last one to have a primer bit active
        # we can get the target by slicing the actual sequence
        # I am terrified of the below
        num_per_class = batch_size // num_items
        target_masks = [tf.tile(
            tf.reshape(
                tf.sparse_to_dense(
                    [[sequence_length-1, num_per_class*i + j]
                     for j in range(num_per_class)],
                    [sequence_length, batch_size],
                    True, default_value=False),
                [sequence_length, batch_size, 1]),
            [1, 1, num_features])
                        for i in range(num_items)]

        for mask, pattern in zip(target_masks, primer_patterns):
            current_sequence = tf.select(mask, pattern, current_sequence)

        # now we have to pull out the targets
        # hopefully we can do this with some fancy slicing/gathering
        special_positions = [pos + offset for pos in primer_positions]
        
        # we have these, we may now need to zero out the rest of the sequence
        if not inbetween_noise:
            current_sequence = zero_inbetween(current_sequence, special_positions,
                                              dimensionality)

        target_positions = []
        for item in range(num_items):
            target_positions.append(
                special_positions[item][num_per_class*item:
                                        num_per_class*(item+1)])
        target_positions = tf.concat(0, target_positions)
        targets = [tf.slice(
            current_sequence, [tpos, i, 0], [1, 1, dimensionality])
                   for i, tpos in enumerate(tf.unpack(target_positions))]
        targets = tf.pack([tf.squeeze(target) for target in targets])
    elif task == 'order':
        # the opposite, we need the last one to be a pattern (without primer
        # bits)
        # and the target is a one hot
        num_per_class = batch_size // num_items
        target_positions = []
        symbol_positions = [pos + offset for pos in primer_positions]

        if not inbetween_noise:
            current_sequence = zero_inbetween(current_sequence, symbol_positions,
                                              dimensionality)
        targets = []
        for item in range(num_items):
            target_positions.append(
                symbol_positions[item][num_per_class*item:
                                       num_per_class*(item+1)])
            targets.extend([item] * num_per_class)
        print(targets)
        target_positions = tf.concat(0, target_positions)
        prompts = [tf.slice(
            current_sequence, [tpos, i, 0], [1, 1, dimensionality])
                   for i, tpos in enumerate(tf.unpack(target_positions))]
        stitch_idces = [tf.range(sequence_length-1), sequence_length-1]
        prompts = tf.concat(1, prompts)

        zeros = tf.zeros([sequence_length-1, batch_size, num_features])

        prompts = tf.pad(prompts, [[0, 0], [0, 0], [0, num_items]])
        prompts = tf.concat(0, [zeros, prompts])

        prompt_masks = tf.ones([1, batch_size, num_features])
        prompt_masks = tf.concat(0, [zeros, prompt_masks])
        prompt_masks = tf.cast(prompt_masks, tf.bool)
        current_sequence = tf.select(prompt_masks, prompts, current_sequence)

        targets = tf.constant(targets, dtype=tf.int64)

    return current_sequence, targets


def zero_inbetween(seq, keep_idcs, num_features):
    """Zeros out a sequence in between specified points (only touches specified 
    features)"""
    # we'll just multiply elementwise with a binary mask
    # which means we'll likely be using sparse_to_dense yet again
    seq_len, batch_size, total_features = seq.get_shape().as_list()
    batch_range = tf.range(batch_size)
    seqs = []
    for idx in keep_idcs:
        locations = tf.pack([batch_range, idx])
        locations = tf.transpose(locations)
        idx_mask = tf.sparse_to_dense(locations,
                                      [batch_size, seq_len],
                                      1.0)
        seqs.append(tf.transpose(idx_mask))
    seqs = tf.expand_dims(tf.add_n(seqs), -1)
    mask = tf.tile(seqs, [1, 1, num_features])

    total_features = seq.get_shape().as_list()[-1]
    if total_features > num_features:
        mask = tf.concat(
            2,
            [mask,
             tf.ones([seq_len, batch_size, total_features-num_features])])
        
    
    return mask * seq


if __name__ == '__main__':
    sess = tf.Session()

    seq, targets = (sess.run(get_recognition_tensors(2, 10, 2, task='order')))
    print('sequence:')
    print(seq)
    print('targets:')
    print(targets)
