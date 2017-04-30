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
        axis=2, values=[noise, tf.zeros([sequence_length, batch_size, num_items])])
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
        primer_positions = tf.unstack(tf.transpose(tf.stack(primer_positions)))
    # one way or another we have positions given by
    # a list of `[batch_size]` int tensors.
    # in order to do a tf.select we need to turn them into
    # `[sequence_length, batch_size, num_features]` bool tensors
    primer_masks = []
    for idces in primer_positions:
        locations = tf.stack([tf.range(batch_size), idces])
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
        current_sequence = tf.where(mask, pattern, current_sequence)

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
            current_sequence = tf.where(mask, pattern, current_sequence)

        # now we have to pull out the targets
        # hopefully we can do this with some fancy slicing/gathering
        special_positions = [pos + offset for pos in primer_positions]

        # we have these, we may now need to zero out the rest of the sequence
        if not inbetween_noise:
            current_sequence = zero_inbetween(current_sequence,
                                              special_positions,
                                              dimensionality)

        target_positions = []
        for item in range(num_items):
            target_positions.append(
                special_positions[item][num_per_class*item:
                                        num_per_class*(item+1)])
        target_positions = tf.concat(axis=0, values=target_positions)
        targets = [tf.slice(
            current_sequence, [tpos, i, 0], [1, 1, dimensionality])
                   for i, tpos in enumerate(tf.unstack(target_positions))]
        targets = tf.stack([tf.squeeze(target) for target in targets])
    elif task == 'order':
        # the opposite, we need the last one to be a pattern (without primer
        # bits)
        # and the target is a one hot
        num_per_class = batch_size // num_items
        target_positions = []
        symbol_positions = [pos + offset for pos in primer_positions]

        if not inbetween_noise:
            current_sequence = zero_inbetween(current_sequence,
                                              symbol_positions,
                                              dimensionality)
        targets = []
        for item in range(num_items):
            target_positions.append(
                symbol_positions[item][num_per_class*item:
                                       num_per_class*(item+1)])
            targets.extend([item] * num_per_class)
        print(targets)
        target_positions = tf.concat(axis=0, values=target_positions)
        prompts = [tf.slice(
            current_sequence, [tpos, i, 0], [1, 1, dimensionality])
                   for i, tpos in enumerate(tf.unstack(target_positions))]
        stitch_idces = [tf.range(sequence_length-1), sequence_length-1]
        prompts = tf.concat(axis=1, values=prompts)

        zeros = tf.zeros([sequence_length-1, batch_size, num_features])

        prompts = tf.pad(prompts, [[0, 0], [0, 0], [0, num_items]])
        prompts = tf.concat(axis=0, values=[zeros, prompts])

        prompt_masks = tf.ones([1, batch_size, num_features])
        prompt_masks = tf.concat(axis=0, values=[zeros, prompt_masks])
        prompt_masks = tf.cast(prompt_masks, tf.bool)
        current_sequence = tf.where(prompt_masks, prompts, current_sequence)

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
        locations = tf.stack([batch_range, idx])
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
            axis=2,
            values=[mask,
             tf.ones([seq_len, batch_size, total_features-num_features])])
    return mask * seq


def get_continuous_binding_tensors(batch_size, sequence_length, num_items=2,
                                   dimensionality=8,
                                   real_patterns=False, min_keep_length=1,
                                   max_keep_length=None,
                                   inbetween_noise=False):
    """Round 2.

    Sequences are pretty much all zeros. There are `sequence_length` timesteps.
    At each timestep the input has `dimensionality` + `num_items` positions.
    The input vectors are divided into two regions, the pattern region from
    0 to `dimensionality`-1 and the label region, which is the remaining
    `num_items` positions.

    Targets are also a sequence of length `sequence_length`, with each input
    having dimension `dimensionality`.

    - `num_items` positions are chosen without replacement from
      [1, ..., `sequence_length`/2], denote A the set
    - if `max_keep_length` = None:
        - draw a set of random ints U from [min_keep_length, sequence_length]
        - let B = max(A+U, sequence_length-1)
    - else:
        - draw a set of random ints U from [min_keep_length, max_keep_length]
        - let B = max(A+U, sequence_length-1)
    - A and B are put together to form a list of pairs C
    - for each pair (a, b) in C:
        - set one of the label bits to one from timestep a to b inclusive.
        - generate a random pattern, insert it into the input sequence at time
          a+1 and into the target pattern at time b+1.

    It is possible for inputs to have to be output at the same time, in this
    case the target is the logical AND of the patterns.

    Arguments:
        batch_size: batch size of the result.
        sequence_length: total length of the sequences.
        num_items: how many different patterns there are to remember.
        dimensionality: the size of the patterns to remember and reconstruct.
        real_patterns (bool): whether the patterns are real numbers (True) or
            binary.
        min_keep_length: the minimum length of time a pattern will be held for.
        max_keep_length: the maximum length of time a pattern will be held for,
            which is key for determining the difficult of the task. If None,
            the (rather unlikely) maximum will be the length of the sequence
            minus 1.
    """
    # we can do this by making sequences of zeros and adding to them.
    input_features = num_items + dimensionality
    if inbetween_noise:
        inputs = tf.round(tf.random_uniform([sequence_length,
                                             batch_size,
                                             dimensionality]))
        inputs = tf.concat([inputs,
                            tf.zeros([sequence_length,
                                      batch_size,
                                      num_items])],
                           2)
    else:
        inputs = tf.zeros([sequence_length, batch_size, input_features])
    targets = tf.zeros([sequence_length, batch_size, dimensionality])
    # we need to choose some positions
    max_start = sequence_length//2
    # slightly easier to deal with each guy separately
    start_range = tf.range(max_start)
    starts = tf.stack([tf.random_shuffle(start_range)
                      for _ in range(batch_size)])
    starts = starts[:, :num_items]
    starts = tf.unstack(tf.transpose(starts))
    # and offsets
    offsets = [tf.random_uniform([batch_size], minval=min_keep_length+1,
                                 maxval=max_keep_length or sequence_length,
                                 dtype=tf.int32)
               for _ in range(num_items)]
    stops = [tf.clip_by_value(start + offset, 0, sequence_length-2)
             for start, offset in zip(starts, offsets)]
    # now we have what we need to start filling in bits
    # this is easier said than done
    # it seems the only way is to do it once per item in the batch :(
    # (mostly because tf.range only takes scalars and we need it to generate
    # indices)
    dim_range = tf.range(dimensionality)
    dim_ones = tf.ones([dimensionality], dtype=tf.int32)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        batch_seqs = []
        batch_inputs = []
        batch_targs = []
        for b_start, b_stop in zip(tf.unstack(start), tf.unstack(stop)):
            seq_idcs = tf.range(b_start, b_stop+1)
            seq_idcs = tf.stack(
                [seq_idcs, tf.ones_like(seq_idcs) * (input_features-i-1)])
            seq_idcs = tf.transpose(seq_idcs)
            label_bits = tf.sparse_to_dense(
                seq_idcs, [sequence_length, input_features], 1.0)
            batch_seqs.append(label_bits)

            # and let's make a pattern
            # probably it would be faster to do this a whole batch at a time
            # but my brain hurts already
            if not inbetween_noise:
                pattern = tf.random_uniform([dimensionality])
                if not real_patterns:
                    pattern = tf.round(pattern)
            else:
                pattern = inputs[b_start+1, i, :dimensionality]
            # now make two `[sequence_length, dimensionality]` with the pattern
            # in the right place for input and target
            # this should mean some careful construction of indices and two
            # calls to sparse_to_dense
            # actually, it's easy, we just need the seq pos on one side and
            # dim_range on the other
            input_idcs = tf.transpose(
                tf.stack([dim_ones * (b_start+1), dim_range]))
            b_input = tf.sparse_to_dense(
                input_idcs, [sequence_length, dimensionality], pattern)

            target_idcs = tf.transpose(
                tf.stack([dim_ones * (b_stop+1), dim_range]))
            b_target = tf.sparse_to_dense(
                target_idcs, [sequence_length, dimensionality], pattern)
            batch_inputs.append(b_input)
            batch_targs.append(b_target)

        batch_labels = tf.stack(batch_seqs)
        batch_labels = tf.transpose(batch_labels, [1, 0, 2])

        batch_inputs = tf.stack(batch_inputs)
        batch_inputs = tf.transpose(batch_inputs, [1, 0, 2])
        batch_inputs = tf.pad(batch_inputs, [[0, 0], [0, 0], [0, num_items]])

        if inbetween_noise:
            presence = tf.reduce_any(tf.cast(batch_inputs, tf.bool),
                                     axis=2)
            presence = tf.reshape(presence, [-1])
            flat_batch_inputs = tf.reshape(batch_inputs,
                                           [-1, input_features])
            flat_inputs = tf.reshape(inputs,
                                     [-1, input_features])
            inputs = tf.where(presence, flat_batch_inputs, flat_inputs)
            inputs = tf.reshape(inputs,
                                [sequence_length, batch_size, input_features])
        else:
            inputs += batch_labels + batch_inputs

        batch_targs = tf.stack(batch_targs)
        batch_targs = tf.transpose(batch_targs, [1, 0, 2])
        targets += batch_targs

    return inputs, tf.clip_by_value(targets, 0.0, 1.0)


if __name__ == '__main__':
    sess = tf.Session()

    sequ, targets = (sess.run(get_recognition_tensors(2, 10, 2, task='order')))
    print('sequence:')
    print(sequ)
    print('targets:')
    print(targets)
