"""Some helpers for loading war and piece as a dataset.
We assume it is a massive sequence (which it is) and chop it up into
whatever length the caller wants."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import csv
import itertools

import numpy as np

from six.moves import xrange
from six.moves import urllib
from io import open

from rnndatasets import helpers


FILENAME = 'warandpeace.txt'
URL = 'https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt'


def _datapath():
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        FILENAME)


def _get_big_string():
    """gets war and peace as a big string"""
    if not hasattr(_get_big_string, 'BIG_STRING'):
        with open(helpers.maybe_download(
                _datapath(), URL), encoding='utf-8') as datafile:
            _get_big_string.BIG_STRING = datafile.read()
    return _get_big_string.BIG_STRING


def _word_spliterator(data):
    """Splits a string into a sequence of words, just breaking on
    white space at this stage.

    Could insert something fancier potentially.
    """
    return data.split()


def _gen_vocab(data, splitter, filename, most_common=None, repeat_thresh=None):
    """generates vocab dict and writes it to file.
    Optionally only keep the `most_common` most common words or those with
    at least `repeat_thresh` repetitions, has to be one or the other,
    `most_common` takes priority.
    """
    # go through and count
    counter = collections.Counter(splitter(data))
    if most_common:
        symbols = counter.most_common(most_common)
    elif repeat_thresh:
        pass
    else:
        symbols = counter.most_common()  # may as well sort them

    # and now assign ids
    # we are also going to have a GO symbol
    symbols.append(['<GO>', 0])
    vocab = {symb[0]: i for i, symb in enumerate(symbols)}
    # now write it to the file
    with open(filename, 'w', encoding='utf-8') as f:
        csvwrite = csv.writer(f)
        for sym in vocab:
            csvwrite.writerow([sym, vocab[sym]])
    return vocab


def _load_vocab(filename):
    """reads in a vocab file"""
    vocab = {}
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            vocab[row[0]] = int(row[1])
    return vocab


def get_vocab(level):
    """Gets the vocab as a dict of symbol -> int id.

    Args:
        level {`char` or `word`}: whether to get character or word
            level vocab.

    Returns:
        dict of all of the symbols.
    """
    filename = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        '{}-vocab.txt'.format(level))
    if not os.path.exists(filename):
        print('{} not found, generating vocab'.format(filename))
        # grab it as a big string
        data = _get_big_string()
        if level == 'char':
            split_func = iter
        else:
            split_func = _word_spliterator
        vocab = _gen_vocab(data, split_func, filename)
    else:
        vocab = _load_vocab(filename)
    return vocab


def _get_sequence(level):
    """gets the sequence and the vocab"""
    data_str = _get_big_string()
    vocab = get_vocab(level)
    if level == 'char':
        seq = [vocab[c] for c in data_str]
    else:
        seq = [vocab[word] for word in _word_spliterator(data_str)]
    return seq, vocab


def get_char_iter(sequence_length, batch_size, report_progress=False,
                  sequential=True, overlap=1, max_chars=None, start=0):
    """Gets an iterator to batches of war and peace data.

    Args:
        sequence_length (int): the length of each chunk. This the maximum
            length you want to unroll your net for.
        batch_size (int): how many sequences to return at once.
        report_progress (bool): whether or not to report back about
            how far through the data you're getting.
        sequential (bool): whether to make sure that each batch gets a
            consistent sequence. This is a good idea if you aren't resetting
            the state in between batches.
        overlap (int): overlap between sequences presented. Using an overlap
            of 1 can be handy if you need to use the shifted sequence as
            targets (which is likely to be most of the time).
        max_chars (int): the maximum number of chars to return,
            useful for testing things without using the full data.

    yields:
        list of `sequence_length` numpy int32 arrays, each with shape
            `[batch_size]`
        if `report_progress` is true, returns a tuple with the above as
            the second element and the fraction of progress through
            the data the first.
    """
    # first we actually have to load the data
    # w & p isn't huge, so the first thing we will do is just pull
    # the lot into memory
    wp_seq, _ = _get_sequence('char')
    if start:
        wp_seq = wp_seq[start:]
    if max_chars:
        wp_seq = wp_seq[:max_chars]
    num_chars = len(wp_seq)
    # this is potentially a little bit slow
    num_batches = num_chars // (sequence_length * batch_size)
    if sequential:
        step = sequence_length - overlap
    else:
        step = sequence_length * batch_size - overlap
    batchnum = 0
    for seq_start in xrange(start, start + num_chars, step):
        # gives us the starting position of the first sequence
        batch = []
        for b in xrange(batch_size):
            if sequential:
                batch_offset = b * num_chars // num_batches
            else:
                batch_offset = b
            batch.append(wp_seq[seq_start + batch_offset:seq_start + batch_offset + sequence_length])
        # we have all of the sequences, now we have to convert to time-major
        time_batch = []
        for seq in xrange(sequence_length):
            time_batch.append(
                np.array([batch[bnum][seq] for bnum in xrange(batch_size)]))
        if report_progress:
            yield (seq_start*batch_size+sequence_length) / num_chars, time_batch
        else:
            yield time_batch
        batchnum += 1
        if batchnum >= num_batches:
            break


def get_split_iters(sequence_length, batch_size, level='char', split=(0.8, 0.1, 0.1),
                    report_progress=False):
    """Gets separate iterators for training, validation and test data.

    Args:
        sequence_length: length of resulting sequences.
        batch_size: sequences per batch.
        level {'char', 'word'}: whether to use characters or words as the
            basic symbols.
        split: What portion of the data to use for train, validation and test. 
            Should be a sequence of floats, default (0.8,0.1,0.1). It is probably
            a good plan to make sure these sum to one.
        report_progress: whether the iterators should return a float as well indicating
            how much they have left.

    Returns:
        iterators over batches of data, with an overlap of 
            one symbol each time (so that it can be used for sequential
            language modelling.
    """
    seq, _ = _get_sequence(level)
    total_length = len(seq)
    train_start = 0
    num_valid = int(split[1] * total_length)
    valid_start = int(total_length * split[0]) + 1
    num_test = int(split[2] * total_length)
    test_start = valid_start + num_valid + 1

    print('~~~~({} training symbols)'.format(valid_start-1))
    print('~~~~({} validation symbols)'.format(num_valid))
    print('~~~~({} test symbols)'.format(num_test))
    
    if level == 'char':
        train_iter = get_char_iter(sequence_length,
                                   batch_size,
                                   report_progress=report_progress,
                                   start=train_start,
                                   max_chars=valid_start)
        valid_iter = get_char_iter(sequence_length,
                                   batch_size,
                                   report_progress=report_progress,
                                   start=valid_start,
                                   max_chars=num_valid+1)
        test_iter = get_char_iter(sequence_length,
                                  batch_size,
                                  report_progress=report_progress,
                                  start=test_start,
                                  max_chars=num_test+1)
                                   
        return train_iter, valid_iter, test_iter
    else:
        raise NotImplementedError('nope')
    
