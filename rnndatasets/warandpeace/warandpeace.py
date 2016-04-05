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
URL = 'http://www.gutenberg.myebook.bg/2/6/0/2600/2600.txt'


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
            vocab[row[0]] = row[1]
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


def get_char_iter(sequence_length, batch_size):
    """Gets an iterator to batches of war and peace data.

    Args:
        sequence_length (int): the length of each chunk. This the maximum
            length you want to unroll your net for.
        batch_size (int): how many sequences to return at once.

    yields:
        list of `sequence_length` numpy int32 arrays, each with shape
            `[batch_size]`
    """
    # first we actually have to load the data
    # w & p isn't huge, so the first thing we will do is just pull
    # the lot into memory
    wp_seq, _ = _get_sequence('char')
    num_chars = len(wp_seq)
    # this is potentially a little bit slow
    num_batches = num_chars // (sequence_length * batch_size)
    print('enough data for {} batches'.format(num_batches))
    for seq_start in xrange(0, sequence_length, sequence_length*batch_size):
        # gives us the starting position of the sequence
        batch = []
        for seq_pos in xrange(seq_start, seq_start + sequence_length):
            # for each position in the final sequence
            # grab a slice of batch_size many indices each seq_length apart
            batch.append(
                np.array(
                    wp_seq[seq_pos:seq_pos+batch_size*sequence_length:sequence_length],
                    dtype=np.int32))
        yield batch
