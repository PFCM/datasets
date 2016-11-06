"""Collection of Crowley and others that I put together for entertainment
purposes. May include Thee Temple ov Psychic Youth, sorry Genesis.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.platform import gfile

from rnndatasets.helpers import maybe_download, batch_iterator


URL = "http://pfcmathews.com/assets/occulttraining.txt"


def _read_chars(filename):
    with gfile.GFile(maybe_download(filename, URL), "r") as f:
        return list(f.read())


def _build_vocab(filename):
    data = _read_chars(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, other = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['<GO>'] = len(word_to_id)

    return word_to_id


def _file_to_char_ids(filename, word_to_id):
    data = _read_chars(filename)
    return [word_to_id[word] for word in data]


def _get_filename():
    """Gets a default filename, based on the directory this file is in"""
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, 'occulttraining.txt')


def labels_to_one_hot(l, num):
    num_labels = l.shape[0]
    index_offset = np.arange(num_labels) * num
    labels_one_hot = np.zeros((num_labels, num))
    labels_one_hot.flat[index_offset + l.ravel()] = 1
    return labels_one_hot


def occult_raw_data(path=None):
    """Loads the occult file.
    Returns:
      tuple (train_data, valid_data, test_data, vocab)
      each can be passed to the iterator function (except vocab)
    """
    if not path:
        path = _get_filename()
    word_to_id = _build_vocab(path)
    all_data = _file_to_char_ids(path, word_to_id)
    # it's just one massive sequence!
    # now we need to slice it up
    # into train, test and valid
    num_chars = len(all_data)
    all_data = np.array(all_data)
    train_end = num_chars//100 * 98
    train = all_data[:train_end, ...]
    test_end = train_end + num_chars//100
    test = all_data[train_end:test_end, ...]
    valid = all_data[test_end:, ...]

    return train, valid, test, word_to_id
