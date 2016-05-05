"""Gets data read for penn treebank as per mikolov et al.
Much of the code here is based on:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import logging
import gzip
import tarfile

from six.moves import xrange

import numpy as np
# import tensorflow as tf

import rnndatasets.helpers as helpers


def _datapath(filename):
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        filename)


def _read(filename):
    """Gets the words out of a file, replacing \n with <eos>"""
    with open(filename) as f:
        return f.read().replace("\n", "<eos>").split()


def get_vocab(filename):
    """Builds the vocabulary from a given file"""
    words = _read(filename)
    counter = collections.Counter(words)
    # sort on both counts and alphabet (hopefully)
    sorted_words = sorted(counter.most_common())
    return dict(zip([item[0] for item in sorted_words], range(len(sorted_words))))


def check_files():
    """Double check the files exist, if not download them"""
    data_dir = _datapath('simple-examples/data')
    if not os.path.exists(data_dir):
        logging.info('Data not found')
        gz_filename = _datapath('simple-examples.tgz')
        with tarfile.open(
                helpers.maybe_download(
                    gz_filename,
                    'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'),
                'r:gz') as f:
            f.extractall(path=_datapath(''))
    return (os.path.join(data_dir, 'ptb.train.txt'),
            os.path.join(data_dir, 'ptb.valid.txt'),
            os.path.join(data_dir, 'ptb.test.txt'))


def _as_ids(filename, vocab):
    """Return the data in the file encoded using vocab"""
    data = _read(filename)
    return [vocab[symbol] for symbol in data]


def get_ptb_data():
    """Gets raw data.
    """
    train_file, valid_file, test_file = check_files()

    vocab = get_vocab(train_file)
    train_data = _as_ids(train_file, vocab)
    valid_data = _as_ids(valid_file, vocab)
    test_data = _as_ids(test_file, vocab)

    return train_data, valid_data, test_data, vocab


def batch_iterator(data, batch_size, num_steps):
    """Iterate in batches.

    Args:
        data: one of the data outputs from get_ptb_data
        batch_size: how big the batches are
        num_steps: how far we are looking back

    Yields:
        pairs of data, inputs and targets (targets are
            inputs shifted to the right by 1). Results in
            batch major order.
    """
    data = np.array(data, dtype=np.int32)

    data_len = len(data)
    num_batches = data_len // batch_size
    batched_data = np.zeros([batch_size, num_batches], dtype=np.int32)
    for i in xrange(batch_size):
        batched_data[i] = data[num_batches * i: num_batches * (i+1)]

    num_epochs = (num_batches - 1) // num_steps

    if num_epochs == 0:
        raise ValueError('batch size or num_steps too big')

    for i in xrange(num_epochs):
        x = batched_data[:, i*num_steps:(i+1)*num_steps]
        y = batched_data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x,y)
