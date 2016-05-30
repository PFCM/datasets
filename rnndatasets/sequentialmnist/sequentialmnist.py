"""Gets mnist, does it sequentially, some number of pixels at
a time (probably one to be consistent with the lit)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import gzip

import numpy as np

from rnndatasets.helpers import maybe_download

FILE_NAMES = {
    'train': {
        'images': 'train-images-idx3-ubyte.gz',
        'labels': 'train-labels-idx1-ubyte.gz'
    },
    'test': {
        'images': 't10k-images-idx3-ubyte.gz',
        'labels': 't10k-labels-idx1-ubyte.gz'
    }
}

URL_BASE = 'http://yann.lecun.com/exdb/mnist/'

IMAGE_SIZE = 28  # size of mnist images


def _datapath():
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        FILENAME)


def get_data(dataset, num_images):
    """Gets the rawest mnist data + labels.

    Args:
        dataset (str): either `test` or `train`.

    Returns:
        (images, labels): both numpy arrays, with images of shape
            `[num_images, IMAGE_PIXELS*IMAGE_PIXELS]` rescaled to
            with pixel values rescaled to `[-0.5, 0.5]`.
            Labels are shape `[num_images]` and of type int64.
    """
    image_file = maybe_download(
        FILE_NAMES[dataset]['images'],
        URL_BASE + FILE_NAMES[dataset]['images'])
    label_file = maybe_download(
        FILE_NAMES[dataset]['labels'],
        URL_BASE + FILE_NAMES[dataset]['labels'])
    with gzip.open(image_file) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (255.0 / 2.0)) / 255.0
        # batch x time x features
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE, 1)
    # and the labels
    with gzip.open(label_file) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return data, labels


def batch_iter(data, batch_size, time_major=True):
    """Makes an iterator which returns `batch_size` chunks of the input.
    If the batch size doesn't divide the data size perfectly, it will
    just ignore the last stuff. Expects the input to be
    `[num x time x features]` (and all the same time)

    Args:
        data: `[num x time x features]` numpy array.
        batch_size: the size of the batches to yield.
        time_major: whether time or batch should be the first index of
            the yielded data.

    Yields:
        (data, labes): data is either `[batch_size x time x features]`
            or `[time x batch_size x features]` depending on time_major.
            Labels is `[batch_size]`
    """
    num_batches = data[0].shape[0] // batch_size
    inputs, labels = data
    for i in xrange(num_batches):
        batch_data = inputs[i*batch_size:(i+1)*batch_size, ...]
        if time_major:
            batch_data = np.transpose(batch_data, [1, 0, 2])
        batch_labels = labels[i*batch_size:(i+1)*batch_size, ...]
        yield (batch_data, batch_labels)


def get_iters(batch_size, time_major=True, shuffle=False):
    """Gets iterators for the train, test and valid sets.

    Args:
        batch_size: size of batches
        time_major: if true, the iterators will return (input, labels)
            where input is `[time x batch x features]`. If false
            `time` and `batch` will be swapped.

    Returns:
        (train, valid, test): iterators for the data.
    """
    raw_data = get_data('train', 60000)
    
    test_data = get_data('test', 10000)

    train_data = (raw_data[0][:-10000, ...], raw_data[1][:-10000])
    valid_data = (raw_data[0][-10000:, ...], raw_data[1][-10000:])

    if shuffle:
        # lazy way of making sure both data and labels gets the same shuffling
        rng_state = np.random.get_state()
        train_data[0] = np.random.shuffle(train_data[0])
        np.random.set_state(rng_state)
        train_data[1] = np.random.shuffle(train_data[1]

    return (batch_iter(train_data, batch_size, time_major),
            batch_iter(valid_data, batch_size, time_major),
            batch_iter(test_data, batch_size, time_major))
