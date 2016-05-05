"""Gets mnist, does it sequentially, some number of pixels at
a time."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
            Labels are shape `[num_images, 1]` and of type int64.
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
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
    # and the labels
    with gzip.open(label_file) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return images, labels


def batch_iter(data, batch_size):
    """Makes an iterator which returns `batch_size` chunks of the input.
    If the batch size doesn't divide the data size perfectly, it will
    just ignore the last stuff.
    """
    pass


def get_test_valid_iters(batch_size):
    """Gets iterators for the train and valid sets.

    Returns:
        (test, valid): iterators for the test and valid data.
    """
    raw_data = get_data('train')
