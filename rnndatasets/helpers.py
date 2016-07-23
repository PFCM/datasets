"""Provides a package for handling datasets consistently, could be helpful.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import shutil

from six.moves import xrange
from six.moves import urllib

import numpy as np


def maybe_download(filepath, url):
    """Checks to see if a file exists. If not, downloads it from the given url.

    Args:
      filepath (str): the path to check.
      url (str): the url which we will attempt to retrieve if the file is not
        present.

    Returns:
      str: the path to the file
    """
    # let's not use tensorflow for this
    # so that it is a bit more flexible
    if not os.path.exists(filepath):
        # dl into a tempfile in case it faoils
        with tempfile.NamedTemporaryFile() as tmpfile:
            temp_file_name = tmpfile.name
            urllib.request.urlretrieve(url, temp_file_name)
            shutil.copy(temp_file_name, filepath)
            print('Downloaded {}'.format(filepath))
    return filepath


def batch_iterator(data, batch_size, num_steps):
    """Iterate in batches. For next-step prediction.

    Args:
        data: a really big sequence
        batch_size: how big the batches are
        num_steps: how far we are looking back

    Yields:
        pairs of data, inputs and targets (targets are
            inputs shifted to the right by 1). Results in
            batch major order (shape [batch, time]) so you will probably
            have to transpose them to feed into an RNN.
    """
    # lets see if the data is an np array, if it isn't we'll treat it as
    # integer labels
    try:
        num_features = data.shape[1]
    except AttributeError:
        data = np.array(data, dtype=np.int32)
        num_features = 0

    data_len = len(data)
    num_batches = data_len // batch_size
    if num_features == 0:
        batched_data = np.zeros([batch_size, num_batches], dtype=np.int32)
    else:
        # could probably just do some reshaping, but head hurts
        batched_data = np.zeros([batch_size, num_batches, num_features],
                                dtype=data.dtype)
    for i in xrange(batch_size):
        batched_data[i] = data[num_batches * i: num_batches * (i+1), ...]

    num_epochs = (num_batches - 1) // num_steps

    if num_epochs == 0:
        raise ValueError('batch size or num_steps too big')

    for i in xrange(num_epochs):
        x = batched_data[:, i*num_steps:(i+1)*num_steps, ...]
        y = batched_data[:, i*num_steps+1:(i+1)*num_steps+1, ...]
        yield (x, y)
