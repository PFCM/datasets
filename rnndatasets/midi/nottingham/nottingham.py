"""Next step prediction on a bunch of folksongs"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

import rnndatasets.helpers as helpers
from rnndatasets.midi.midi_helpers import one_big_sequence
from rnndatasets.helpers import batch_iterator


DATA_URL = 'http://www-etud.iro.umontreal.ca/' \
           '~boulanni/Nottingham.pickle'
EOS = 94  # one after the highest
NUM_FEATURES = 63  # 31 - 93 + EOS
HIGHEST = 93
LOWEST = 31


def _filename():
    """Get a path for the data in the directory of this file."""
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'Nottingham.pickle')


def _load_nottingham():
    """checks the data is present, if not downloads it.

    Returns:
        data: dict containing keys
            - train
            - valid
            - test
            data within is a list of lists of ints representing the notes
            that are on every eighth note.
    """
    path = helpers.maybe_download(_filename(),
                                  DATA_URL)
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def get_data():
    """Get the data, as np arrays representing the whole lot as big
    sequences.


    Returns:
        (train, valid, test): the data.
    """
    data = _load_nottingham()
    return [one_big_sequence(data[key], HIGHEST, LOWEST)
            for key in ['train', 'valid', 'test']]  # ensure order
