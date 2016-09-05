"""Next step prediction on some piano stuff"""
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
           '~boulanni/Piano-midi.de.pickle'
EOS = 109  # one after the highest
NUM_FEATURES = 88  # 31 - 93 + EOS
HIGHEST = 108
LOWEST = 21


def _filename():
    """Get a path for the data in the directory of this file."""
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'pianomidi.pickle')


def _load_data():
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
    data = _load_data()
    return [one_big_sequence(data[key], HIGHEST, LOWEST)
            for key in ['train', 'valid', 'test']]  # ensure order
