"""Load up JSB chorales data, next step prediction on a bunch of chorales
by JS Bach."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

import rnndatasets.helpers as helpers
from rnndatasets.helpers import batch_iterator


DATA_URL = 'http://www-etud.iro.umontreal.ca/' \
           '~boulanni/JSB%20Chorales.pickle'
EOS = 97  # the end of sequence placeholder is one higher than the highest note
NUM_FEATURES = 55  # the size of the resulting feature vectors


def _filename():
    """Get a path for the data in the directory of this file."""
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'jsb_data.pickle')


def _load_JSB():
    """checks the data is present, if not downloads it.

    Returns:
        data: dict containing keys
            - train
            - valid
            - test
            data within is a list of lists of ints representing the notes
            that are on every quarter note.
    """
    path = helpers.maybe_download(_filename(),
                                  DATA_URL)
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def _one_big_sequence(data):
    """Turns a sequence x time x notes list of lists of lists into
    a big numpy array of sequence x notes by sticking them all together.
    Just uses a 1.0 for all of the notes that are present and a special
    symbol for end of sequence.

    The chorales have all been transposed already into C major or minor,
    -- the highest note is midi note 96 (C6) and the lowest is 43 (G1).
    Therefore the input needs to have dimension 55, to allow one for each
    note and an EOS symbol. There is almost always 4 notes per time-step,
    occasionally 3. EOS will always be alone.
    """
    big_sequence = [item for sequence in data for item in (sequence + [[EOS]])]
    np_data = np.zeros((len(big_sequence), NUM_FEATURES), dtype=np.float32)

    # now fill in the ones
    for i, notes in enumerate(big_sequence):
        np_data[i, tuple(note-43 for note in notes)] = 1.0

    return np_data


def get_data():
    """Get the data, as np arrays representing the whole lot as big
    sequences.


    Returns:
        (train, valid, test): the data.
    """
    data = _load_JSB()
    return [_one_big_sequence(data[key]) for key in ['train', 'valid', 'test']]
