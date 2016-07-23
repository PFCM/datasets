"""Load up JSB chorales data, next step prediction on a bunch of chorales
by JS Bach."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

import rnndatasets.helpers as helpers


DATA_URL = 'http://www-etud.iro.umontreal.ca/' \
           '~boulanni/JSB%20Chorales.pickle'


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
