"""Quick tests for the functions around the war and peace data.
Not really intended to be thorough, because this stuff is pretty
rough and ready.

Note that these will test the data downloads, so requires internet.
Also will leave the dataset lying around, so be aware.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import rnndatasets.warandpeace as warandpeace
from rnndatasets.warandpeace.warandpeace import _get_big_string


# TODO: (pfcm) this, but better
class WarAndPeaceTests(unittest.TestCase):
    """quick tests, make sure everything seems to be in order"""

    def test_get_big_string(self):
        """make sure we can get the data in its rawest form"""
        self.assertIsNotNone(_get_big_string())

    def test_get_vocab(self):
        """make sure we get something without an error"""
        self.assertIsNotNone(warandpeace.get_vocab('char'))

    def test_iterator(self):
        """see if we can iterate war and peace in batches"""
        slen = 12
        blen = 11
        for batch in warandpeace.get_char_iter(slen, blen):
            self.assertEqual(len(batch), blen)
            for item in batch:
                self.assertEqual(item.shape, (slen,))
