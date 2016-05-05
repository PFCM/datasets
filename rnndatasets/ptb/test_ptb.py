"""ludicrously simplistic tests"""
import unittest

import rnndatasets.ptb as ptb


class TestPTBIter(unittest.TestCase):

    def test_get_data(self):
        """just try and get the data"""
        data = ptb.get_ptb_data()
        self.assertEqual(len(data), 4)

    def test_iterate(self):
        """get an iterator"""
        train, valid, test, vocab = ptb.get_ptb_data()
        for batch in ptb.batch_iterator(train, 100, 64):
            self.assertEqual(batch[0].shape, (100, 64))
