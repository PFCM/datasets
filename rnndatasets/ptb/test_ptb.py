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

    def test_data_in_vocab(self):
        """Make sure we can get everything back out"""
        train, valid, test, vocab = ptb.get_ptb_data()
        inverse_vocab = {vocab[key]: key for key in vocab}
        for batch in ptb.batch_iterator(train, 100, 64):
            words = [inverse_vocab[int(id_)] for id_ in batch[0].flatten()]
        for batch in ptb.batch_iterator(valid, 100, 64):
            words = [inverse_vocab[id_] for id_ in batch[0].flatten()]
        for batch in ptb.batch_iterator(test, 100, 12):
            words = [inverse_vocab[id_] for id_ in batch[0].flatten()]
            print(' '.join([inverse_vocab[id_] for id_ in batch[0][0, :]]))
