"""Tests for chorales"""
import unittest

import rnndatasets.jsbchorales.jsb as jsb


class TestJSBData(unittest.TestCase):

    def test_get_raw(self):
        """Make sure we can get the raw data dict"""
        data = jsb._load_JSB()
        for key in ['test', 'valid', 'train']:
            self.assertIn(key, data)

    def test_big_sequence(self):
        """Make sure we can get the big sequence and it is the
        shape we expect"""
        data = jsb._load_JSB()
        for key in data:
            np_data = jsb._one_big_sequence(data[key])
            # the expected shape is
            # sum(seq lens) + num_seqs by 54
            expected_shape = sum(len(seq) for seq in data[key])
            expected_shape += len(data[key])
            expected_shape = (expected_shape, 55)
            self.assertEqual(np_data.shape, expected_shape)

    def test_batch_iterator(self):
        """iterate in batches and makesure they are the right shape"""
        data = jsb.get_data()

        for split in data:
            for batch in jsb.batch_iterator(split, 10, 11):
                self.assertEqual(batch[0].shape, (10, 11, 55))
                self.assertEqual(batch[1].shape, (10, 11, 55))
