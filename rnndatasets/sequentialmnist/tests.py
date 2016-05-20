"""Sequential mnist tests"""
import unittest

import rnndatasets.sequentialmnist.sequentialmnist as sm


class SMTests(unittest.TestCase):

    def test_get_iters_time_major(self):
        """make sure we can load the data and all the iterators
        give us the right shape"""
        a, b, c = sm.get_iters(64, time_major=True)
        num_batches = 50000 // 64  # how many we should have
        found_batches = 0
        for data, labels in a:
            found_batches += 1
            self.assertEqual(data.shape, (784, 64, 1))
            self.assertEqual(labels.shape, (64,))
        self.assertEqual(found_batches, num_batches)
        # valid and test are 10,000
        num_batches = 10000 // 64
        found_batches = 0
        for data, labels in b:
            found_batches += 1
            self.assertEqual(data.shape, (784, 64, 1))
            self.assertEqual(labels.shape, (64,))
        self.assertEqual(found_batches, num_batches)
        found_batches = 0
        for data, labels in c:
            found_batches += 1
            self.assertEqual(data.shape, (784, 64, 1))
            self.assertEqual(labels.shape, (64,))
        self.assertEqual(found_batches, num_batches)

        def test_get_iters_batch_major(self):
            """make sure we can load the data and all the iterators
            give us the right shape"""
            a, b, c = sm.get_iters(64, time_major=False)
            num_batches = 50000 // 64  # how many we should have
            found_batches = 0
            for data, labels in a:
                found_batches += 1
                self.assertEqual(data.shape, (64, 784, 1))
                self.assertEqual(labels.shape, (64,))
            self.assertEqual(found_batches, num_batches)
            # valid and test are 10,000
            num_batches = 10000 // 64
            found_batches = 0
            for data, labels in b:
                found_batches += 1
                self.assertEqual(data.shape, (64, 784, 1))
                self.assertEqual(labels.shape, (64,))
            self.assertEqual(found_batches, num_batches)
            found_batches = 0
            for data, labels in c:
                found_batches += 1
                self.assertEqual(data.shape, (64, 784, 1))
                self.assertEqual(labels.shape, (64,))
            self.assertEqual(found_batches, num_batches)
