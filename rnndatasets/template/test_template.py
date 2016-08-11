"""tests for the template matching problems"""
import unittest

import rnndatasets.template.template as template
import tensorflow as tf


class TestTemplate(unittest.TestCase):

    def test_block_finding(self):

        sequences = template.online_block_tensors(4, 25, 5)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        seqs, lengths, labels = sess.run(sequences)
        self.assertEqual(seqs.shape, (25, 4, 1))
        self.assertEqual(lengths.shape, (4,))
        self.assertEqual(labels.shape, (4,))
