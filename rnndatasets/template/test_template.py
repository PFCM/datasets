"""tests for the template matching problems"""
import unittest

import template

class TestTemplate(unittest.TestCase):

    def test_block_finding(self):

        sequences = template.online_block_tensors(1, 10, 4)

        sess = tf.Session()
        sess.run(tf.initialise_all_variables())

        print(sess.run(sequences))
        self.assertTrue(False)
