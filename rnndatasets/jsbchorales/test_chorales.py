"""Tests for chorales"""
import unittest

import rnndatasets.jsbchorales.jsb as jsb


class TestJSBData(unittest.TestCase):

    def test_get_raw(self):
        """Make sure we can get the raw data dict"""
        data = jsb._load_JSB()
        for key in ['test', 'valid', 'train']:
            self.assertIn(key, data)
