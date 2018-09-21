# -*- coding: utf-8 -*-

from .context import tagger

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_true(self):
        assert True


if __name__ == '__main__':
    unittest.main()
