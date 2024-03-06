#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for processing.interpolate_1d()

Examples
---------
>>> python -m unittest -v tests.processing.test_interpolate_1d
"""

# import dependencies
import unittest
from pathlib import Path
import sys
import os
import numpy as np

# import function to test
from moepy.dataset import interpolate_1d


class TestInterpolate1D(unittest.TestCase):
    """
    Test class for processing.interpolate_1d()
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # get path of test data directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data\test_dow_rev5')

        # define the test values
        cls.x_min = 400
        cls.x_max = 1000
        cls.x_test = np.arange(400, 1000, 1)
        cls.y_test = np.exp(cls.x_test/100)
        cls.kind = 'linear'
        cls.spacing = 1

    def test_interpolate_1d(self):
        """
        ---->  test case for interpolate_1d()
        """

        test_x, text_y = interpolate_1d(
            self.x_test, self.y_test, self.x_min, self.x_max, kind=self.kind, spacing=self.spacing)

        self.assertEqual(np.shape(test_x), np.shape(text_y))

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up test resources.
        """
        sys.stdout.write('\nDone!')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()