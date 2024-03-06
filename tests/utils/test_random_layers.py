#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains a test suite for utils.random_layers().

Examples
---------
>>> python -m unittest -v tests.utils.test_random_layers
"""

# import external packages
import unittest
import sys

# import tff_lib for testing
from moepy.utils import random_layers


class TestRandomLayers(unittest.TestCase):
    """
    Test class for the MOE().generate_layers().
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # define test parameters
        cls.test_max_layers = 20
        cls.test_thickness = 0.94

    def test_random_layers(self):
        """
        ---------> default test case
        """

        # test generate_layers()
        _layers = random_layers(self.test_max_layers, self.test_thickness)

        #print('\n', _layers)

        # verify output
        self.assertTrue(len(_layers) <= 20)
        self.assertTrue(type(_layers) == list)
        self.assertTrue(type(_layers[0]) == tuple)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()