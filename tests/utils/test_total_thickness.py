#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suite for moe.total_thickness().

Examples
----------------
>>> python -m unittest -v tests.moe.test_total_thickness
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
import pathlib
import sys
import os
import json

# import moepy modules
from moepy.utils import total_thickness


class TestTotalThickness(unittest.TestCase):
    """
    Test suite for moe.total_thickness().
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup the test class.
        """


        cls.dir = pathlib.Path(__file__).resolve().parent.parent

        # define file containing test data
        cls.test_file = os.path.join(cls.dir, r'data\\test_dow_rev5\\out\\Dow_MOE_Rev5.json')

        # read in json file with test expected output data
        with open(cls.test_file) as dat:
            cls.test_data = json.load(dat)

        # define test input parameters
        cls.test_high = np.asarray(cls.test_data['setup']['high_material'])
        cls.test_low = np.asarray(cls.test_data['setup']['low_material'])
        cls.test_spec_res = int(cls.test_data['design']['spec_res'])

        cls.test_exp = 5523.0245552568595

    def test_get_total_thickness(self):
        """
        ------------> default test case
        """

        # call get_total_thick()
        _thick = total_thickness(self.test_high, self.test_low, self.test_spec_res)

        # verify output
        self.assertEqual(self.test_exp, _thick)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()