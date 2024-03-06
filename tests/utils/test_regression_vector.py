#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for utils.regression_vector()

Example Usage
---------
>>> python -m unittest -v tests.utils.test.regression_vector
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json
import matplotlib.pyplot as plt

# import function to test
from moepy.utils import regression_vector, generate_sine

class TestRegressionVector(unittest.TestCase):
    """
    Test class for utils.regression_vector()
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # get path of test directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data\test_dow_rev5')
        cls.output_dir = os.path.join(cls.dir, 'out')

        # path to test data file
        cls.data_file = os.path.join(cls.dir, 'out/Dow_MOE_Rev5.json')

        # read in json file with test input data
        with open(cls.data_file) as dat:
            cls.test_data = json.load(dat)

        # test transmission and reflection arrays
        cls.test_WV, cls.test_T = generate_sine(start=1650, end=2050, freq=.05, amp=1, shift=1)
        _, cls.test_R = generate_sine(start=1650, end=2050, freq=.025, amp=1, shift=1, theta=.25)

        # compute the expected regression vectors
        cls.rv_dict = {}
        for opt in range(0, 12):
            # calculation based on 'oc' parameter
            if opt in (0, 6):
                cls.rv_dict[opt] = cls.test_T - cls.test_R
            elif opt in (1, 7):
                cls.rv_dict[opt] = (cls.test_T - cls.test_R) / (cls.test_T + cls.test_R)
            elif opt in (2, 8):
                cls.rv_dict[opt] = cls.test_T - .5 * cls.test_R
            elif opt in (3, 9):
                cls.rv_dict[opt] = 2 * cls.test_T - np.ones(np.shape(cls.test_WV)).astype(complex)
            elif opt in (4, 5, 10, 11):
                cls.rv_dict[opt] = cls.test_T

        # visualize the test data
        plt.figure(figsize=(12, 6))
        plt.plot(cls.test_WV, cls.test_T, label="T")
        plt.plot(cls.test_WV, cls.test_R, label="R")
        plt.title('Regression Vector Inputs')
        plt.xlabel('Value')
        plt.ylabel('Wavelength (nm)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(cls.output_dir, 'reg_vec_input.png'))

    def test_reg_vec(self):
        """
        --------->  default test case for reg_vec()
        """

        # test reg_vec() method
        for opt in range(0, 12):
            _rv = regression_vector(self.test_WV, self.test_T, self.test_R, opt)

            # verify the output
            self.assertEqual(np.ndarray, type(_rv))
            nptest.assert_array_equal(self.rv_dict[opt], _rv)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up test resources.
        """

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()