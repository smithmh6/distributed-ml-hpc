#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for moe_utils.spec_mean() function.
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json

# import moepy modules
from moepy.utils import spec_mean


class TestSpecMean(unittest.TestCase):
    """
    Test suite moe_utils.spec_mean().
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        # path to parent directory
        cls.dir = Path(__file__).resolve().parent.parent

        # input data filie
        cls.data_file = os.path.join(cls.dir, 'data/test_dataset_4/design_id_16.json')

        # read in the test data
        with open(cls.data_file) as dat:
            cls.test_data = json.load(dat)

        # extract x-calibration and x-validation data
        cls.test_xcal = np.asarray(cls.test_data['calibration']['xcal'])
        cls.test_xval = np.asarray(cls.test_data['validation']['xval'])

        # calculate the default expected results of spec_mean
        cls.exp_avg = np.mean(np.concatenate((cls.test_xcal, cls.test_xval), axis=0))

        #######################################################################
        # calculate scaled test data to test thresholding and scaling #

        # multiply the test data by 5 to increase
        # the mean value so the threshold will be
        # activated
        cls.upscale_xcal = cls.test_xcal * 5
        cls.upscale_xval = cls.test_xval * 5

        # calculate the expected values
        cls.exp_scaled_xcal = cls.upscale_xcal / 100.0
        cls.exp_scaled_xval = cls.upscale_xval / 100.0
        cls.exp_scaled_avg = np.mean(
            np.concatenate((cls.exp_scaled_xcal, cls.exp_scaled_xval), axis=0)
        )

    def test_spec_mean(self):
        """
        -------> default named args
        """

        # execute the spec_mean() function
        _avg, _xcal, _xval = spec_mean(self.test_xcal, self.test_xval)

        # assert mean values are equal
        self.assertEqual(_avg, self.exp_avg)

        # for this case, xcal and xval should be untouched
        # by the function
        nptest.assert_equal(_xcal, self.test_xcal)
        nptest.assert_equal(_xval, self.test_xval)

    def test_spec_mean_thresh_int(self):
        """
        -------> set a new integer threshold value
        """

        # execute the function with a threshold setpoint
        _avg, _xcal, _xval = spec_mean(self.upscale_xcal, self.upscale_xval, thresh=2)

        # assert the results match expected values
        self.assertEqual(_avg, self.exp_scaled_avg)
        nptest.assert_equal(_xcal, self.exp_scaled_xcal)
        nptest.assert_equal(_xval, self.exp_scaled_xval)

    def test_spec_mean_thresh_float(self):
        """
        -------> set a new floating point threshold value
        """

        # execute the function with a threshold setpoint
        _avg, _xcal, _xval = spec_mean(self.upscale_xcal, self.upscale_xval, thresh=2.5)

        # assert the results match expected values
        self.assertEqual(_avg, self.exp_scaled_avg)
        nptest.assert_equal(_xcal, self.exp_scaled_xcal)
        nptest.assert_equal(_xval, self.exp_scaled_xval)

    def test_spec_mean_wrong_types(self):
        """
        -------> input wrong types as arguments
        """

        # verify that the correct exceptions are raised
        with self.assertRaises(TypeError):
            spec_mean('wrong type', self.test_xval)
        with self.assertRaises(TypeError):
            spec_mean(self.test_xcal, 'wrong type')
        with self.assertRaises(TypeError):
            spec_mean(self.test_xcal, self.test_xval, thresh='wrong type')
        with self.assertRaises(TypeError):
            spec_mean(self.test_xcal, self.test_xval, scale='wrong type')

    def test_spec_mean_wrong_values(self):
        """
        -------> input wrong values for thresh
        """

        # verify that a value error is raised at thresh == 0
        wrong_thresh = 0
        with self.assertRaises(ValueError):
            spec_mean(self.upscale_xcal, self.upscale_xval, thresh=wrong_thresh)

        # verify that a value error is raised at thresh < 0
        wrong_thresh = -2
        with self.assertRaises(ValueError):
            spec_mean(self.upscale_xcal, self.upscale_xval, thresh=wrong_thresh)


    @classmethod
    def tearDownClass(cls):
        sys.stdout.write('\nRunning teardown procedure... SUCCESS')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()