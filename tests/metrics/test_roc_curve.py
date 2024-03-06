#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module is the test suite for metrics.roc_curve().

Examples
---------
>>>  python -m unittest -v tests.metrics.test_roc_curve
"""

# standard libraries
import unittest
import pathlib
import sys
import os
import json
#import matplotlib.pyplot as plt

# external packages
import numpy as np
from numpy import testing as nptest

# import modules to test
from moepy.metrics import roc_curve

class TestRocCurve(unittest.TestCase):
    """
    Test class for roc_curve() method.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        cls.dir = pathlib.Path(__file__).resolve().parent.parent

        # define file containing test data
        cls.test_file = os.path.join(cls.dir, r'data\\test_dow_rev5\\test_roc_data.json')

        # read in json file with test expected output data
        with open(cls.test_file) as dat:
            cls.test_data = json.load(dat)

        # define test input parameters
        cls.test_truth = np.asarray(cls.test_data['roc_curve_input']['truth'])
        cls.test_detections = np.asarray(cls.test_data['roc_curve_input']['detections'])
        cls.test_thresh = np.asarray(cls.test_data['roc_curve_input']['thresh'])

       # define expected output from test_expected.json
        cls.roc_curve_exp = cls.test_data['roc_curve_expected']

        # threshold for floating point comparison
        cls.thresh = 14

    def test_roc_curve(self):
        """
        -----------> use test inputs
        """

        # execute roc_curve() function
        test_roc = roc_curve(
            self.test_truth, self.test_detections, self.test_thresh)

        ### plt.figure(figsize=(10, 8))
        ### plt.plot(test_roc['Pfa'], test_roc['Pd'])
        ### plt.title('Receiver Operator Characteristic (ROC) Curve')
        ### plt.xlabel('Probability of False Alarm (1 - Specificity)')
        ### plt.ylabel('Probability of Detection (Sensitivity)')
        ### plt.show()

        # assert test output equals expected values
        self.assertEqual(test_roc['AUROC'], self.roc_curve_exp['AUROC'])
        self.assertEqual(test_roc['t_val'], self.roc_curve_exp['t_val'])
        self.assertEqual(test_roc['Se'], self.roc_curve_exp['Se'])
        self.assertEqual(test_roc['Sp'], self.roc_curve_exp['Sp'])
        nptest.assert_array_almost_equal(
            np.asarray(test_roc['Pd']),
            np.asarray(self.roc_curve_exp['Pd']),
            decimal=self.thresh, verbose=True
        )
        nptest.assert_array_almost_equal(
            np.array(test_roc['Pfa']),
            np.array(self.roc_curve_exp['Pfa']),
            decimal=self.thresh, verbose=True
        )


    def test_roc_curve_arange(self):
        """
        -----------> generated threshold range
        """

        # parameters used to create threshold
        # go from 0 --> 1 in steps of 0.01
        start = 0
        steps = 0.01
        stop = 1 + steps  # <--- ensures that the 1 gets included
        t_range = np.arange(start, stop, steps)

        # execute roc_curve() function using generated threshold
        test_roc = roc_curve(
            self.test_truth, self.test_detections, t_range)

        # assert test output equals expected values
        # assert test output equals expected values
        self.assertEqual(test_roc['AUROC'], self.roc_curve_exp['AUROC'])
        self.assertEqual(test_roc['t_val'], self.roc_curve_exp['t_val'])
        self.assertEqual(test_roc['Se'], self.roc_curve_exp['Se'])
        self.assertEqual(test_roc['Sp'], self.roc_curve_exp['Sp'])
        nptest.assert_array_almost_equal(
            np.asarray(test_roc['Pd']),
            np.asarray(self.roc_curve_exp['Pd']),
            decimal=self.thresh, verbose=True
        )
        nptest.assert_array_almost_equal(
            np.array(test_roc['Pfa']),
            np.array(self.roc_curve_exp['Pfa']),
            decimal=self.thresh, verbose=True
        )

    def test_roc_curve_wrong_shapes(self):
        """
        -----------> truth and detections shape mismatch
        """

        # add an element to truth array
        wrong_truth = np.append(self.test_truth, 0)

        # add an element to detections array
        wrong_detect = np.append(self.test_detections, 0)

        # verify a value error is raised
        with self.assertRaises(ValueError):
            roc_curve(wrong_truth, self.test_detections, self.test_thresh)

        with self.assertRaises(ValueError):
            roc_curve(self.test_truth, wrong_detect, self.test_thresh)



    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
