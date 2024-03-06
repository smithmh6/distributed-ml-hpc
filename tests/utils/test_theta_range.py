#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for theta_range() function.

Examples
----------------
>>> python -m unittest -v tests.moe_utils.test_theta_range
"""

# import packages
import unittest
import sys
import numpy as np
from numpy import testing as nptest

# import moepy module to test
from moepy.utils import theta_range

class TestThetaRange(unittest.TestCase):
    """
    Test suite for moe_utils.theta_range().
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        # set up the test data
        cls.test_angle = 30
        cls.test_fov = 50
        cls.test_npoints = 10

    def test_theta_range_ints(self):
        """
        -------> zero's as ints and default n_points
        """

        # test theta_range()
        test_range = theta_range(0, 0)

        # should equal an array with a single 0
        nptest.assert_array_equal(test_range, np.array([0]))

    def test_theta_range_floats(self):
        """
        -------> zero's as floats and default n_points
        """

        # test theta_range()
        test_range = theta_range(0.0, 0.0)

        # should equal an array with a single 0
        nptest.assert_array_equal(test_range, np.array([0]))


    def test_theta_range_strings(self):
        """
        -------> strings as args
        """

        # test get_theta_range() with strings
        with self.assertRaises(TypeError):
            theta_range('string', 0)

        with self.assertRaises(TypeError):
            theta_range(0, 'string')

        with self.assertRaises(TypeError):
            theta_range(0, 0, n_points='string')


    def test_theta_range_default_npoints(self):
        """
        -------> default n_points
        """

        # test get_theta_range()
        test_range = theta_range(self.test_angle, self.test_fov)

        # define the expected values
        exp_start = self.test_angle - (self.test_fov / 2)
        exp_stop = self.test_angle + (self.test_fov / 2)
        exp_range = np.linspace(exp_start, exp_stop, 7)

        # assert that the test value equals the expected value
        nptest.assert_equal(test_range, exp_range)

    def test_theta_range_npoints(self):
        """
        -------> increase n_points
        """

        # test get_theta_range()
        test_range = theta_range(self.test_angle, self.test_fov, self.test_npoints)

        # define the expected values
        exp_start = self.test_angle - (self.test_fov / 2)
        exp_stop = self.test_angle + (self.test_fov / 2)
        exp_range = np.linspace(exp_start, exp_stop, self.test_npoints)

        # assert that the test value equals the expected value
        nptest.assert_equal(test_range, exp_range)

    def test_theta_range_zero_fov(self):
        """
        -------> fov set to zero
        """

        # test get_theta_range()
        test_range = theta_range(self.test_angle, 0)

        # assert that the test value equals the expected value
        nptest.assert_equal(test_range, np.array([self.test_angle]))


    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()