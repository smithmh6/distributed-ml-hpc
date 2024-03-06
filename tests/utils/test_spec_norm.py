#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for moe_utils.spec_norm().

Examples
-----------
>>> python -m unittest -v tests.moe_utils.test_spec_norm
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json

# import function to test
from moepy.utils import spec_norm


class TestSpecNorm(unittest.TestCase):
    """
    Test suite for moe_utils.spec_norm().
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        # path to parent directory
        cls.dir = Path(__file__).resolve().parent.parent

        # full path to data file (json)
        cls.data_file = os.path.join(cls.dir, 'data/test_dataset_4/design_id_16.json')

        # read in the test data
        with open(cls.data_file) as dat:
            cls.test_data = json.load(dat)

        # extract x-calibration and x-validation data
        cls.test_xcal = np.asarray(cls.test_data['calibration']['xcal'])
        cls.test_xval = np.asarray(cls.test_data['validation']['xval'])
        cls.test_axis = 1

        cls.exp_xcal = np.linalg.norm(cls.test_xcal, axis=cls.test_axis)
        cls.exp_xval = np.linalg.norm(cls.test_xval, axis=cls.test_axis)

    def test_spec_norm_default(self):
        """
        ----> default parameters
        """

        # use test_xcal/test_xval as input to get_spec_norm()
        xcal_norm, xval_norm = spec_norm(self.test_xcal, self.test_xval)

        # arrays should pass through unchanged
        nptest.assert_array_equal(xcal_norm, self.test_xcal)
        nptest.assert_array_equal(xval_norm, self.test_xval)

    def test_spec_norm_wrong_types(self):
        """
        ----> wrong data types
        """

        # assert TypeError is raised for these cases
        with self.assertRaises(TypeError):
            spec_norm(self.test_xcal, 'wrong type')
        with self.assertRaises(TypeError):
            spec_norm('wrong type', self.test_xval)
        with self.assertRaises(TypeError):
            spec_norm(self.test_xcal, self.test_xval, axis='wrong type')

    def test_spec_norm_with_axis(self):
        """
        ----> supply non-zero value for 'axis'
        """

        # execute function with test axis
        xcal_norm, xval_norm = spec_norm(self.test_xcal, self.test_xval, axis=self.test_axis)

        # assert results equal expected values
        nptest.assert_array_equal(xcal_norm, self.exp_xcal)
        nptest.assert_array_equal(xval_norm, self.exp_xval)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any remaining resources.
        """

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()