#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for utils.system_response().

Example Usage
---------
>>> python -m unittest -v tests.utils.test.system_response
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

# import moepy modules
from moepy.utils import system_response, generate_sine

class TestSystemResponse(unittest.TestCase):
    """
    Test class for utils.system_response().
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        # path to data directory and output directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data/test_dow_rev5')
        cls.output_dir = os.path.join(cls.dir, 'out')

        # read in data for x_cal and x_val
        with open(os.path.join(cls.dir, 'test_sys_resp_data.json'), 'r') as df:
            cls.test_data = json.load(df)

        # get the xcal/xval data as arrays from file
        cls.test_xcal = np.asarray(cls.test_data['calibration']['xcal'])
        cls.text_xval = np.asarray(cls.test_data['validation']['xval'])

        # simulate component spectra and wavelength array
        start = 1650
        end = 2051
        cls.test_wv, one = generate_sine(start=start, end=end, freq=.02, theta=0.3, shift=1)
        _, two = generate_sine(start=start, end=end, freq=.03, theta=0.1, shift=1)
        _, three = generate_sine(start=start, end=end, freq=.015, theta=0.2, shift=1)
        cls.test_optics = {
            'one': one,
            'two': two,
            'three': three
        }

        # compute the expected output
        cls.exp_response = one * two * three
        cls.exp_xcal = cls.test_xcal * np.sum(cls.exp_response)
        cls.exp_xval = cls.text_xval * np.sum(cls.exp_response)

        # plot and save the test data
        ## plt.figure(figsize=(12, 6))
        ## plt.plot(cls.test_wv, one, label="Component 01")
        ## plt.plot(cls.test_wv, two, label="Component 02")
        ## plt.plot(cls.test_wv, three, label="Component 03")
        ## plt.plot(cls.test_wv, cls.exp_response, label='System Response')
        ## plt.title('System Response Inputs')
        ## plt.xlabel('Value')
        ## plt.ylabel('Wavelength (nm)')
        ## plt.legend(loc="lower right")
        ## plt.tight_layout()
        ## plt.savefig(os.path.join(cls.output_dir, 'sys_resp_input.png'))

    def test_get_sys_response(self):
        """
        -----------> default test case for system_response()
        """

        # test the system_response() function
        _resp, _xcal, _xval = system_response(
            self.test_wv, self.test_optics, self.test_xcal, self.text_xval)

        # verify the output
        nptest.assert_array_equal(self.exp_response, _resp)
        nptest.assert_array_equal(self.exp_xcal, _xcal)
        nptest.assert_array_equal(self.exp_xval, _xval)

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()