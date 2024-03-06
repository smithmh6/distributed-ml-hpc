#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for MOE.get_msq() method.
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
from moepy.moe_agent import MOE


# test the get_moe_data method
class TestGetMsq(unittest.TestCase):

    # set up the test case
    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        sys.stdout.write('\nSetting up test class... ')

        # path to parent directory
        cls.data_dir = Path(__file__).resolve().parent.parent

        # data directory + file name
        cls.data_file = ('data'
                        + os.sep
                        + 'test_output'
                        + os.sep
                        + 'test_07.json')

        # full path to data file (json)
        cls.data_path = os.path.join(cls.data_dir, cls.data_file)

        # name of expected results file
        cls.exp_data_file = ('data'
                        + os.sep
                        + 'test_dataset_3'
                        + os.sep
                        + 'test_expected.json')

        # path to json with expected test results
        cls.expected_data_path = os.path.join(cls.data_dir, cls.exp_data_file)

        # load exp_data_file.json
        with open(cls.expected_data_path) as exp_data:
            # load in expected data for tests
            cls.expected_data = json.load(exp_data)

        sys.stdout.write('SUCCESS ')

    def test_get_msq(self):
        """
        Test get_msq() method.
        """

        sys.stdout.write('\nTesting get_msq()... ')


        # setup test layers and materials (see test_dataset_3/test_input.json)
        test_layers = [
            2796.3, 470.8, 480.3, 1099.7, 1659, 369.4, 1601.6, 1585.9, 2271.7
        ]
        test_materials = ["H", "L", "H", "L", "H", "L", "H", "L", "H"]
        test_angles = [0]

        # instance of MOE class
        test_moe = MOE(self.data_path)

        test_res, test_xcal, test_xval = test_moe.get_sys_response()

        test_xc_norm, test_xv_norm = test_moe.get_spec_norm(test_xcal, test_xval)

        # call get_msq()
        test_msq = test_moe.get_msq(
            test_layers,
            test_angles,
            test_materials,
            test_res,
            test_xc_norm,
            test_xv_norm
        )

        # write results to std out
        sys.stdout.write('\n\n MSQ: ' + str(test_msq) + '\n')

        # make call to get_moe_performance()
        #test_perf = test_moe.optimize_design(
        #    test_angles, test_layers, test_materials,
        #    test_res, test_xc_norm, test_xv_norm
        #)


        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\n [x] NO ASSERTIONS')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
