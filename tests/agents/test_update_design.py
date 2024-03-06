#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for MOE.update_design() method.
"""
# import external packages
import time
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json
import plotly.graph_objects as go
import pyodbc

# import moepy modules
from moepy.moe_agent import MOE


class TestUpdateDesign(unittest.TestCase):

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

    def test_update_design(self):
        """
        Test update_design() method.
        """

        sys.stdout.write('\nTesting update_design()... ')

        # instance of MOE class
        test_moe = MOE(self.data_path)

        test_record = test_moe.update_design(
            count=5, status=2
        )

        self.assertNotEqual(test_record, None)
        self.assertEqual(len(test_record), 2)
        self.assertEqual(test_record[0], 5)
        self.assertEqual(test_record[1], 2)

        # reset previous/status fields to 0
        test_moe.update_design(count=0, status=0)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\n PASSED')

    def test_update_design_noStatus(self):
        """
        Test update_design() method with no 'status' arg.
        """

        sys.stdout.write('\nTesting update_design_noStatus()... ')

        # instance of MOE class
        test_moe = MOE(self.data_path)

        test_record = test_moe.update_design(
            count=5
        )

        self.assertNotEqual(test_record, None)
        self.assertEqual(len(test_record), 2)
        self.assertEqual(test_record[0], 5)
        self.assertEqual(test_record[1], 0)

        # reset previous/status fields to 0
        test_moe.update_design(count=0, status=0)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\n PASSED')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure...')

        sys.stdout.write('\nSUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
