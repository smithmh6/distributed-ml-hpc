#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for __init__() method of MOE class.
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
from moepy.moe_agent import MOE


# test the __init__() method
class TestMoeInit(unittest.TestCase):

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

    def test_moe_init(self):
        """
        Test the __init__() method of MOE class.
        """

        sys.stdout.write('\nTesting moe_init()... ')

        # instance of MOE class
        test_init = MOE(self.data_path)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    def test_moe_init_raises(self):
        """
        Asserts that __init__ raises an IOError when
        there is a problem reading the data file.
        """

        sys.stdout.write('\nTesting moe_init_raises()... ')

        # assert that an exception is raised
        with self.assertRaises(IOError):
            # instantiate with a bad directory
            test_init = MOE(r'C:\User\wrong_dir')

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()