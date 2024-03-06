#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test class for testing
the 'GetData' class.
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

# import GetData class
from moepy.data_utils import GetData

# test the get_moe_data method
class TestGetData(unittest.TestCase):

    # set up the test case
    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        sys.stdout.write('\nSetting up test class... ')

        # path to parent directory
        cls.parent_dir = Path(__file__).resolve().parent.parent

        cls.output_path = os.path.join(cls.parent_dir, 'data/test_output')

        cls.test_id = 13

        sys.stdout.write('SUCCESS ')

    def test_get_data(self):

        sys.stdout.write('\nTesting get_data()... ')

        # instantiate GetData() class
        d = GetData(self.test_id)

        # test the get_design() method
        test_design = d.get_design()

        # write the results to console for now
        #sys.stdout.write(
        #    '\n\nResult:\n------------\n\n'
        #    + str(test_design) + '\n'
        #)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')


    def test_get_setup(self):

        sys.stdout.write('\nTesting get_setup()... ')

        # instantiate GetData() class
        d = GetData(self.test_id)

        # test the get_design() method
        test_setup = d.get_setup()

        # write the results to console for now
        #sys.stdout.write(
        #    '\n\nResult:\n------------\n\n'
        #    + str(test_setup) + '\n'
        #)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')

    def test_get_constraints(self):

        sys.stdout.write('\nTesting get_constraints()... ')

        # instantiate GetData() class
        d = GetData(self.test_id)

        # test the get_design() method
        test_cons = d.get_constraints()

        # write the results to console for now
        #sys.stdout.write(
        #    '\n\nResult:\n------------\n\n'
        #    + str(test_cons) + '\n'
        #)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')


    def test_seed_points(self):
        """
        Tests querying the seedpoint table in
        the spec lib database.
        """

        sys.stdout.write('\nTesting get_seed_points()... ')

        # instantiate GetData() class
        d = GetData(self.test_id)

        # test the get_design() method
        test_sp = d.get_seed_points()

        # write the results to console for now
        #sys.stdout.write(
        #    '\n\nResult:\n------------\n\n'
        #    + str(test_sp) + '\n'
        #)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')


    def test_get_cal_set(self):

        sys.stdout.write('\nTesting get_cal_set()... ')

        d = GetData(self.test_id)

        # test the get_design() method
        test_cal = d.get_cal_set(10)

        # write the results to console for now
        #sys.stdout.write(
        #    '\n\nResult:\n------------\n\n'
        #    + str(test_cal) + '\n'
        #)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')


    def test_get_val_set(self):

        sys.stdout.write('\nTesting get_val_set()... ')

        d = GetData(self.test_id)

        # test the get_design() method
        test_val = d.get_val_set(4)

        # write the results to console for now
        #sys.stdout.write(
        #    '\n\nResult:\n------------\n\n'
        #    + str(test_val) + '\n'
        #)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')


    def test_save_file(self):

        sys.stdout.write('\nTesting save_file()... ')

        # instantiate GetData() class
        test_filename = 'test_07.json'
        test_path = os.path.join(self.output_path, test_filename)
        d = GetData(self.test_id, test_path)

        # test the get_design() method
        test_save = d.save_file()

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\nPENDING ASSERTIONS...\n')


    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()