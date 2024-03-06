#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for processing.test_excel_to_json()

Examples
---------
>>> python -m unittest tests.processing.test_excel_to_json
"""

# import dependencies
import unittest
from pathlib import Path
import sys
import os
import configparser as cp

# import function to test
from moepy.dataset import excel_to_json


class TestExcelToJson(unittest.TestCase):
    """
    Test class for processing.test_excel_to_json()
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # get path of test data directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data\test_dow_rev5')

        # define the test excel file
        cls.test_file = os.path.join(cls.dir, 'Dow_MOE_Rev5.xlsx')

        # define and check the output directory
        cls.output_dir = os.path.join(cls.dir, 'out')
        if not os.path.exists(cls.output_dir):
            os.mkdir(cls.output_dir)

    def test_excel_to_json(self):
        """
        --------->  test case for excel_to_json()
        """

        _output = excel_to_json(self.test_file, self.output_dir)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up test resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()