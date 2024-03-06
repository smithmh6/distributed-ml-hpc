#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for processing.read_xlsx()

Examples
---------
>>> python -m unittest -v tests.processing.test_read_xlsx
"""

# import dependencies
import unittest
from pathlib import Path
import sys
import os

# import function to test
from moepy.dataset import _read_xlsx

class TestReadXlsx(unittest.TestCase):
    """
    Test class for processing._read_xlsx()
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # get path of test data directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent.parent, r'data\uster_plastic_moe6')

        # define the test excel file
        cls.test_file = os.path.join(cls.dir, 'uster_plastic_id_moe6.xlsx')

        # define expected keys in output dict
        cls.expected_keys = [
            'wavelength', 'lamp', 'detector', 'env_int', 'component_01', 'component_02', 'component_03',
            'component_04', 'component_05', 'component_06', 'component_07', 'component_08', 'component_09',
            'component_10', 'substrate', 'high_material', 'low_material', 'xcal', 'ycal', 'xval', 'yval']

    def test_read_xlsx(self):
        """
        --------->  test case for _read_xlsx()
        """

        test_dataset = _read_xlsx(self.test_file)

        # validate output keys
        for k in self.expected_keys:
            self.assertIn(k, list(test_dataset.keys()))

    @classmethod
    def tearDownClass(cls):
        """
        Clean up test resources.
        """
        sys.stdout.write('\nDone!')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()