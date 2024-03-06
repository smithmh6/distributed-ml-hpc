#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for MOE.write_to_db() method.
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


class TestWriteToDb(unittest.TestCase):

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

    def test_write_to_db(self):
        """
        Test write_to_db() method.
        """

        sys.stdout.write('\nTesting write_to_db()... ')

        start = time.time()

        # setup test layers and materials (see test_dataset_3/test_input.json)
        test_layers = [
            2796.3, 470.8, 480.3, 1099.7, 1659, 369.4, 1601.6, 1585.9, 2271.7
        ]

        test_materials = ["H", "L", "H", "L", "H", "L", "H", "L", "H"]

        # instance of MOE class
        test_moe = MOE(self.data_path)

        # test angles
        test_angles = test_moe.get_theta_range()

        # pre-process the data so the
        # performance can be calculated
        spec_mean = test_moe.get_spec_mean()

        # calculate radiometric curve
        response, xcal, xval, ei = test_moe.get_sys_response()

        # normalize the data
        xc_norm, xv_norm = test_moe.get_spec_norm(xcal, xval)

        # calculate the performance
        test_perf = test_moe.get_performance(
            test_layers, test_angles, test_materials, response, xc_norm, xv_norm
        )

        test_time = time.time() - start

        # test write_to_db method
        test_input = {
            'design_time': test_time,
            'sec': test_perf['SEC'][0][0],
            'sep': test_perf['SEP'][0][0],
            'snr': test_perf['SNR'][0][0],
            'ab_per_unit': test_perf['delta_AdivB'][0][0],
            'ab_over_ei': test_perf['delta_AdivB_EI'][0][0],
            'quote_hw_req': 0,
            'gain': test_perf['G'][0][0],
            'offset': test_perf['fil_off'][0][0],
            'pcnt_t': 0,
            'total_thick': np.sum(test_layers),
            'total_layers': len(test_layers)
        }

        for i, v in enumerate(test_layers):
            test_input['layer_' + str(i)] = v

        test_record_id = test_moe.write_to_db(test_input)

        self.assertNotEqual(test_record_id, None)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\n PASSED')


        #----- REMOVE RECORD FROM DB ---- #

        # setup the database connection
        server = 'tcp:ar-database-thorlabs.database.windows.net,1433'
        database = 'TSW_Web'
        username = 'hsmith'
        password = '{Welcome1}'
        driver= '{ODBC Driver 17 for SQL Server}'

        # build connection
        conn = pyodbc.connect(
            'DRIVER=' + driver
            + ';SERVER=' + server
            + ';DATABASE=' + database
            + ';UID=' + username
            + ';PWD=' + password
        )

        if test_record_id is not None:
            # remove the test record from database after test
            sys.stdout.write('\nRemoving test record from database...')
            # open db connection
            with conn as c:
                # open cursor object
                with c.cursor() as cur:
                    # execute sql command and get cursor object
                    cur.execute(
                        "DELETE FROM [dbo].[moe_app_output] WHERE id='%s';" % test_record_id
                    )
        sys.stdout.write('\nSUCCESS')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure...')

        sys.stdout.write('\nSUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
