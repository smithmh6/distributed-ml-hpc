#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for MOE.optimize_design() method.
"""
# import external packages
import time
from unittest.loader import TestLoader
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json
import plotly.graph_objects as go


# import moepy modules
from moepy.moe_agent import MOE

class TestOptimizeDesign(unittest.TestCase):

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

    def test_optimize_design(self):
        """
        Test optimize_design() method.
        """

        sys.stdout.write('\nTesting optimize_design()... ')

        # setup test layers and materials (see test_dataset_3/test_input.json)
        #test_layers = [
        #    2796.3, 470.8, 480.3, 1099.7, 1659, 369.4, 1601.6, 1585.9, 2271.7
        #]
        #test_materials = ["H", "L", "H", "L", "H", "L", "H", "L", "H"]

        test_angles = [0]

        # instance of MOE class
        test_moe = MOE(self.data_path)

        test_layers, test_materials = test_moe.generate_layers()

        # calculate the system response
        test_res,test_xcal,test_xval,response_ei = test_moe.get_sys_response()

        # normalize spectral data
        test_xc_norm, test_xv_norm = test_moe.get_spec_norm(test_xcal, test_xval)

        # calculate performance pre-optimization
        reference_perf = test_moe.get_performance(
            test_layers,
            test_angles,
            test_materials,
            test_res,
            test_xc_norm,
            test_xv_norm
        )

        # call optimize_design()
        test_opt_layers, test_opt_perf, test_opt_output = test_moe.optimize_design(
            test_layers,
            test_angles,
            test_materials,
            test_res,
            test_xc_norm,
            test_xv_norm
        )

        # write the results of optimize_design() to output stream
        sys.stdout.write(
            '\nInput Layer Stack: '
            + str(test_layers) + '\n'
        )
        sys.stdout.write(
            '\nOptimized Layer Stack: \n'
            + str(test_opt_layers) + '\n'
        )
        sys.stdout.write(
            '\nOptimizer Output: \n'
            + str(test_opt_output) + '\n'
        )
        sys.stdout.write(
            '\nOptimized Performance: \n'
            + '\nGAIN: ' + str(test_opt_perf['G'][0][0])
            + '\nOffset: ' + str(test_opt_perf['fil_off'][0][0])
            + '\nSEC: ' + str(test_opt_perf['SEC'][0][0])
            + '\nSEP: ' + str(test_opt_perf['SEP'][0][0])
            + '\nSNR: ' + str(test_opt_perf['SNR'][0][0]) + '\n'
        )

        # plot the before/after transmission
        fig = go.Figure()

        # add reference spectrum
        fig.add_trace(
            go.Scatter(
                x=np.squeeze(np.real(test_moe.setup['wv'])),
                y=np.squeeze(np.real(reference_perf['spec']['T'])),
                mode='lines',
                name=('T - Reference ' + str(reference_perf['SEC'][0][0]))
            )
        )

        # add the optimized spectrum
        fig.add_trace(
            go.Scatter(
                x=np.squeeze(np.real(test_moe.setup['wv'])),
                y=np.squeeze(np.real(test_opt_perf['spec']['T'])),
                mode='lines',
                name=('T - Optimized ' + str(test_opt_perf['SEC'][0][0]))
            )
        )
        fig.update_layout(
            title="Optimization Results",
            xaxis_title="Wavelength (nm)",
            yaxis_title="% T"
        )
        #fig.show()

        # assert that the optimizer actually iterates
        self.assertGreater(test_opt_output.nit, 0)
        # assert optimization completed successfully
        self.assertEqual(test_opt_output.status, 0)
        self.assertEqual(test_opt_output.success, True)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\n PASSED')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
