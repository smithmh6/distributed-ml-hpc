#!user/bin/python
# -*- coding: utf-8 -*-
"""
Test suite for moe.get_performance() function.

Examples
-----------
>>> python -m unittest -v tests.moe.test_performance
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

# import function to test
from moepy.moe import performance

# import helper functions
from moepy.dataset import convert_to_numpy
from moepy.utils import system_response
from tff_lib.utils import film_matrix

# test the get_moe_data method
class TestGetPerformance(unittest.TestCase):

    # set up the test case
    @classmethod
    def setUpClass(cls):
        """
        Setup the test class and initialize test variables
        and expected test results.
        """

        # path to data directory and output directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data/test_dow_rev5')
        cls.output_dir = os.path.join(cls.dir, 'out')

        # read in test dataset
        with open(os.path.join(cls.dir, r'out/Dow_MOE_Rev5.json'), 'r') as df:
            cls.test_data = json.load(df)

        # extract necessary test data
        cls.test_data = convert_to_numpy(cls.test_data)



        # read in file with test layer stack
        with open(os.path.join(cls.dir, r'test_performance_data.json'), 'r') as lf:
            cls.test_layer_data = json.load(lf)
        # extract test layers
        cls.test_layers = cls.test_layer_data['layers']

        # create the film matrix of refractive indices
        cls.test_films = film_matrix(
                                cls.test_layers,
                                cls.test_data['setup']['high_material'],
                                cls.test_data['setup']['low_material'])

        # extract optical components from 'setup'
        exclude = ('high_material', 'low_material', 'substrate', 'env_int', 'wv')
        cls.components = {k: v for k, v in cls.test_data['setup'].items() if not k in exclude}

        # calculate the system response w/ & w/o EI
        cls.test_resp, cls.test_resp_ei = system_response(
            cls.test_data['setup']['wv'], cls.components, cls.test_data['setup']['env_int']
        )
        ## plt.figure(figsize=(10, 10))
        ## plt.plot(cls.test_data['setup']['wv'], cls.test_resp)
        ## plt.show()

        # convolve x_cal and x_val with system response
        cls.test_xcal = cls.test_data['calibration']['xcal'] * cls.test_resp_ei
        cls.test_xval = cls.test_data['validation']['xval'] * cls.test_resp_ei

        # create test angle array and incident medium refractive index
        cls.test_angles = np.array([0.0])
        cls.test_med = np.ones(np.shape(cls.test_data['setup']['wv']))

        # define fit order
        cls.test_fit_order = 1

        # define expected output
        cls.exp_T = np.asarray(cls.test_layer_data['T'])

        # define the expected comparison precision
        cls.precision = 14

    def test_performance(self):
        """
        -----------> default test case
        """

        # make call to get_moe_performance()
        _perf = performance(
            self.test_data['setup']['wv'],
            self.test_layers,
            self.test_films,
            self.test_data['setup']['substrate'],
            self.test_med,
            self.test_angles,
            self.test_resp,
            self.test_resp_ei,
            self.test_data['design']['sub_thick'],
            self.test_xcal,
            self.test_data['calibration']['ycal'],
            self.test_xval,
            self.test_data['validation']['yval'],
            self.test_data['design']['opt_comp'],
            self.test_fit_order
        )

        ## print(np.shape(self.test_data['calibration']['xcal']))
        ## print(np.shape(self.test_data['calibration']['ycal']))
        ## print(np.shape(self.test_data['validation']['xval']))
        ## print(np.shape(self.test_data['validation']['yval']))
        ## print(self.test_data['calibration']['xcal'][4, 4])

        ## plt.figure(figsize=(12, 12))
        ## for i in range(np.shape(self.test_xcal)[0]):
        ##     plt.plot(self.test_data['setup']['wv'], self.test_xcal[i, :])
        ## plt.show()

        print()
        print('SEC: ', _perf['SEC'])
        print('SEP: ', _perf['SEP'])
        print('Gain: ', _perf['gain'])
        print('Offset: ', _perf['offset'])
        print('SNR: ', _perf['SNR'])
        print('AdivB: ', _perf['delta_AdivB'])
        print('AdivB EI: ', _perf['delta_AdivB_EI'])

        nptest.assert_array_almost_equal(self.exp_T, _perf['fil_spec']['T'], decimal=self.precision)

        ### plt.figure(figsize=(8, 8))
        ### plt.plot(self.test_data['setup']['wv'], _perf['fil_spec']['T'])
        ### plt.show()

        ### #print('\n\n', test_perf, '\n\n')
        ### # plot the results
        ### plt.figure(figsize=(12, 6))
        ###
        ### plt.plot(
        ###     range(0, np.shape(test_perf['yhatcal'])[0]),
        ###     np.squeeze(test_perf['yhatcal']),
        ###     label='y hat cal'
        ### )
        ### plt.plot(
        ###     range(0, np.shape(test_moe.cal_data['ycal'])[1]),
        ###     np.squeeze(test_moe.cal_data['ycal']),
        ###     label='y cal'
        ### )
        ### plt.plot(
        ###     range(0, np.shape(test_perf['yhatval'])[0]),
        ###     np.squeeze(test_perf['yhatval']),
        ###     label='y hat val'
        ### )
        ### plt.plot(
        ###     range(0, np.shape(test_moe.val_data['yval'])[1]),
        ###     np.squeeze(test_moe.val_data['yval']),
        ###     label='y val'
        ### )
        ### plt.title('MOE Performance Test')
        ### plt.ylabel('Concentration')
        ### plt.xlabel('Index')
        ### plt.legend(loc='upper right')
        ### plt.show()
        ### # write 'PASSED' to output stream if
        ### # all assertions pass
        ### sys.stdout.write(' [x] NO ASSERTIONS')

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up open resources.
        """
        sys.stdout.write('\nRunning teardown procedure...\nDONE!')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
