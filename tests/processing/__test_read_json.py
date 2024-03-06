
# import external packages
import unittest
import sys
import json
import numpy as np

# test reading json input data file
class TestReadJson(unittest.TestCase):

    # set up the test case
    def setUp(self):

        # path to data file... hard coded for testing purposes only
        self.data_file = r'E:\source_code\devops\moepy\data\test_dataset_1\test_design_input_params.json'

    # test case to read the input data
    def test_read_input_params(self):

        # initialize input_data dictionary
        input_data = {}

        # use json API to load in file
        with open(self.data_file) as test_data:
            # set input_data with json object
            input_data = json.load(test_data)

        test_array = np.ones(np.shape(input_data['initial_conditions']['filter_1']))

        # multiply all filter arrays and create a new array in design_params
        # initialize an array of 1's that matches shape of 'filter_1' array
        filter_product = np.ones(np.shape(input_data['initial_conditions']['filter_1']))

        # iterate filters
        for i in range(1, 11):
            filter_name = 'filter_' + str(i)
            filter_product *= input_data['initial_conditions'][filter_name]

        input_data['initial_conditions']['filter_product'] = filter_product

        # pick some params to write to output stream
        """
        sys.stdout.write(str(input_data['design_settings']['design_description'])
                        + str(input_data['initial_conditions']['init_layers'])
                        + str(input_data['initial_conditions']['wv'])
                        + str(np.shape(input_data['initial_conditions']['wv']))
                        + str(np.shape(test_array)))
        """
        sys.stdout.write(str(np.shape(input_data['spectral_data']['xc']))
                            + str(np.shape(input_data['spectral_data']['xv']))
                            + str(np.shape(input_data['spectral_data']['ycal']))
                            + str(np.shape(input_data['spectral_data']['yval']))
                            + str(input_data['initial_conditions']['filter_product']))        


if __name__=='__main__':
    unittest.main()