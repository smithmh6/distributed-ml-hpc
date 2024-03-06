#!user/bin/python
# -*- coding: utf-8 -*-
"""

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


#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation, Flatten
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import plot_model
#from rl.agents.dqn import DQNAgent
#from rl.policy import EpsGreedyQPolicy
#from rl.memory import SequentialMemory

# import moepy modules
from moepy.moe_agent import MOE

def progress_bar(count, total, status=''):
    # setup progress bar
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


class TestMoeEnv(unittest.TestCase):

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

    def test_moe_env(self):
        """
        Test optimize_design() method.
        """

        sys.stdout.write('\n\nSetting up MOE environment...\n')

        test_angles = [0]

        test_moe_env = MOE(self.data_path)


        print("Action Space: ", test_moe_env.action_space)
        print("Obs Space: ", test_moe_env.observation_space)


        (
            test_res,
            test_xcal,
            test_xval,
            response_ei
        ) = test_moe_env.get_sys_response()

        test_xc_norm, test_xv_norm = test_moe_env.get_spec_norm(test_xcal, test_xval)

        #fig = go.Figure()

        #f = open('test_output.csv', 'w')
        #f.write(
        #    'iteration,sec,snr,sep,msq,nit,time,layers,message,'
        #    + 'layer_0,layer_1,layer_2,layer_3,layer_4,layer_5,layer_6,layer_7,'
        #    + 'layer_8,layer_9,layer_10,layer_11,layer_12,layer_13,layer_14,layer_15'
        #    + 'layer_16,layer_17,layer_18,layer_19,layer_20\n'
        #)

        n = 5
        for i in range(0, n):

            progress_bar(i, n, str(i + 1) + '/100 Reset environment...')

            start = time.time()


            test_moe_env.reset()

            progress_bar(i, n, str(i + 1) + '/100 Generating layers...')

            done = False
            while not done:
                obs, rew, done, info = test_moe_env.step(
                    test_moe_env.action_space.sample(),
                    test_angles,
                    test_res,
                    test_xc_norm,
                    test_xv_norm
                )

            #print('Input Stack: ', test_moe_env.layers)
            #print('Materials: ', test_moe_env.materials)
            #print(len(test_moe_env.layers), len(test_moe_env.materials))



            # build single hidden layer neural network

            #model = Sequential()
            #print("Flattened to: ", test_moe_env.observation_space.shape)
            #
            ## Flattened transforms input array
            #model.add(
            #    Flatten(input_shape=test_moe_env.observation_space.shape)
            #)
            #model.add(Dense(16))
            #model.add(Activation('relu'))
            #model.add(Dense(test_moe_env.action_space[0].n))
            #model.add(Activation('linear'))
            #print("Model Summary: ", model.summary())
            #print("Model Shape: ", tuple(model.output.shape))
            #plot_model(model, to_file='model.png', show_shapes=True)


            # configure and compile our agent. We will set our policy as
            # Epsilon Greedy and our memory as Sequential Memory because
            # we want to store the result of actions we performed and the
            # rewards we get for each action.
            """
            policy = EpsGreedyQPolicy()
            memory = SequentialMemory(
                limit=50000,
                window_length=1
            )

            dqn = DQNAgent(
                model=model, nb_actions=test_moe_env.action_space[0].n,
                memory=memory, nb_steps_warmup=10,
                target_model_update=1e-2, policy=policy
            )
            dqn.compile(
                Adam(lr=1e-3),
                metrics=['mae']
            )
            """

            # visualize the training here for show,
            # but this slows down training quite a lot.
            #dqn.fit(
            #    test_moe_env,
            #    nb_steps=1000,
            #    visualize=True,
            #    verbose=2
            #)


            #dqn.test(
            #    test_moe_env,
            #    nb_episodes=5,
            #    #visualize=True
            #)

            progress_bar(i, n, str(i + 1) + '/100 Optimizing...          ')

            opt_layers, opt_perf, opt_output = test_moe_env.optimize_design(
                test_moe_env.layers,
                test_angles,
                test_moe_env.materials,
                test_res,
                test_xc_norm,
                test_xv_norm
            )


            # calculate the performance
            #test_perf = test_moe_env.get_performance(
            #    opt_layers,
            #    test_angles,
            #    test_moe_env.materials,
            #    test_res,
            #    test_xc_norm,
            #    test_xv_norm
            #)

            test_msq = test_moe_env.get_msq(
                opt_layers,
                test_angles,
                test_moe_env.materials,
                test_res,
                test_xc_norm,
                test_xv_norm
            )

            print("SEC: ", opt_perf['SEC'][0][0])
            print("SNR: ", opt_perf['SNR'][0][0])
            print("SEP: ", opt_perf['SEP'][0][0])
            print("MSQ: ", test_msq)
            print("Iterations: ", opt_output.nit)
            print('Input Stack: ', test_moe_env.layers)
            print("Output Stack: ", opt_layers)

            end = time.time()

            iter_time = end - start

            progress_bar(i+1, n, str(i + 1) + '/100 Writing to file...')

            #f.write(
            #    str(i + 1) + ','
            #    + str(opt_perf['SEC'][0][0]) + ','
            #    + str(opt_perf['SNR'][0][0]) + ','
            #    + str(opt_perf['SEP'][0][0]) + ','
            #    + str(test_msq) + ','
            #    + str(opt_output.nit) + ','
            #    + str(iter_time) + ','
            #    + str(len(opt_layers)) + ','
            #    + str(opt_output.message) + ','
            #    + ",".join([str(l) for l in opt_layers]) + '\n'
            #)



        #f.close()

        """
        fig.add_trace(
            go.Scatter(
                x=np.squeeze(np.real(test_moe_env.setup['wv'])),
                y=np.squeeze(np.real(opt_perf['spec']['T'])),
                mode='lines',
                name=('Transmission')
            )
        )


        fig.update_layout(
            title="Optimization Results",
            xaxis_title="Wavelength (nm)",
            yaxis_title="% T"
        )

        fig.show()
        """



        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('\n [x] NO ASSERTIONS')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
