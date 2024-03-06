"""
MOE Design example.
"""

# import dependencies
import argparse as ap
import numpy as np
from mpi4py import MPI as mpi
from datetime import datetime as dt
#import json
import os
import sys
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from concurrent.futures import ProcessPoolExecutor
#import threading
import queue
import pathlib
from termcolor import colored
from tff_lib.utils import film_matrix

# get path of working directory
pwd = os.getcwd()

# add moepy package path to PATH
sys.path.insert(0, os.path.join(pwd, 'moepy'))
import moe, utils, processing, metrics


# create Arg parser
ARG_PARSER = ap.ArgumentParser(
    prog="design.py",
    description="Example script simulating MOE designs."
)
ARG_PARSER.add_argument("-d", "--designs", type=int)

# define constants
LOG = colored('[LOG]', 'light_blue')

# additional helper functions
def NOW():
     return colored(str(dt.now().isoformat()), 'light_yellow')

def performance_queue(q:queue.Queue):
    """
    """
    while True:
        if not q.empty():
            print(f'{NOW()} {LOG} ---> Queue: {q.get()}')

def main():

    args = ARG_PARSER.parse_args()
    N_DESIGNS = args.designs

    # get basic info about MPI communicator
    world_comm = mpi.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()
    my_name = mpi.Get_processor_name()
    if my_rank == 0:
        print(f'{NOW()} {LOG} [{my_name}] ---> MPI World size == {world_size}')


    # distribute the workload on each rank
    print(f'{NOW()} {LOG} [{my_name}] ---> Calculating workload.')
    sys.stdout.flush()
    wload = [ N_DESIGNS // world_size for i in range(world_size) ]
    for i in range(N_DESIGNS % world_size):
        wload[i] += 1
    my_start = 0
    for i in range(my_rank):
        my_start += wload[i]
    my_end = my_start + wload[my_rank]
    if my_rank == 0:
        print(f'{NOW()} {LOG} [{my_name}] ---> Workloads == {wload}')

    print(f'{NOW()} {LOG} [{my_name}] ---> processor name == {my_name}')
    print(f'{NOW()} {LOG} [{my_name}] ---> my rank == {my_rank}')
    print(f'{NOW()} {LOG} [{my_name}] ---> workload start == {my_start}')
    print(f'{NOW()} {LOG} [{my_name}] ---> workload end == {my_end}')
    sys.stdout.flush()


    ## perform pre-processing only once on a single node
    ## this section is not as computationally demanding as
    ## the actual optimization piece
    if my_rank == 0:

        # path to data file, output path to json
        f_path = pathlib.Path(r'./data/test_uster_plastic_moe/Uster_Plastic-ID_MOE6.xlsx')
        f_out = pathlib.Path(r'./data/test_uster_plastic_moe/Uster_Plastic-ID_MOE6.json')

        print(f'{NOW()} {LOG} [{my_name}] ---> Reading excel file from {f_path}')
        sys.stdout.flush()
        # read the excel file
        df = processing.excel_to_json(f_path, f_out)

        print(f'{NOW()} {LOG} [{my_name}] ---> Converting data to numpy.ndarray.')
        sys.stdout.flush()
        # pre-process the data into numpy arrays
        data = processing.convert_to_numpy(df)

        # Compute the range of angles with default N points = 7
        angles = utils.theta_range(data['design']['incident_angle'], data['design']['fov'])
        print(f'{NOW()} {LOG} [{my_name}] ---> Theta range= {angles}')
        sys.stdout.flush()

        # calculate mean of cal/val spectral data
        avg, x_cal, x_val = utils.spec_mean(data['calibration']['xcal'], data['validation']['xval'])
        print(f'{NOW()} {LOG} [{my_name}] ---> Spectral Mean= {avg}')
        sys.stdout.flush()

        # extract optical components from 'setup'
        print(f'{NOW()} {LOG} [{my_name}] ---> Extracting optical components from setup.')
        sys.stdout.flush()
        exclude = ('high_material', 'low_material', 'substrate', 'env_int', 'wv')
        components = {k: v for k, v in data['setup'].items() if not k in exclude}

        print(f'{NOW()} {LOG} [{my_name}] ---> Calculating radiometric response.')
        sys.stdout.flush()
        # calculate radiometric response
        resp, resp_ei = utils.system_response(data['setup']['wv'], components, data['setup']['env_int'])

        print(f'{NOW()} {LOG} [{my_name}] ---> Normalizing spectral data.')
        sys.stdout.flush()
        # normalize the spectral data (axis=0 by default ---> no normalization)
        x_cal, x_val = utils.spec_norm(x_cal, x_val)

        print(f'{NOW()} {LOG} [{my_name}] ---> Convolving system response and spectral data.')
        sys.stdout.flush()
        # convolve the system response with the spectral data
        x_cal = data['calibration']['xcal'] * resp_ei
        x_val = data['validation']['xval'] * resp_ei

        # calculate the estimated total physical thickness
        est_thickness = utils.total_thickness(
            data['setup']['high_material'], data['setup']['low_material'], data['design']['spec_res'])
        print(f'{NOW()} {LOG} [{my_name}] ---> Estimated thickness= {est_thickness} nanometers')
        sys.stdout.flush()


        print(f'{NOW()} {LOG} [{my_name}] ---> Generating spectral data plot.')
        sys.stdout.flush()
        # plot the spectral data and system response
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Radiometric Response', 'Calibration', 'Validation'))
        fig.add_trace(go.Scatter(x=data['setup']['wv'], y=resp, name='w/o EI', legendgroup='response', legendgrouptitle_text='Response Group'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['setup']['wv'], y=resp_ei, name='w/ EI', legendgroup='response', line={'dash':'dash'}), row=1, col=1)

        for i in range(x_cal.shape[0]):
            fig.add_trace(go.Scatter(x=data['setup']['wv'], y=x_cal[i, :], showlegend=False), row=2, col=1)

        for i in range(x_val.shape[0]):
            fig.add_trace(go.Scatter(x=data['setup']['wv'], y=x_val[i, :], showlegend=False), row=3, col=1)

        fig.update_xaxes(title_text="Wavelength(nm)", row=3, col=1)
        fig.update_yaxes(title_text="T", row=2, col=1)
        fig.update_layout(
            template='plotly_dark',
            height=800,
            width=1000
        )

        # save image to file, then display
        fig.write_image('./examples/out/data.png')

        # create an iterable for the thread pool executer
        print(f'{NOW()} {LOG} [{my_name}] ---> Generating {N_DESIGNS} random layer stacks.')
        sys.stdout.flush()
        inputs = []
        for i in range(N_DESIGNS):
            inputs.append((utils.random_layers(5, 20, est_thickness), angles, data, x_cal, x_val, resp, resp_ei))

        # send data to other nodes
        for i in range(1, world_size):
             world_comm.send(inputs, dest=i, tag=i)
             print(f'{NOW()} {LOG} [{my_name}] ---> Sending input layer stacks to rank {i}.')

    else:
        # receive data on other nodes
        inputs = world_comm.recv(source=0, tag=my_rank)
        print(f'{NOW()} {LOG} [{my_name}] ---> Received input layer stacks on rank {my_rank}.')

    sys.stdout.flush()

    ### uncomment this block to use the output Queue
    #outputs = queue.Queue()
    #threading.Thread(target=performance_queue, args=(outputs, ), daemon=True).start()

    ### Uncomment this section to use ProcessPool
    ## print(f'{NOW()} {LOG} [{my_name}] ---> Starting {N_DESIGNS} design processes.')
    ## # run the simulations with a thread pool executer
    ## outputs = []
    ## with ProcessPoolExecutor(max_workers=8) as ex:
    ##     start_time = time.perf_counter()
    ##     for n, lyrs in ex.map(moe.run_optimizer, inputs):
    ##         print(f'{NOW()} {LOG} [{my_name}] ---> Iters= {n} Layers= {len(lyrs)}')
    ##         outputs.append((n, lyrs))
    ##     end_time = time.perf_counter()
    ##     print(f'{NOW()} {LOG} [{my_name}] ---> Completed {N_DESIGNS} designs in {(end_time - start_time)/60} minutes.')

    ## mpi enabled designs
    print(f'{NOW()} {LOG} [{my_name}] ---> Starting design processes with rank [{my_rank}].')
    sys.stdout.flush()
    outputs = []
    for input_stack in inputs[my_start:my_end]:
        t1 = time.perf_counter()
        n_opt, lyrs_opt = moe.run_optimizer(input_stack)
        t2 = time.perf_counter()
        print(f'{NOW()} {LOG} [{my_name}] ---> Optimization completed in {(t2 - t1)} seconds.')
        print(f'{NOW()} {LOG} [{my_name}] ---> Iters= {n_opt} Layers= {len(lyrs_opt)}')
        sys.stdout.flush()
        outputs.append((n_opt, lyrs_opt))


    global_outputs = world_comm.gather(outputs, root=0)

    if my_rank == 0:

        print(f'{NOW()} {LOG} [{my_name}] ---> Calculating performance results..')
        # calculate the performance for each result
        results = []
        timers = []
        for opt_output in global_outputs:
            for out in opt_output:
                materials = ["H" if i % 2 == 0 else "L" for i in range(len(out[1]))]
                stack = [(m, l) for m, l in zip(materials, out[1])]
                films = film_matrix(stack, data['setup']['high_material'], data['setup']['low_material'])
                inc_med = np.ones(np.shape(data['setup']['wv']))

                t1 = time.perf_counter()
                perf = moe.performance(data['setup']['wv'],
                        out[1], films, data['setup']['substrate'], inc_med, angles, resp, resp_ei,
                        data['design']['sub_thick'], x_cal, data['calibration']['ycal'],
                        x_val, data['validation']['yval'], data['design']['opt_comp'])
                t2 = time.perf_counter()
                print(f'{NOW()} {LOG} [{my_name}] ---> Calculated performance in {(t2 - t1)} seconds.')
                timers.append(t2 - t1)

                results.append((perf['SEC'][0], perf['SEP'][0], perf['SNR'][0], perf['fil_spec']['T'], len(out[1]), out[0], out[1].flatten().tolist()))

        print(f'{NOW()} {LOG} [{my_name}] ---> Average performance calculation done in {np.mean(timers)} seconds.')

        print(f'{NOW()} {LOG} [{my_name}] ---> Writing results to file..')
        # Write results to csv file, without transmission spectrum
        with open('./examples/out/results.csv', 'w') as outfile:
            outfile.write('SEC,SEP,SNR,Iters,Layers')
            for i in range(20):
                outfile.write(f",Layer_{i + 1}")
            outfile.write('\n')
            for r in results:
                row = [r[0], r[1], r[2], r[5], r[4]]
                for l in r[6]:
                    row.append(l)
                outfile.write(f"{','.join([str(x) for x in row])}\n")

        # read csv as dataframe
        df = pd.read_csv('./examples/out/results.csv')

        # use min(results) to show the lowest SEC
        # use results[index] to plot a specific result
        best = min(results)

        test_layers = [478.2,343.6,496.4,739.4,456.7,186.8,660.7,558.4,254.9,208.6,466.2,644.9,166.2,396.8]
        test_materials = ["H" if i % 2 == 0 else "L" for i in range(len(test_layers))]
        test_stack = [(m, l) for m, l in zip(test_materials, test_layers)]
        test_films = film_matrix(test_stack, data['setup']['high_material'], data['setup']['low_material'])

        test_perf = moe.performance(data['setup']['wv'],
                test_layers, test_films, data['setup']['substrate'], inc_med, angles, resp, resp_ei,
                data['design']['sub_thick'], x_cal, data['calibration']['ycal'],
                x_val, data['validation']['yval'], data['design']['opt_comp'])

        # calculate total thicknesses
        test_total_thick = np.sum(test_layers)
        sim_total_thick = np.sum(best[6])

        print(f'{NOW()} {LOG} [{my_name}] ---> Test: Layers= {len(test_layers)}  Total Thick= {round(test_total_thick, 2)}  SEC= {round(test_perf["SEC"][0], 2)}  SEP= {round(test_perf["SEP"][0], 2)}  SNR= {round(test_perf["SNR"][0], 2)}')
        print(f'{NOW()} {LOG} [{my_name}] ---> Sim: Layers= {best[4]}  Total Thick= {round(sim_total_thick, 2)}  SEC= {round(best[0], 2)}  SEP= {round(best[1], 2)}  SNR= {round(best[2], 2)}  Iters= {best[5]}')
        sys.stdout.flush()

        start_stack = inputs[results.index(best)][0]
        print(f'{NOW()} {LOG} [{my_name}] ---> Optimizations:')
        for j, ss in enumerate(start_stack):
                # calculate layer difference
                diff = round((abs(ss[1] - best[6][j])/ss[1]*100), 2)
                if ss[1] < best[6][j]:
                        sign = '+'
                else:
                        sign = '-'
                print(f'{NOW()} {LOG} [{my_name}] ---> [{j}] {round(ss[1], 2)} -----> {round(best[6][j], 2)}  ({sign}{diff} %)')

        # plot the best performance and the test sample
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['setup']['wv'], y=best[3], name='Simulated'))
        fig.add_trace(go.Scatter(x=data['setup']['wv'], y=test_perf['fil_spec']['T'], name='Test'))
        fig.update_layout(
                template='plotly_dark',
                height=500,
                width=1000,
                title={'text':'Simulation Results', 'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
                xaxis_title="Wavelength (nm)",
                yaxis_title="T"
                'Simulation Results')

        # save image to file, then display
        fig.write_image('./examples/out/results.png')

        print(f'{NOW()} {LOG} [{my_name}] ---> Design complete!')
        print('-'*80)
        sys.stdout.flush()

if __name__ == '__main__':

    # set random seed numpy
    random.seed(123)

    main()
