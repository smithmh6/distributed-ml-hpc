"""
This module provides the main command line interface
for running MOE designs.

Example Usage
---------------
>>> moepy-run -c <path-to_config>
"""

# import dependencies
from argparse import ArgumentParser, Namespace
from collections import deque
from configparser import ConfigParser
from mpi4py import MPI as mpi
from os import mkdir
from os.path import exists, join
from pathlib import Path
from random import uniform, seed, sample
import sys
from time import perf_counter
from typing import List
import matplotlib.pyplot as plt
from moepy import Dataset
from moepy.agent import MoeAgent
from moepy.environment import MoeEnv
from moepy.utils import iprint, theta_range, epsilon_decay
import numpy as np
from tff_lib import OpticalMedium

import warnings
warnings.filterwarnings('ignore')

def get_args(args: List[str] = None) -> Namespace:
    """
    Setup argparse and return args Namespace.
    """

    # build arg parser
    parser = ArgumentParser(
        prog='run.py', description="Runs MOE design and generates optimized filter stacks.")
    parser.add_argument("-c", "--config", type=Path)

    # parse the args
    return parser.parse_args(args=args)  # None defaults to sys.argv[1:]

def load_config(args: Namespace) -> ConfigParser:
    """
    Loads the specified config file from args
     and return config Dict.
    """

    # read in the config file
    config = ConfigParser()
    config.read(args.config)

    ########################################################
    ##### insert function to validate config file here #####
    ########################################################

    return config

def run(cfg: ConfigParser) -> int:
    """
    Runs an MOE design.

    args
    ----------
    cfg: Dict, config file
    """

    # get info about MPI communication
    world_comm = mpi.COMM_WORLD
    world_size = world_comm.Get_size()
    rank = world_comm.Get_rank()

    if rank == 0:
        iprint(f'MPI World size is {world_size}.')

        # distribute the workload on each rank
        iprint('Calculating workloads')

        total_episodes = cfg.getint('training', 'episodes')
        wload = total_episodes // (world_size-1)
        iprint(f"Workloads --> {wload} per node")

        # send dataset to compute nodes
        iprint("Sending workload to compute nodes")
        for i in range(1, world_size):
            world_comm.send(wload, dest=i, tag=i)
    else:
        iprint(f"Receiving workload [rank={rank}]")
        wload = world_comm.recv(source=0, tag=rank)


    # configure dataset and send to compute nodes
    if rank == 0:

        # scan working directory, create job folder
        iprint("Scanning working directory")
        pwd = Path(cfg.get('paths', 'path_wd'))
        job_dir = join(pwd, cfg.get('paths', 'name'))
        if not exists(job_dir):
            iprint(f"Creating job directory {job_dir}")
            mkdir(job_dir)

        # check the runlabel directory
        run_dir = join(job_dir, cfg.get('paths', 'runlabel'))
        if not exists(run_dir):
            iprint(f"Creating run directory {run_dir}")
            mkdir(run_dir)

        # load in the dataset
        iprint("Importing dataset")
        ds = Dataset(Path(cfg.get('paths', 'dataset')),
                        cfg.get('input', 'format'))

        # spectral mean
        iprint("Computing spectral mean")
        ds.spectral_mean(cfg.getfloat('input', 'mean_thresh'))

        # normalize, if requested
        norm = cfg.getfloat('input', 'normalize')
        if norm > 0:
            iprint("Normalizing input data")
            ds.normalize_spectra(norm)

        # radiometric response
        iprint("Computing radiometric response")
        ds.radiometric_response(cfg.getboolean('simulation', 'use_ei'))

        ###############################################################
        ##########      Check for plotting options here      ##########
        ###############################################################

        # send dataset to compute nodes
        iprint("Sending dataset to compute nodes")
        for i in range(1, world_size):
            world_comm.send(ds, dest=i, tag=i)
    else:
        iprint(f"Receiving dataset [rank={rank}]")
        ds = world_comm.recv(source=0, tag=rank)

    if rank == 0:
        # compute incident angles
        iprint("Generating range of incident angles")
        inc_angles = theta_range(cfg.getfloat('design', 'incident_angle'),
                                    cfg.getfloat('design', 'field_of_view'),
                                    cfg.getint('design', 'n_angles'))

        # send inc_angles to compute nodes
        iprint("Sending inc_angles to compute nodes")
        for i in range(1, world_size):
            world_comm.send(inc_angles, dest=i, tag=i)
    else:
        iprint(f"Receiving inc_angles [rank={rank}]")
        inc_angles = world_comm.recv(source=0, tag=rank)

    if rank == 0:
        # create substrate object (same for all nodes),
        # convert thickness to nanometers (10**6)
        iprint("Building substrate")
        sub = OpticalMedium(ds.data['waves'],
                            ds.data['sub'],
                            thick=cfg.getfloat('design', 'sub_thick_mm') * 10**6)

        # send substrate to compute nodes
        iprint("Sending substrate to compute nodes")
        for i in range(1, world_size):
            world_comm.send(sub, dest=i, tag=i)

    else:
        iprint(f"Receiving substrate [rank={rank}]")
        sub = world_comm.recv(source=0, tag=rank)

    if rank == 0:
        # create infinite incident medium of air
        # (same for all nodes)
        iprint("Building incident medium")
        inc = OpticalMedium(ds.data['waves'],
                            [complex(1) for x in ds.data['waves']])

        # send incident medium to compute nodes
        iprint("Sending incident medium to compute nodes")
        for i in range(1, world_size):
            world_comm.send(inc, dest=i, tag=i)

    else:
        iprint(f"Receiving incident medium [rank={rank}]")
        inc = world_comm.recv(source=0, tag=rank)


    if rank == 0:
        iprint("Constructing kwargs to define MOE environment")
        kwargs = {
            # kwargs used to define FilmStack
            'spec_res': cfg.getfloat('design', 'resolution'),
            'max_total_thick': cfg.getfloat('design', 'max_total_thick_nm'),
            'max_layers': cfg.getint('design', 'max_layers'),
            'min_layers': cfg.getint('design', 'min_layers'),
            'first_lyr_min_thick': cfg.getfloat('design', 'first_lyr_min_thick_nm'),
            'min_thick': cfg.getfloat('design', 'min_thick_nm'),
            'max_thick': cfg.getfloat('design', 'max_thick_nm'),

            # kwargs needed for MOE design and simulation
            'analysis_type': cfg.get('simulation', 'analysis'),
            'opt_comp': cfg.getint('simulation', 'opt_comp'),
            'fit_order': cfg.getint('simulation', 'fit_order'),
            'use_ei': cfg.getboolean('simulation', 'use_ei'),
            'fom': cfg.getint('error', 'fig_of_merit'),
            'ab_thresh': cfg.getfloat('error', 'a_over_b_per_unit'),
            'sec_thresh': cfg.getfloat('error', 'sec_thresh'),
            'snr_thresh': cfg.getfloat('error', 'snr_thresh'),
            'angles': inc_angles,

            # kwargs for GYM environment
            'render_mode': cfg.get('simulation', 'render'),
            'relative_reward': cfg.getboolean('training', 'relative_reward'),
            'sparse_reward': cfg.getboolean('training', 'sparse_reward'),
            'rand_seed': cfg.getint('training', 'rand_seed')
        }

        # send kwargs to compute nodes
        iprint("Sending kwargs to compute nodes")
        for i in range(1, world_size):
            world_comm.send(kwargs, dest=i, tag=i)

    else:
        iprint(f"Receiving kwargs [rank={rank}]")
        kwargs = world_comm.recv(source=0, tag=rank)



    # configure the target model and replay
    # memory to be stored on rank 0
    if rank == 0:
        # set up an environment on main node
        #iprint(f"Setting up MOE training environment [rank={rank}]")
        env = MoeEnv(ds, sub, inc, **kwargs)

        iprint("Configuring the target network")
        target = MoeAgent(input_shape=env.observation_space.shape,
                        learning_rate=cfg.getfloat('training', 'learning_rate'),
                        rand_seed=cfg.getint('training', 'rand_seed'))
        target_model = target.model
        update_target = cfg.getint('training', 'update_target')

        # set up the replay memory with a max size
        iprint("Configuring replay memory")
        replay_memory = deque(maxlen=cfg.getint('training', 'max_replay_size'))

        # settings used if polyak averaging weights
        tau = cfg.getfloat('training', 'tau')
        polyak = cfg.getboolean('training', 'polyak')

        # statistical information
        rwd_total = []
        n_lyrs = []
        sec_vals = []
        snr_vals = []

        # store the resulting layer stacks
        results = []

        global_episodes = 0
        begin_replay_sampling = False
        while global_episodes < cfg.getint('training', 'episodes'):
            msg = world_comm.recv()
            if msg:

                if not "request mini batch" in msg and not "done" in msg:
                    replay_memory.append(msg)

                if "done" in msg:
                    _, agent_weights, agent_layers, stats, agent_id = msg

                    # statistical information
                    lyrs, rwrds, numlyrs, sec, snr = stats
                    results.append(lyrs)
                    rwd_total.append(rwrds)
                    n_lyrs.append(numlyrs)
                    sec_vals.append(sec)
                    snr_vals.append(snr)


                    global_episodes += 1
                    if global_episodes % 25 == 0:
                        iprint(f"Episode {global_episodes} of {cfg.getint('training', 'episodes')} complete")

                    if global_episodes % update_target == 0:
                        iprint(f"Updating target network [agent_id: {agent_id}]")
                        if polyak:
                            weights = []
                            for m, t in zip(agent_layers, target_model.layers):
                                w = tau * m.get_weights()[0] + (1 - tau) * t.get_weights()[0]
                                weights.append(w)
                                b = tau * m.get_weights()[1] + (1 - tau) * t.get_weights()[1]
                                weights.append(b)
                            target_model.set_weights(weights)
                        else:
                            target_model.set_weights(agent_weights)

                if "request mini batch" in msg:

                    rank = str(msg).replace("request mini batch ", "")
                    rank = int(rank)

                    # check length of replay memory
                    if len(replay_memory) > cfg.getint('training', 'min_replay_size'):
                        if begin_replay_sampling == False:
                            iprint("Begin sampling from replay memory")
                            begin_replay_sampling = True

                        # sample from the replay memory
                        sample_batch = sample(replay_memory, cfg.getint('training', 'batch_size'))

                        # get the next states from mini batch
                        future_states = np.array([transition[3] for transition in sample_batch])

                        # calculate the future Q values
                        future_q_values = target_model.predict(future_states, verbose=0)

                        # send sample_batch & future Q values back to node
                        world_comm.send((sample_batch, future_q_values), dest=rank, tag=rank)
                    else:
                        world_comm.send((None, None), dest=rank, tag=rank)


        # plot training statistics
        fig, ax = plt.subplots(nrows=4, ncols=1)
        n_episodes = np.arange(cfg.getint('training', 'episodes'))
        ax[0].plot(n_episodes, rwd_total, label="Total Rewards")
        ax[1].plot(n_episodes, n_lyrs, label="Num Layers")
        ax[2].plot(n_episodes, sec_vals, label="SEC")
        ax[3].plot(n_episodes, snr_vals, label="SNR")

        ax[0].set(title='Training Results')
        ax[3].set(xlabel='Episodes')

        ax[0].set(ylabel='Rewards')
        ax[1].set(ylabel='Layers')
        ax[2].set(ylabel='SEC')
        ax[3].set(ylabel='SNR')

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()

        #ax[0].legend()
        #ax[1].legend()
        #ax[2].legend()
        #ax[3].legend()

        plt.tight_layout()
        fig.savefig(join(run_dir, "test.png"), dpi=500)

        # plot the decay function
        epsilon = 1
        max_epsilon = 1  # can't explore more than 100%, so max is always 1
        min_epsilon = cfg.getfloat('training', 'min_epsilon')
        decay = cfg.getfloat('training', 'decay')
        df = epsilon_decay(min_epsilon, max_epsilon, decay, n_episodes)
        fig1, ax1 = plt.subplots()
        ax1.plot(n_episodes, df)
        ax1.set(title='Epsilon Decay Function',
                xlabel='Episodes',
                ylabel='Epsilon')
        ax1.grid()
        plt.tight_layout()
        fig1.savefig(join(run_dir, "decay.png"), dpi=500)


    # configure distributed training to run on
    # worker nodes
    else:

        # set up an environment on each worker node
        iprint(f"Setting up MOE training environment [rank={rank}]")
        env = MoeEnv(ds, sub, inc, **kwargs)

        # set up main model on each node
        iprint(f"Configuring agent network [rank={rank}]")
        agent = MoeAgent(input_shape=env.observation_space.shape,
                         learning_rate=cfg.getfloat('training', 'learning_rate'),
                         agent_lr=cfg.getfloat('training', 'agent_lr'),
                         discount_factor=cfg.getfloat('training', 'discount_factor'),
                         batch_size=cfg.getint('training', 'batch_size'),
                         rand_seed=cfg.getint('training', 'rand_seed'))

        # configure epsilon greedy strategy on each node
        # Epsilon-greedy algorithm initialized at 1 meaning
        # every step is random at the start
        iprint(f"Configuring epsilon greedy strategy [rank={rank}]")
        epsilon = 1
        max_epsilon = 1  # can't explore more than 100%, so max is always 1
        min_epsilon = cfg.getfloat('training', 'min_epsilon')
        decay = cfg.getfloat('training', 'decay')

        # configure model update interval
        update_agent = cfg.getint('training', 'update_agent')

        n_steps_agent = 0

        # set random seed if requested
        rand_seed = cfg.getint('training', 'rand_seed')
        if rand_seed > 0:
            iprint(f"Setting random seed= {rand_seed} [rank={rank}]")
            seed(rand_seed)

        # begin iterating random layer stacks
        iprint(f"Beginning training loop [rank={rank}]")
        for episode in range(wload):

            obs, rwd, done, trunc, perf = env.reset()
            total_rewards = 0

            while not done:
                n_steps_agent += 1            # increment steps
                r = uniform(0.0, 1.0)   # generate random number

                # explore according to Epsilon Greedy Exploration strategy
                if r <= epsilon:
                    # explore stochastically
                    action = env.action_space.sample()

                else:
                    # exploit the best known action
                    predicted = agent.model.predict(np.expand_dims(obs, axis=0),
                                                    verbose=0).flatten()
                    index = np.argmax(predicted)
                    action = (index, np.array([np.max(predicted)]))

                # take a new step
                new_obs, rwd, done, trunc, perf = env.step(action)

                #iprint(f"Sending state transition to replay memory [rank={rank}]")
                world_comm.send([obs, action, rwd, new_obs, done, trunc], dest=0, tag=rank)

                # update the agent with the Bellman Equation
                if n_steps_agent % update_agent == 0 or done:
                        sample_batch, future_q_values = world_comm.sendrecv(
                                                            f"request mini batch {rank}",
                                                            dest=0,
                                                            sendtag=rank,
                                                            source=0,
                                                            recvtag=rank)
                        if sample_batch is not None:
                            ##iprint(f"Updating agent network [rank={rank}]")
                            agent.train(sample_batch, future_q_values)

                obs = new_obs           # update the observation
                total_rewards += rwd    # sum the reward values

            # update epsilon greedy policy after each episode
            epsilon = epsilon_decay(min_epsilon, max_epsilon, decay, episode)

            # send the updated agent model object to rank 0
            stats = (
                env.moe.stack.layers,
                total_rewards,
                env.moe.stack.num_layers,
                np.mean(perf['sec']),
                np.mean(perf['snr'])
            )
            world_comm.send(
                ("done", agent.model.get_weights(), agent.model.layers, stats, rank),
                dest=0,
                tag=rank)



        env.close()

    return 0

def main() -> None:
    """
    Entry point for running MOE designs.
    """

    # get command line args
    args = get_args()

    # load config file
    config = load_config(args)

    # run moe design
    sys.exit(run(config))


if __name__ == "__main__":
    main()