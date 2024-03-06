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
from os import mkdir
from os.path import exists, join
from pathlib import Path
from random import uniform, seed, sample
import sys
from time import perf_counter
from typing import List
import warnings
import matplotlib.pyplot as plt
from matplotlib import colors, lines
import numpy as np
from tff_lib import OpticalMedium
from tqdm import tqdm as pbar

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

    # import package modules
    from moepy import Dataset
    from moepy.utils import iprint, theta_range, epsilon_decay
    from moepy.environment import MoeEnv
    from moepy.agent import MoeAgent

    # start the clock
    t_start = perf_counter()

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


    # generate the range of incident angles
    iprint("Generating range of incident angles")
    inc_angles = theta_range(cfg.getfloat('design', 'incident_angle'),
                                cfg.getfloat('design', 'field_of_view'),
                                cfg.getint('design', 'n_angles'))


    # create substrate object (same for all nodes),
    # convert thickness to nanometers (10**6)
    iprint("Building substrate")
    sub = OpticalMedium(ds.data['waves'],
                        ds.data['sub'],
                        thick=cfg.getfloat('design', 'sub_thick_mm') * 10**6)

    # create infinite incident medium of air (same for all nodes)
    iprint("Building incident medium")
    inc = OpticalMedium(ds.data['waves'],
                        [complex(1) for x in ds.data['waves']])

    # create kwargs for (same for all designs)
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

    # create AI agent environment
    iprint("Setting up MOE training environment")
    env = MoeEnv(ds, sub, inc, **kwargs)

    # set up main model
    iprint("Configuring main network")
    agent = MoeAgent(input_shape=env.observation_space.shape,
                     learning_rate=cfg.getfloat('training', 'learning_rate'),
                     agent_lr=cfg.getfloat('training', 'agent_lr'),
                     discount_factor=cfg.getfloat('training', 'discount_factor'),
                     batch_size=cfg.getint('training', 'batch_size'),
                     rand_seed=cfg.getint('training', 'rand_seed'))

    # set up the target model
    iprint("Configuring the target network")
    target = MoeAgent(input_shape=env.observation_space.shape,
                      learning_rate=cfg.getfloat('training', 'learning_rate'),
                      rand_seed=cfg.getint('training', 'rand_seed'))
    target_model = target.model

    # set up the replay memory with a max size
    replay_memory = deque(maxlen=cfg.getint('training', 'max_replay_size'))
    min_replay_size = cfg.getint('training', 'min_replay_size')

    # Epsilon-greedy algorithm initialized at 1 meaning
    # every step is random at the start
    epsilon = 1
    max_epsilon = 1  # can't explore more than 100%, so max is always 1
    min_epsilon = cfg.getfloat('training', 'min_epsilon')
    decay = cfg.getfloat('training', 'decay')

    # settings used if polyak averaging weights
    tau = cfg.getfloat('training', 'tau')
    polyak = cfg.getboolean('training', 'polyak')

    # configure model update interval
    update_main = cfg.getint('training', 'update_main')
    update_target = cfg.getint('training', 'update_target')
    n_steps = 0

    # statistical information
    rwd_total = []
    n_lyrs = []
    sec_vals = []
    snr_vals = []
    rand_actions = 0
    exploit_actions = 0

    # set random seed if requested
    rand_seed = cfg.getint('training', 'rand_seed')
    if rand_seed > 0:
        iprint(f"Setting random seed= {rand_seed}")
        seed(rand_seed)
        np.random.seed(rand_seed)

    # store the resulting layer stacks
    results = []

    # begin iterating random layer stacks
    iprint("Beginning training algorithm")
    for episode in range(cfg.getint('training', 'episodes')):

        iprint(f"Starting episode [{episode}]")

        # reset environment with 2 random initial layers
        obs, rwd, done, trunc, perf = env.reset()
        n_steps += 2

        # track episode rewards
        total_rewards = 0

        while not done:
            n_steps += 1  # increment steps

            # explore according to Epsilon Greedy Exploration strategy
            r = uniform(0.0, 1.0)

            if r <= epsilon:
                # explore stochastically
                action = env.action_space.sample()
                rand_actions += 1

            else:
                exploit_actions += 1
                # exploit the best known action
                predicted = agent.model.predict(np.expand_dims(obs, axis=0),
                                                verbose=0).flatten()
                index = np.argmax(predicted)
                action = (index, np.array([predicted[index]]))

            # take a new step
            new_obs, rwd, done, trunc, perf = env.step(action)

            # add the state transition to replay memory
            replay_memory.append([obs, action, rwd, new_obs, done, trunc])

            # update the agent with the Bellman Equation
            if n_steps % update_main == 0 or done:

                # check length of replay memory
                if len(replay_memory) > min_replay_size:

                    iprint("Sampling from replay memory")

                    # sample from the replay memory
                    sample_batch = sample(replay_memory, cfg.getint('training', 'batch_size'))

                    # update agent network
                    agent.train(target_model, sample_batch)

            obs = new_obs           # update the observation
            total_rewards += rwd    # sum the reward values


            if n_steps % update_target == 0:
                iprint(f"Updating target network")
                if polyak:
                    weights = []
                    for m, t in zip(agent.model.layers, target_model.layers):
                        w = tau * m.get_weights()[0] + (1 - tau) * t.get_weights()[0]
                        weights.append(w)
                        b = tau * m.get_weights()[1] + (1 - tau) * t.get_weights()[1]
                        weights.append(b)

                    target_model.set_weights(weights)
                else:
                    target_model.set_weights(agent.model.get_weights())

        # statistical information
        results.append(env.moe.stack.layers)
        rwd_total.append(total_rewards)
        n_lyrs.append(env.moe.stack.num_layers)
        sec_vals.append(np.mean(perf['sec']))
        snr_vals.append(np.mean(perf['snr']))

        # update epsilon greedy policy after each episode
        epsilon = epsilon_decay(min_epsilon, max_epsilon, decay, episode)

    # plot the last layer stack
    #env.savefig(join(run_dir, 'stack.png'))

    env.close()

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
    df = epsilon_decay(min_epsilon, max_epsilon, decay, n_episodes)
    fig1, ax1 = plt.subplots()
    ax1.plot(n_episodes, df)
    ax1.set(title='Epsilon Decay Function',
            xlabel='Episodes',
            ylabel='Epsilon')
    ax1.grid()
    plt.tight_layout()
    fig1.savefig(join(run_dir, "decay.png"), dpi=500)

    iprint(f"RWD min= {float(np.min(rwd_total))}  max= {float(np.max(rwd_total))}")
    iprint(f"SEC min= {float(np.min(sec_vals))}  max= {float(np.max(sec_vals))}")
    iprint(f"SNR min= {float(np.min(snr_vals))}  max= {float(np.max(snr_vals))}")
    iprint(f"rand actions= {rand_actions}  exploit actions= {exploit_actions}")
    iprint(f"total steps= {n_steps}")

    # get stop time
    t_end = perf_counter()

    # calculate total run time
    iprint(f"Elapsed time= {round((t_end - t_start) / 60, 3)} mins")

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