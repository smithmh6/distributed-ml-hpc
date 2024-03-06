#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the GYM environment for
an MOE design using Deep Q Learning.
"""

# import dependencies
from typing import Tuple, SupportsFloat, Dict, Any
from gym import Env, spaces
from gym.core import ActType, ObsType
import matplotlib.pyplot as plt
from matplotlib import lines, colors
import numpy as np
from tff_lib import ThinFilm, FilmStack, OpticalMedium
from .moe import MOE
from .dataset import Dataset
from .utils import iprint, ssq_prediction_error

class MoeEnv(Env):
    """
    This class builds a GYM environment to be used
    in machine learning applications.

    Attributes
    ----------
    moe: MOE, the MOE object (initialized with empty stack)
    """

    def __init__(
            self,
            dat: Dataset,
            sub: OpticalMedium,
            inc: OpticalMedium,
            **kwargs
    ) -> None:
        """
        Set up the GYM environment.
        """

        # setup the OpenAI gym environment
        # action_space and observation_space must be
        # set in all subclasses of Env

        # define action space
        # actions can be:
        #  - add high index material
        #  - add low index material
        #  - take no action
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(low=0, high=1, shape=(1,))
        ))

        # define the lower bound of the observation/state space
        # [sec, snr, sep, num_layers, total_thick, %T]
        obs_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                float(kwargs.get('min_layers')),
                float(kwargs.get('first_lyr_min_thick')),
                0.0
            ],
            dtype=np.float32
        )

        # define the upper bound of the state space
        obs_high = np.array(
            [
                1.0,
                np.inf,
                1.0,
                float(kwargs.get('max_layers')),
                float(kwargs.get('max_total_thick')),
                1.0
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # set render mode (default None)
        self.render_mode = kwargs.pop('render_mode', None)

        # set the reward mode
        self.relative_reward = bool(kwargs.get('relative_reward', False))
        self.sparse_reward = bool(kwargs.get('sparse_reward', True))

        # create a plot object for rendering
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        # set the dataset and moe objects
        self.dat = dat

        # initialize an MOE with an empty layer stack
        self.moe = MOE(sub, FilmStack([], **kwargs), inc)

        # save the kwargs for later use
        self.kwargs = kwargs

    def reset(self):
        """
        Resets the environment prior to an episode.
        Creates a starting stack with 2 random layers.
        """

        # set termination/truncation flags
        self.terminated = False
        self.truncated = False

        # reset reward to 0
        self.reward = 0

        # reset the moe stack
        for i in range(self.moe.stack.num_layers):
            self.moe.stack.remove_layer()

        #self.moe.stack = FilmStack([], **self.kwargs)

        # add the first layer to the stack -- high index material
        action0 = self.action_space.sample()

        # scale/de-normalize the thickness value
        t_scaled0 = ((self.moe.stack.max_thick - self.moe.stack.min_thick)
                    * action0[1][0] + self.moe.stack.min_thick)

        # enforce first layer minimum thickness
        if t_scaled0 < self.moe.stack.first_lyr_min_thick:
            t_scaled0 += self.moe.stack.first_lyr_min_thick

        lyr_0 = ThinFilm(self.dat.data['waves'],
                         self.dat.data['high'],
                         thick=round(t_scaled0, 4),
                         ntype=1)

        # add new layer to stack
        self.moe.stack.append_layer(lyr_0)

        # add a second layer to the stack -- low index material
        action1 = self.action_space.sample()

        # enforce minimum thickness
        t_scaled1 = ((self.moe.stack.max_thick - self.moe.stack.min_thick)
                    * action1[1][0] + self.moe.stack.min_thick)
        if t_scaled1 < self.moe.stack.min_thick:
            t_scaled1 += self.moe.stack.min_thick

        lyr_1 = ThinFilm(self.dat.data['waves'],
                         self.dat.data['low'],
                         thick=round(t_scaled1, 4),
                         ntype=0)

        self.moe.stack.append_layer(lyr_1)

        # calculate the performance of the current stack
        self.perf = self.moe.performance(self.dat, **self.kwargs)

        self.reward = 0
        if not self.sparse_reward:
            self.reward = self.reward_func(self.dat,
                                           self.perf,
                                           self.moe.stack.num_layers,
                                           **self.kwargs)

        self.observation = [np.mean(self.perf['sec']),
                            np.mean(self.perf['snr'] * 10**-3),
                            np.mean(self.perf['sep']),
                            self.moe.stack.num_layers,
                            self.moe.stack.total_thick,
                            np.mean(self.perf['fil_spec']['T'])]

        if self.render_mode == "human":
            self.render(reset=True)

        return self.observation, self.reward, self.terminated, self.truncated, self.perf

    def step(
            self,
            action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Runs one timestep of environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, terminated, truncated, info).

        Parameters
        -------------
        action (object): an action provided by the agent.

        Returns
        -------------
        observation, reward, terminated, truncated, info
        """

        # terminate if action[0] == 0
        self.terminated = True if action[0] == 0 else False

        # scale/de-normalize the thickness value
        t_scaled = ((self.moe.stack.max_thick - self.moe.stack.min_thick)
                    * action[1][0] + self.moe.stack.min_thick)

        # check total thickness before adding layer to stack
        remaining_thick = self.moe.stack.max_total_thick - self.moe.stack.total_thick
        if t_scaled > remaining_thick:
            self.terminated = True
            self.truncated = True

        # terminate if max layers reached
        if self.moe.stack.num_layers == self.moe.stack.max_layers:
            self.terminated = True
            self.truncated = True

        if t_scaled == 0 or t_scaled < -1 or -1 < t_scaled < 0:
            self.terminated = True
            self.truncated = True

        # if design not terminated/truncated, add layer
        if not self.terminated and not self.truncated:

            # get last layer from stack
            last_lyr = self.moe.stack.get_layer(-1)

            # if action[0] matches last_ntype, combine thickness values
            if last_lyr.ntype == action[0]:
                # Remove last layer
                last_lyr = self.moe.stack.remove_layer(-1)
                t = last_lyr.thick
                last_lyr.thick = round(t + t_scaled, 4)

                # add layer back into stack
                self.moe.stack.append_layer(last_lyr)

            else:
                new_nref = self.dat.data['low'] if action[0] == 1 else self.dat.data['high']
                new_lyr = ThinFilm(self.dat.data['waves'],
                                   new_nref,
                                   thick=round(t_scaled, 4),
                                   ntype=0 if action[0] == 1 else 1)

                # add new layer to stack
                self.moe.stack.append_layer(new_lyr)

        # calculate the performance of the current stack
        self.perf = self.moe.performance(self.dat, **self.kwargs)

        # calculate the current observed state
        self.observation = [np.mean(self.perf['sec']),
                            np.mean(self.perf['snr'] * 10**-3),
                            np.mean(self.perf['sep']),
                            self.moe.stack.num_layers,
                            self.moe.stack.total_thick,
                            np.mean(self.perf['fil_spec']['T'])]

        previous_reward = self.reward
        self.reward = 0
        # calculate reward if sparse rewards disabled
        if self.terminated or self.truncated or not self.sparse_reward:
            new_reward = self.reward_func(self.dat,
                                          self.perf,
                                          self.moe.stack.num_layers,
                                          **self.kwargs)

            # compute the reward independently or compute as
            # difference from previous award
            if self.relative_reward:
                self.reward = new_reward - previous_reward
            else:
                self.reward = new_reward

        if self.render_mode == "human":
            self.render()

        return self.observation, self.reward, self.terminated, self.truncated, self.perf

    def render(self, reset=False) -> None:
        """
        Render the current state of the environment.
        """

        clrs = list(colors.TABLEAU_COLORS.keys())
        width = 0.75

        if reset:
            # clear the previous layer stack
            self.ax[0].clear()

        # plot the simulated layer stack
        for i, lyr in enumerate(self.moe.stack.layers):
            self.ax[0].bar("obs",
                    lyr,
                    width,
                    bottom=np.sum(self.moe.stack.layers[:i]),
                    color=clrs[2 if i % 2 == 0 else 3])

        # make a 2-element list of legend handles
        self.ax[0].legend(handles=[
            lines.Line2D([0],
                         [0],
                         color=clrs[2 if i % 2 == 0 else 3],
                         linewidth=5,
                         label='H' if i % 2 == 0 else 'L')
                         for i in range(2)])
        # hide x-axis tick marks
        self.ax[0].xaxis.set_ticks([])
        self.ax[0].set(xlabel=f"{self.moe.stack.num_layers} layers",
                       ylabel="Layers [nm]",
                       title="MOE Film Stack")
        #self.ax[0].grid()

        ## plot the resulting spectra
        self.ax[1].clear()
        self.ax[1].plot(self.dat.data['waves'], self.perf['fil_spec']['T'])
        #self.ax[1].legend(["obs"])
        self.ax[1].set(xlabel="Wavelength [nm]",
                       ylabel="Transmission",
                       title="MOE Performance")
        self.ax[1].grid()

        # set axis limits
        self.ax[0].set_ylim([0, self.moe.stack.max_total_thick])
        self.ax[1].set_ylim([0, 1])

        plt.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.01)
            plt.show(block=False)

    @staticmethod
    def reward_func(dat, perf, n_layers, **kwargs) -> float:
        """
        Calculate the reward of the current step.
        """
        weights = [1.0, 0, 0]

        # scale the SNR value for reward function
        snr = float(1 - np.mean(perf['snr']) * 10**-3)

        # combine the y-calibration and y-validation data
        y_cal_val = np.concatenate((dat.data['ycal'], dat.data['yval']), axis=0)

        # calculate the prediction error
        err = float(1 - ssq_prediction_error(y_cal_val, perf, **kwargs))

        # penalize the design for high numbers of layers (>10)
        # scale the number of layers between 0-1
        lyr_penalty = 0
        lyr_thresh = 12
        if n_layers > lyr_thresh:
            lyr_penalty = -((n_layers - lyr_thresh) /
                            (kwargs.get('max_layers') - lyr_thresh))

        # compute and return the reward
        return err * weights[0] + snr * weights[1] + lyr_penalty * weights[2]

    def savefig(self, out: str):
        """
        Save the final plot to disk.
        """

        clrs = list(colors.TABLEAU_COLORS.keys())
        width = 0.75

        # plot the simulated layer stack
        for i, lyr in enumerate(self.moe.stack.layers):
            self.ax[0].bar("obs",
                    lyr,
                    width,
                    bottom=np.sum(self.moe.stack.layers[:i]),
                    color=clrs[2 if i % 2 == 0 else 3])

        # make a 2-element list of legend handles
        self.ax[0].legend(handles=[
            lines.Line2D([0],
                         [0],
                         color=clrs[2 if i % 2 == 0 else 3],
                         linewidth=5,
                         label='H' if i % 2 == 0 else 'L')
                         for i in range(2)])
        # hide x-axis tick marks
        self.ax[0].xaxis.set_ticks([])
        self.ax[0].set(xlabel=f"{self.moe.stack.num_layers} layers",
                       ylabel="Layers [nm]",
                       title="MOE Film Stack")
        self.ax[0].grid()

        ## plot the resulting spectra
        self.ax[1].clear()
        self.ax[1].plot(self.dat.data['waves'], self.perf['fil_spec']['T'])
        #self.ax[1].legend(["obs"])
        self.ax[1].set(xlabel="Wavelength [nm]",
                       ylabel="Transmission",
                       title="MOE Performance")
        self.ax[1].grid()
        plt.tight_layout()
        self.fig.savefig(out)

    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.fig is not None:
            self.fig.clear()

