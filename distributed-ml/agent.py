"""
The learning agent used in MOE Deep RL.
"""

# import dependencies

import numpy as np
import tensorflow as tf
from .utils import iprint

class MoeAgent():
    """
    MOE Learning Agent.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the learning agent.
        """

        if int(kwargs.get('rand_seed', 0)) > 0:
            tf.random.set_seed(int(kwargs.get('rand_seed')))

        self.learning_rate = float(kwargs.get('learning_rate', 0.01))

        self.init_kernel = tf.keras.initializers.GlorotUniform()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=kwargs.get('input_shape')),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer=self.init_kernel),
            tf.keras.layers.Dense(16, activation='relu', kernel_initializer=self.init_kernel),
            tf.keras.layers.Dense(8, activation='relu', kernel_initializer=self.init_kernel),
            tf.keras.layers.Dense(3, activation='linear', kernel_initializer=self.init_kernel),
        ])

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

        # store kwargs in instance variable
        self.kwargs = kwargs

    def train(self, sample_batch, future_q_values) -> None:
        """
        Update the model using the bellman equation.
        """

        batch_size = int(self.kwargs.get('batch_size', 32))
        discount = float(self.kwargs.get('discount_factor'))
        agent_lr = float(self.kwargs.get('agent_lr', 0.1))

        # extract the current states from mini batch
        current_states = np.array([transition[0] for transition in sample_batch])

        # get Q values from the states of the sample batch
        current_q_values = self.model.predict(current_states, verbose=0)


        x_data = []
        y_data = []

        # update q with bellman equation
        for index, (obs, action, rwd, _new_obs, done, _trunc) in enumerate(sample_batch):
            if not done:
                max_future_q = (rwd + discount * np.max(future_q_values[index]))
            else:
                max_future_q = rwd

            current_qs = current_q_values[index]
            current_qs[action[0]] = (1 - agent_lr) * current_qs[action[0]] + agent_lr * max_future_q

            x_data.append(obs)
            y_data.append(current_qs)

        # update the model weights and activations with the new sample batch
        self.model.fit(np.array(x_data),
                       np.array(y_data),
                       batch_size=batch_size,
                       verbose=0,
                       shuffle=True)
