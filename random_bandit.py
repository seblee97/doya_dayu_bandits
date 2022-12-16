import abc

import jax
import jax.numpy as jnp
import numpy as np
import optax


class RandomMAB(abc.ABC):
    def __init__(self, n_arms: int):
        self._n_arms = n_arms

    def play(self):
        return np.random.choice(self._n_arms)

    @abc.abstractmethod
    def update(self, arm, reward):
        pass

    def learning_rate(self, arm):
        return 1.0

    def policy(self):
        return np.ones(self._n_arms) / self._n_arms

    def predict_bandits(self):
        return np.zeros(self._n_arms), np.ones(self._n_arms)
