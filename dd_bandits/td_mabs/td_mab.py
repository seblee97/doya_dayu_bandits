import abc

import jax
import numpy as np


class TDMAB(abc.ABC):
    def __init__(self, num_arms: int, rng):

        if rng is None:
            rng = np.random.RandomState()

        self._rng = rng
        self._rng_key = jax.random.PRNGKey(self._rng.randint(1000000))

        self._num_arms = num_arms

        # Total and per-arm step count
        self._step = 0
        self._step_arm = np.zeros(self._num_arms)
        self._arm_seen = np.zeros(self._num_arms, dtype=bool)

    @abc.abstractmethod
    def _setup_optimizer(self):
        pass

    @abc.abstractmethod
    def play(self):
        pass

    @abc.abstractmethod
    def update(self, arm, reward):
        pass

    @abc.abstractmethod
    def predict_bandits(self):
        pass

    @abc.abstractmethod
    def policy(self):
        pass

    @abc.abstractmethod
    def learning_rate(self, action: int):
        pass

    @property
    def min_epistemic_uncertainty(self):
        return None

    @property
    def epistemic_uncertainty(self):
        return None

    @property
    def aleatoric_uncertainty(self):
        return None
