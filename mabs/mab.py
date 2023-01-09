import abc

import jax
import numpy as np


class MAB(abc.ABC):
    def __init__(self, n_arms: int, rng):

        if rng is None:
            rng = np.random.RandomState()

        self._rng = rng
        self._rng_key = jax.random.PRNGKey(self._rng.randint(1000000))

        self._n_arms = n_arms

        self._rewards = np.zeros(self._n_arms)

        # Total and per-arm step count
        self._step = 0
        self._step_arm = np.zeros(self._n_arms)
        self._arm_seen = np.zeros(self._n_arms, dtype=bool)

    @abc.abstractmethod
    def play(self):
        pass

    def update(self, arm, reward):
        self._arm_seen[arm] = True

        self._update(arm=arm, reward=reward)

        self._step += 1
        self._step_arm[arm] += 1
        self._rewards[arm] += reward

    @abc.abstractmethod
    def _update(self, arm, reward):
        pass

    @abc.abstractmethod
    def predict_bandits(self):
        pass

    @abc.abstractmethod
    def policy(self):
        pass

    def learning_rate(self, action: int):
        return None

    @property
    def min_epistemic_uncertainty(self):
        return None

    @property
    def epistemic_uncertainty(self):
        return None

    @property
    def aleatoric_uncertainty(self):
        return None
