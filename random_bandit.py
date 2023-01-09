import jax
import jax.numpy as jnp

# import mab
import numpy as np
import optax

# class RandomMAB(mab.MAB):
#     def __init__(self, n_arms: int, rng):
#         super().__init__(n_arms=n_arms, rng=rng)

#     def play(self):
#         return np.random.choice(self._n_arms)

#     def update(self, arm, reward):
#         pass

#     def learning_rate(self, arm):
#         return 1.0

#     def policy(self):
#         return np.ones(self._n_arms) / self._n_arms

#     def predict_bandits(self):
#         return np.zeros(self._n_arms), np.ones(self._n_arms)


class RandomMAB:
    def __init__(self, n_arms):
        self._n_arms = n_arms

    def play(self):
        return np.random.choice(self._n_arms)

    def update(self, arm, reward):
        pass

    def learning_rate(self, arm):
        return 1.0

    def policy(self):
        return np.ones(self._n_arms) / self._n_arms

    def predict_bandits(self):
        return np.zeros(self._n_arms), np.ones(self._n_arms)
