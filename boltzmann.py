import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from random_bandit import RandomMAB


class Boltzmann(RandomMAB):
    def __init__(self, n_arms, temperature, optimizer, Q0=0.001, rng=None):

        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng
        self._n_arms = n_arms
        self._temperature = temperature

        if Q0 > 0.0:
            Q0 = rng.normal(scale=Q0, size=n_arms)

        self.qvals = np.ones(n_arms) * Q0
        self.arm_seen = np.zeros(n_arms, dtype=bool)

        # Optimization setup
        self.qvals = jax.device_put(self.qvals)
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.qvals)
        self._rng_key = jax.random.PRNGKey(rng.randint(1000000))

        def loss(params, arm, reward, rng_key):
            loss_val = 0.5 * jnp.mean((reward - params[arm]) ** 2)
            return loss_val

        self._loss = jax.jit(loss)

        def update(params, arm, reward, opt_state, rng_key):
            d_loss_d_params = jax.grad(self._loss)(params, arm, reward, rng_key)
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            params = optax.apply_updates(params, updates)
            return params, new_opt_state

        self._update = jax.jit(update)

    def predict_bandits(self):
        return self.qvals, np.ones(self._n_arms)

    def policy(self):
        return jax.device_get(
            rlax.softmax(self._temperature).probs(self.qvals - self.qvals.max())
        )

    def play(self):
        if not self.arm_seen.all():
            return self.arm_seen.argmin()

        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)

        return jax.device_get(
            rlax.softmax(self._temperature).sample(
                rng_key, self.qvals - self.qvals.max()
            )
        ).ravel()[0]

    def update(self, arm, reward):
        self.arm_seen[arm] = True
        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)
        self.qvals, self.opt_state = self._update(
            self.qvals, arm, reward, self.opt_state, rng_key
        )
