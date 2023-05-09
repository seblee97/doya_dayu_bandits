import jax
import jax.numpy as jnp
import numpy as np
import optax

from random_bandit import RandomMAB


class ThompsonSamplingGaussian(RandomMAB):
    def __init__(self, num_arms, optimizer, Q0=0.0, rng=None):

        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng
        self._num_arms = num_arms

        if Q0 > 0.0:
            Q0 = rng.normal(scale=Q0, size=num_arms)

        self.qvals = np.ones(num_arms) * Q0
        self.arm_seen = np.zeros(num_arms, dtype=bool)

        # Total and per-arm step count
        self.step = 0
        self.step_arm = np.zeros(self._num_arms)

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
        return self.qvals, 1.0 / (self.step_arm + 1.0)

    def policy(self):
        samples = self.rng.normal(
            jax.device_get(self.qvals),
            np.sqrt(1.0 / (self.step_arm + 1.0)),
            size=(100, self._num_arms),
        )
        return np.eye(self._num_arms)[samples.argmax(-1)].mean(0)

    def play(self):
        if not self.arm_seen.all():
            return self.arm_seen.argmin()

        sampled = self.rng.normal(
            jax.device_get(self.qvals), np.sqrt(1.0 / (self.step_arm + 1.0))
        )
        return sampled.argmax()

    def update(self, arm, reward):
        self.arm_seen[arm] = True
        self.step += 1
        self.step_arm[arm] += 1
        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)
        self.qvals, self.opt_state = self._update(
            self.qvals, arm, reward, self.opt_state, rng_key
        )
