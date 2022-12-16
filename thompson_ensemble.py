import jax
import jax.numpy as jnp
import numpy as np
import optax

from random_bandit import RandomMAB


class ThompsonSamplingEnsemble(RandomMAB):
    def __init__(self, n_arms, n_ens, optimizer, mask_p=0.5, Q0=0.01, rng=None):

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self._n_arms = n_arms
        self._n_ens = n_ens
        self._mask_p = mask_p

        if Q0 > 0:
            Q0 = rng.normal(scale=Q0, size=(n_arms, n_ens))

        self.qvals = np.ones((n_arms, n_ens)) * Q0
        self.arm_seen = np.zeros(n_arms, dtype=bool)

        # Total and per-arm step count
        self.step = 0
        self.step_arm = np.zeros(self._n_arms)

        # Optimization setup
        self.qvals = jax.device_put(self.qvals)
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.qvals)
        self._rng_key = jax.random.PRNGKey(rng.randint(1000000))

        def loss(params, arm, reward, rng_key):
            mask = jax.random.bernoulli(rng_key, p=mask_p, shape=(n_ens,)).astype(
                jnp.float32
            )

            loss_val = 0.5 * jnp.sum(mask * (reward - params[arm]) ** 2) / jnp.sum(mask)
            return loss_val

        self._loss = jax.jit(loss)

        def update(params, arm, reward, opt_state, rng_key):
            d_loss_d_params = jax.grad(self._loss)(params, arm, reward, rng_key)
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            params = optax.apply_updates(params, updates)
            return params, new_opt_state

        self._update = jax.jit(update)

    def predict_bandits(self):
        return self.qvals.mean(axis=1), np.ones(self._n_arms)

    def policy(self):
        return np.eye(self._n_arms)[jax.device_get(self.qvals).argmax(0)].mean(0)

    def play(self):
        if not self.arm_seen.all():
            return self.arm_seen.argmin()

        ind = self.rng.choice(self._n_ens, size=self._n_arms)
        return jax.device_get(
            self.qvals[np.arange(self._n_arms), ind].argmax()
        ).ravel()[0]

    def update(self, arm, reward):
        self.arm_seen[arm] = True
        self.step += 1
        self.step_arm[arm] += 1
        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)
        self.qvals, self.opt_state = self._update(
            self.qvals, arm, reward, self.opt_state, rng_key
        )
