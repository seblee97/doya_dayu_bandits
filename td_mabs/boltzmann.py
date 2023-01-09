import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from td_mabs import td_mab


class Boltzmann(td_mab.TDMAB):
    def __init__(
        self, n_arms, temperature, learning_rate, q_initialisation=0.0, rng=None
    ):

        self._temperature = temperature
        self._learning_rate = learning_rate
        self._q_initialisation = q_initialisation

        super().__init__(n_arms=n_arms, rng=rng)

        self._setup_values()
        self._setup_optimizer()

    def _setup_values(self):
        if self._q_initialisation > 0.0:
            self._q_initialisation = self._rng.normal(
                scale=self._q_initialisation, size=self._n_arms
            )

        self._qvals = np.ones(self._n_arms) * self._q_initialisation
        self._qvals = jax.device_put(self._qvals)

    def _setup_optimizer(self):
        optimizer = optax.sgd(self._learning_rate)
        self._opt_state = optimizer.init(self._qvals)

        def loss(params, arm, reward):
            loss_val = 0.5 * jnp.mean((reward - params[arm]) ** 2)
            return loss_val

        self._loss = jax.jit(loss)

        def update(params, arm, reward, opt_state):
            d_loss_d_params = jax.grad(self._loss)(params, arm, reward)
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            params = optax.apply_updates(params, updates)
            return params, new_opt_state

        self._update = jax.jit(update)

    def predict_bandits(self):
        return self._qvals, np.ones(self._n_arms)

    def policy(self):
        return np.eye(self._n_arms)[self.play()]
        # return jax.device_get(
        #     rlax.softmax(self._temperature).probs(self._qvals - self._qvals.max())
        # )

    def learning_rate(self, action: int):
        return self._learning_rate

    def temperature(self):
        return self._temperature

    def play(self):
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)

        return jax.device_get(
            rlax.softmax(self._temperature).sample(
                rng_key, self._qvals - self._qvals.max()
            )
        ).ravel()[0]

    def update(self, arm, reward):
        self._arm_seen[arm] = True
        self._qvals, self._opt_state = self._update(
            self._qvals, arm, reward, self._opt_state
        )
