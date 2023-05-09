import jax
import jax.numpy as jnp
import numpy as np
import optax

from dd_bandits.td_mabs import td_mab


class EpsilonGreedy(td_mab.TDMAB):
    def __init__(self, num_arms, learning_rate, epsilon, q_initialisation, rng=None):

        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._q_initialisation = q_initialisation

        super().__init__(num_arms=num_arms, rng=rng)

        self._setup_values()
        self._setup_optimizer()

    def _setup_values(self):
        if self._q_initialisation > 0.0:
            self._q_initialisation = self._rng.normal(
                scale=self._q_initialisation, size=self._num_arms
            )

        self._qvals = np.ones(self._num_arms) * self._q_initialisation
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
        return self._qvals, np.ones(self._num_arms)

    def policy(self):
        a = jax.device_get(self._qvals.argmax()).ravel()[0]
        return (1 - self._epsilon) * np.eye(self._num_arms)[a] + (
            self._epsilon / self._num_arms
        )

    def learning_rate(self, action: int):
        return self._learning_rate

    def temperature(self):
        return None

    def play(self):
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        if self._rng.random() <= self._epsilon:
            return self._rng.choice(self._num_arms)

        return jax.device_get(self._qvals.argmax()).ravel()[0]

    def update(self, arm, reward):
        self._arm_seen[arm] = True
        self._qvals, self._opt_state = self._update(
            self._qvals, arm, reward, self._opt_state
        )
