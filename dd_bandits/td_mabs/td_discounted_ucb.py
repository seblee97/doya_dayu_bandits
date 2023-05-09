from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dd_bandits import constants
from dd_bandits.td_mabs import td_mab


class DiscountedUCB(td_mab.TDMAB):
    def __init__(
        self,
        num_arms: int,
        rho: float,
        gamma: float,
        learning_rate: float,
        q_initialisation: float,
        scalar_log_spec: List[str],
        rng=None,
    ):

        self._rho = rho
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._q_initialisation = q_initialisation

        self._scalar_log_spec = scalar_log_spec

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
        return self._qvals, np.ones(
            self._num_arms
        )  # np.log(self._step) / (self._step_arm + 1.0)

    def policy(self):
        return np.eye(self._num_arms)[self.play()]

    def play(self):
        # ensure all arms are played at least once initially.
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        ucb_values = np.zeros(self._num_arms)
        for arm in range(self._num_arms):
            if self._step_arm[arm] > 0:
                ucb_values[arm] = np.sqrt(
                    self._rho * np.log(self._step) / self._step_arm[arm]
                )

        action = (jax.device_get(self._qvals) + ucb_values).argmax()
        return action

    def learning_rate(self, action: int):
        return self._learning_rate

    def temperature(self):
        return None

    def update(self, arm, reward):
        self._arm_seen[arm] = True
        self._step *= self._gamma
        self._step_arm[arm] *= self._gamma
        self._qvals, self._opt_state = self._update(
            self._qvals, arm, reward, self._opt_state
        )

    def scalar_log(self):
        log = {}

        if constants.MEAN_EST in self._scalar_log_spec:
            mean = self._qvals.mean()
            mean = self._qvals.mean()
            log[constants.MEAN_EST] = mean
        # if constants.MEAN_VAR in self._scalar_log_spec:
        #     mean_var = self._qvals[..., 0].var(axis=1)
        #     log[constants.MEAN_VAR] = mean_var.mean()
        # if self._use_direct:
        #     # exp since we learn log variances
        #     var_preds = np.exp(self._qvals[..., 1])
        # else:
        #     var_preds = self._qvals[..., 1]
        # if constants.VAR_MEAN in self._scalar_log_spec:
        #     var_mean = var_preds.mean(axis=1)
        #     log[constants.VAR_MEAN] = var_mean.mean()
        # if constants.VAR_MEAN in self._scalar_log_spec:
        #     var_var = var_preds.var(axis=1)
        #     log[constants.VAR_VAR] = var_var.mean()

        # for k, v in self._adaptation_modules.items():
        #     if k in self._scalar_log_spec:
        #         log[k] = v(None)

        return log
