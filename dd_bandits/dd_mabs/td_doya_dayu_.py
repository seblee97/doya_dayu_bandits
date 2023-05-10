import functools
from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from dd_bandits import constants
from dd_bandits.dd_mabs.adaptation_modules import (
    constant_adaptation,
    cumulative_sum_module,
    deup_module,
    likelihood_shift_adaptation,
    reliability_index_adaptation,
)


class DoyaDayu:

    DUMMY_DEFAULT = 100
    BASELINE = 100

    def __init__(
        self,
        num_arms: int,
        n_ens: int,
        mask_p: float,
        q_initialisation: float,
        s_initialisation: float,
        adaptation_modules: List,
        learning_rate: Union[str, float, Dict],
        temperature: Union[str, float, Dict],
        lr_per_arm: bool,
        use_direct: bool,
        scalar_log_spec: List[str],
        rng=None,
    ):
        """Class Constructor

        Args:
            num_arms: number of actions in the bandit
            n_ens: size of the ensemble of learners
            mask_p: probability of updating each ensemble member at each step
            q_initialisation: initialisation scale for return mean estimates
            s_initialisation: initialisation scale for return variance estimates
            learning_rate: None for adaptive learning rate, otherwise hard-coded value
            temperature: specify way of adapting temperature
            lr_per_arm: whether to have a separate adaptive lr per arm (or average)
            use_direct: whether to use direct variance estimation or standard GD
            scalar_log_spec: list of quantities to log.
            rng: rng state
        """
        if rng is None:
            rng = np.random.RandomState()

        self._rng = rng
        self._rng_key = jax.random.PRNGKey(self._rng.randint(1000000))

        self._num_arms = num_arms
        self._n_ens = n_ens
        self._mask_p = mask_p

        self._q_initialisation = q_initialisation
        self._qv_initialisation = s_initialisation

        # Total and per-arm step count
        self._total_steps = 0
        self._step = np.zeros(self._n_ens)
        self._step_arm = np.zeros((self._num_arms, self._n_ens))

        self._scalar_log_spec = scalar_log_spec

        self._lr_per_arm = lr_per_arm
        self._use_direct = use_direct
        self._arm_seen = np.zeros(self._num_arms, dtype=bool)

        self._setup_values()
        self._setup_optimizer()

        # lr_spec = {}
        # for lr_arg in learning_rate:
        #     lr_spec = {**lr_spec, **lr_arg}

        # temperature_spec = {}
        # for temperature_arg in temperature:
        #     temperature_spec = {**temperature_spec, **temperature_arg}

        self._adaptation_modules = {
            k: self._setup_adaptation_module(v) for k, v in adaptation_modules.items()
        }

        if isinstance(learning_rate, dict):
            self._learning_rate_module_id = None
            operation = learning_rate[constants.LEARNING_RATE_OPERATION]
            operands = learning_rate[constants.LEARNING_RATE_OPERANDS]
            if operation == constants.MULTIPLY:
                self._learning_rate_operation = lambda x: functools.reduce(
                    lambda x1, x2: x1 * x2,
                    [self._adaptation_modules[o](None) for o in operands],
                )
            elif operation == constants.DIVIDE:
                self._learning_rate_operation = lambda x: functools.reduce(
                    lambda x1, x2: x1 / x2,
                    [self.scalar_log().get(o) for o in operands],
                )
            elif operation == constants.RATIO:
                self._learning_rate_operation = lambda x: self.scalar_log()[
                    operands[0]
                ] / (
                    self.scalar_log()[operands[0]]
                    + np.sqrt(self.scalar_log()[operands[1]])
                )
            elif operation == constants.ORACLE:
                self._learning_rate_operation = lambda x: 0.1 + self.scalar_log()[
                    operands[0]
                ] / (self.scalar_log()[operands[0]] + self._oracle_aleatoric)
            elif operation == constants.FULL_ORACLE:
                self._learning_rate_operation = (
                    lambda x: 0.1
                    + self._oracle_epistemic()
                    / (self._oracle_epistemic() + self._oracle_aleatoric)
                )
            elif operation == constants.TANH_RATIO:
                operand_vals = []
                for op in operands:
                    if isinstance(op, float):
                        operand_vals.append(op)
                    else:
                        operand_vals.append(self.scalar_log()[op])
                self._learning_rate_operation = lambda x: np.tanh(
                    self.scalar_log()[operand_vals[0]]
                    / (
                        self.scalar_log()[operand_vals[0]]
                        + self.scalar_log()[operand_vals[1]]
                    )
                )
            elif operation == constants.RATIO_MULTIPLY:
                self._learning_rate_operation = (
                    lambda x: self.scalar_log()[operands[0]]
                    * self.scalar_log()[operands[1]]
                    / (self.scalar_log()[operands[1]] + self.scalar_log()[operands[2]])
                )
            elif operation == constants.RATIO_SELECT:
                self._learning_rate_operation = lambda x: self.scalar_log()[
                    operands[0]
                ] / (
                    self.scalar_log()[operands[0]]
                    + self.scalar_log()[f"{operands[1]}_{x}"]
                )
            elif operation == constants.LOG:
                self._learning_rate_operation = lambda x: np.log(
                    self._scalar_log()[operands[0]]
                )
        else:
            self._learning_rate_module_id = learning_rate
        if isinstance(temperature, dict):
            self._temperature_module_id = None
            operation = temperature[constants.TEMPERATURE_OPERATION]
            operands = temperature[constants.TEMPERATURE_OPERANDS]
            if operation == constants.MULTIPLY:
                self._temperature_operation = lambda x: functools.reduce(
                    lambda x1, x2: x1 * x2,
                    [self._adaptation_modules[o](None) for o in operands],
                )
            elif operation == constants.DIVIDE:
                self._temperature_operation = lambda x: functools.reduce(
                    lambda x1, x2: x1 / x2,
                    [self.scalar_log().get(o) for o in operands],
                )
            elif operation == constants.RATIO:
                self._temperature_operation = lambda x: self.scalar_log()[
                    operands[0]
                ] / (self.scalar_log()[operands[0]] + self.scalar_log()[operands[1]])
            elif operation == constants.LOG:
                self._temperature_operation = lambda x: np.log(
                    self._scalar_log()[operands[0]]
                )
        else:
            self._temperature_module_id = temperature

        self._dist = None
        self._oracle_aleatoric = np.inf

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, dist):
        self._dist = dist
        self._oracle_aleatoric = np.mean([d.std() for d in dist])

    @property
    def lr_module(self):
        if self._learning_rate_module_id is not None:
            return self._adaptation_modules[self._learning_rate_module_id]
        else:
            return self._learning_rate_operation

    @property
    def temperature_module(self):
        if self._temperature_module_id is not None:
            return self._adaptation_modules[self._temperature_module_id]
        else:
            return self._temperature_operation

    def _setup_adaptation_module(self, adaptation_module_spec: Dict):

        adaptation_spec = {}
        for adaptation_arg in adaptation_module_spec:
            adaptation_spec = {**adaptation_spec, **adaptation_arg}

        adaptation_type = adaptation_spec.pop(constants.TYPE)

        if adaptation_type == constants.CONSTANT:
            return constant_adaptation.ConstantAdaptation(**adaptation_spec)
        elif adaptation_type == constants.LIKELIHOOD_SHIFT:
            return likelihood_shift_adaptation.LikelihoodShiftAdaptation(
                **adaptation_spec
            )
        elif adaptation_type == constants.RELIABILITY_INDEX:
            return reliability_index_adaptation.ReliabilityIndexAdaptation(
                **adaptation_spec
            )
        elif adaptation_type == constants.CSUM_PLUS:
            return cumulative_sum_module.CumulativeSumPlusAdaptation(**adaptation_spec)
        elif adaptation_type == constants.CSUM_MINUS:
            return cumulative_sum_module.CumulativeSumMinusAdaptation(**adaptation_spec)
        elif adaptation_type == constants.DEUP:
            return deup_module.DEUP(**adaptation_spec)
        else:
            raise ValueError(f"Adaptation type {adaptation_type} not recognised.")

    def _setup_values(self):
        if self._q_initialisation > 0.0:
            self._q_initialisation = self._rng.normal(
                scale=self._q_initialisation, size=(self._num_arms, self._n_ens)
            )
        if self._qv_initialisation > 0.0:
            self._qv_initialisation = np.abs(
                self._rng.normal(
                    scale=self._qv_initialisation, size=(self._num_arms, self._n_ens)
                )
            )

        self._qvals = np.ones((self._num_arms, self._n_ens, 2))
        self._qvals[..., 0] *= self._q_initialisation
        self._qvals[..., 1] *= self._qv_initialisation
        self._qvals = jax.device_put(self._qvals)

    def _setup_optimizer(self):

        optimizer = optax.inject_hyperparams(optax.sgd)(
            learning_rate=self.DUMMY_DEFAULT,
        )

        self._opt_state = optimizer.init(self._qvals)

        def loss(params, arm, reward, rng_key):
            mask = jax.random.bernoulli(
                rng_key, p=self._mask_p, shape=(self._n_ens,)
            ).astype(jnp.float32)

            delta = reward - params[arm, :, 0]
            loss_val = 0.5 * jnp.sum(mask * (delta**2))

            if self._use_direct:
                # 'Direct' method, estimating based on observed squared errors
                # TD version: delta**2 + gamma**2 * Var' - Var
                delta_bar = jax.lax.stop_gradient(delta**2) - params[arm, :, 1]
            else:
                # 'Indirect' method, estimating the expected squared error,
                # then subtract squared expectation later.
                delta_bar = jax.lax.stop_gradient(reward**2) - params[arm, :, 1]

            loss_val += jnp.sum(mask * (delta_bar**2))

            return loss_val, delta  # * self.learning_rate(arm),

        self._loss = jax.jit(loss)

        def update(params, arm, reward, opt_state, rng_key):
            d_loss_d_params, delta = jax.grad(self._loss, has_aux=True)(
                params, arm, reward, rng_key
            )
            # if self._lr_noise_multiplier is not None:
            #     lr_noise = (
            #         self._lr_noise_multiplier
            #         * np.linspace(0.0, 1.0, self._n_ens)[
            #             np.asarray(jax.random.permutation(self._rng_key, self._n_ens))
            #         ]
            #     )
            #     d_loss_d_params = d_loss_d_params.at[arm].set(
            #         jnp.multiply(
            #             d_loss_d_params[arm],
            #             self._lr_noise_multiplier * np.vstack((lr_noise, lr_noise)).T,
            #         )
            #     )
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            params = optax.apply_updates(params, updates)
            return params, new_opt_state, delta

        self._update = jax.jit(update)

    def _setup_lr_module(self, lr_spec: Dict):

        if lr_spec[constants.TYPE] == constants.CONSTANT:
            lr_module = constant_adaptation.ConstantAdaptation(lr_spec[constants.VALUE])
        elif lr_spec[constants.TYPE] == constants.RELIABILITY_INDEX:
            lr_module = reliability_index_adaptation.ReliabilityIndexAdaptation(
                lr=lr_spec[constants.LEARNING_RATE], num_arms=self._num_arms
            )
        else:
            raise ValueError(
                f"learning rate module spec {lr_spec[constants.TYPE]} not recognised."
            )
        return lr_module

    def _setup_temperature_module(self, temperature_spec: Dict):
        if temperature_spec[constants.TYPE] == constants.CONSTANT:
            temperature_module = constant_adaptation.ConstantAdaptation(
                temperature_spec[constants.VALUE]
            )
        elif temperature_spec[constants.TYPE] == constants.RELIABILITY_INDEX:
            temperature_module = (
                reliability_index_adaptation.ReliabilityIndexAdaptation(
                    lr=temperature_spec[constants.LEARNING_RATE],
                    num_arms=self._num_arms,
                )
            )
        else:
            raise ValueError(
                f"temperature module spec {temperature_spec[constants.TYPE]} not recognised."
            )
        return temperature_module

    def predict_bandits(self):
        mean = self._qvals[..., 0].mean(axis=1)
        if self._use_direct:
            # exp since we learn log variances
            var = np.exp(self._qvals[..., 1]).mean(axis=1)
            return mean, var

        var = jnp.clip(
            self._qvals[..., 1] - self._qvals[..., 0].mean(axis=1, keepdims=True) ** 2,
            self.BASELINE,
            100.0,
        ).mean(axis=1)

        return mean, var

    def learning_rate(self, arm):
        return self.lr_module(arm)

    def temperature(self):
        return self.temperature_module(None)

    def scalar_log(self):
        log = {}

        if constants.MEAN_MEAN in self._scalar_log_spec:
            mean_mean = self._qvals[..., 0].mean(axis=1)
            log[constants.MEAN_MEAN] = mean_mean.mean()
        if constants.MEAN_VAR in self._scalar_log_spec:
            mean_var = self._qvals[..., 0].var(axis=1)
            log[constants.MEAN_VAR] = mean_var.mean()
        if self._use_direct:
            # exp since we learn log variances
            var_preds = np.exp(self._qvals[..., 1])
        else:
            var_preds = self._qvals[..., 1]
        if constants.VAR_MEAN in self._scalar_log_spec:
            var_mean = var_preds.mean(axis=1)
            log[constants.VAR_MEAN] = var_mean.mean()
            for arm in range(self._num_arms):
                log[f"{constants.VAR_MEAN}_{arm}"] = var_mean[arm]
        if constants.VAR_VAR in self._scalar_log_spec:
            var_var = var_preds.var(axis=1)
            log[constants.VAR_VAR] = var_var.mean()

        for k, v in self._adaptation_modules.items():
            if k in self._scalar_log_spec:
                log[k] = v(None)

        return log

    def policy(self):
        temperature = self.temperature()

        logits = np.exp(self.predict_bandits()[0])
        logits /= logits.sum()
        # logits = self._qvals[..., 0].mean(-1) - self._qvals[..., 0].mean(-1).max()
        return rlax.softmax(temperature).probs(logits)

        # Otherwise, acting with thompson sampling
        return np.eye(self._num_arms)[
            jax.device_get(self._qvals[..., 0].argmax(0))
        ].mean(0)

    def play(self):
        if not self._arm_seen.all():
            return jax.device_get(self._arm_seen.argmin())

        pi = self.policy()

        return jax.device_get(np.where(self._rng.random() <= pi.cumsum())[0][0])

    def update(self, arm, reward):
        self._arm_seen[arm] = True
        self._step += 1
        self._step_arm[arm] += 1
        self._total_steps += 1
        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)

        learning_rate = self.learning_rate(arm)
        self._opt_state.hyperparams["learning_rate"] = learning_rate

        self._qvals, self._opt_state, delta = self._update(
            self._qvals, arm, reward, self._opt_state, rng_key
        )

        for v in self._adaptation_modules.values():
            v.update(qvals=self._qvals, delta=delta, arm=arm, reward=reward)

        # self._lr_module.update()
        # self._temperature_module.update()

        # self._likelihood_memory_module_5.update(
        #     qvals=self._qvals, arm=arm, reward=reward
        # )
        # self._likelihood_memory_module_10.update(
        #     qvals=self._qvals, arm=arm, reward=reward
        # )
        # self._likelihood_memory_module_20.update(
        #     qvals=self._qvals, arm=arm, reward=reward
        # )
        # self._likelihood_memory_module_50.update(
        #     qvals=self._qvals, arm=arm, reward=reward
        # )
        # self._reliability_index_module.update(delta=delta, arm=arm)
