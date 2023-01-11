import collections
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

import utils


class DoyaDaYu:
    ### TODO: Think about initialization, for a slow start, we
    # want means close to each other and scales near 1,
    # if we want a faster start, we want scales near zero, and means far apart.

    BASELINE = 0.00001
    DUMMY_DEFAULT = 0.1

    def __init__(
        self,
        n_arms: int,
        n_ens: int,
        mask_p: float,
        Q0: float,
        S0: float,
        learning_rate: Union[float, None],
        adapt_temperature: bool,
        lrate_per_arm: bool,
        lr_noise_multiplier: Union[float, int],
        use_direct: bool,
        aleatoric: str,
        rng=None,
    ):
        """Class Constructor

        Args:
            n_arms: number of actions in the bandit
            n_ens: size of the ensemble of learners
            mask_p: probability of updating each ensemble member at each step
            Q0: initialisation scale for return mean estimates
            S0: initialisation scale for return variance estimates
            learning_rate: None for adaptive learning rate, otherwise hard-coded value
            adapt_temperature: whether to adapt temperature or use Thompson
            lrate_per_arm: whether to have a separate adaptive lr per arm (or average)
            lr_noise_multiplier: size of randomisation for learning rates for each ensemble member
            use_direct: whether to use direct variance estimation or standard GD
            aleatoric: string specifying type of aleatoric uncertainty
                (variance, pairwise_kl_mean, pairwise_kl_max, information_radius)
            rng: rng state
        """

        if rng is None:
            rng = np.random.RandomState()

        self._rng = rng
        self._rng_key = jax.random.PRNGKey(self._rng.randint(1000000))

        self._n_arms = n_arms
        self._n_ens = n_ens
        self._mask_p = mask_p
        self._lr_noise_multiplier = lr_noise_multiplier
        self._use_direct = use_direct
        self._aleatoric = aleatoric

        self._q_initialisation = Q0
        self._qv_initialisation = S0

        # Total and per-arm step count
        self._total_steps = 0
        self._step = np.zeros(self._n_ens)
        self._step_arm = np.zeros((self._n_arms, self._n_ens))

        self._temperature_memory = 10
        self._lr_memory = 20
        self._likelihood_memory = collections.deque(
            self._temperature_memory * [self.BASELINE], self._temperature_memory
        )
        self._per_arm_likelihood_memory = {
            arm: collections.deque(
                int(2 * np.max([self._lr_memory, self._temperature_memory]))
                * [self.BASELINE],
                int(2 * np.max([self._lr_memory, self._temperature_memory])),
            )
            for arm in range(self._n_arms)
        }

        # If we do not adapt learning rate, then we will just use the optimizer given
        # If we do not adapt temperature, then we will just do thompson sampling
        self._adapt_temperature = adapt_temperature
        self._learning_rate = learning_rate
        self._lrate_per_arm = lrate_per_arm
        self._arm_seen = np.zeros(self._n_arms, dtype=bool)

        self._min_epistemic_uncertainty = np.inf

        self._setup_values()
        self._setup_optimizer()

    @property
    def min_epistemic_uncertainty(self):
        return self._min_epistemic_uncertainty

    @property
    def epistemic_uncertainty(self):
        return self.get_epistemic().mean()

    @property
    def aleatoric_uncertainty(self):
        return self.get_aleatoric().mean()

    def _setup_values(self):
        if self._q_initialisation > 0.0:
            self._q_initialisation = self._rng.normal(
                scale=self._q_initialisation, size=(self._n_arms, self._n_ens)
            )
        if self._qv_initialisation > 0.0:
            self._qv_initialisation = np.abs(
                self._rng.normal(
                    scale=self._qv_initialisation, size=(self._n_arms, self._n_ens)
                )
            )

        self._qvals = np.ones((self._n_arms, self._n_ens, 2))
        self._qvals[..., 0] *= self._q_initialisation
        self._qvals[..., 1] *= self._qv_initialisation
        self._qvals = jax.device_put(self._qvals)

    def _setup_optimizer(self):

        if self._learning_rate is None:
            optimizer = optax.inject_hyperparams(optax.sgd)(
                learning_rate=self.DUMMY_DEFAULT,
            )
        else:
            optimizer = optax.sgd(self._learning_rate)

        self._opt_state = optimizer.init(self._qvals)

        # # If we do not adapt learning rate, then we will just use the optimizer given
        # # If we do not adapt temperature, then we will just do thompson sampling
        # self.adapt_lrate = adapt_lrate
        # self.adapt_temperature = adapt_temperature
        # self.lrate_per_arm = lrate_per_arm

        # if Q0 > 0:
        #     Q0 = rng.normal(scale=Q0, size=(n_arms, n_ens))
        # if S0 > 0:
        #     S0 = np.abs(rng.normal(scale=S0, size=(n_arms, n_ens)))

        # self.qvals = np.ones((n_arms, n_ens, 2))
        # self.qvals[..., 0] *= Q0
        # self.qvals[..., 1] *= S0
        # # self.qvals[..., 1] = np.log(np.exp(np.abs(
        # #     rng.normal(scale=S0, size=(n_arms, n_ens)))) - 1)

        # self.arm_seen = np.zeros(n_arms, dtype=bool)

        # # Total and per-arm step count
        # self.step = 0
        # self.step_arm = np.zeros(self._n_arms)

        # # Optimization setup
        # self.qvals = jax.device_put(self.qvals)
        # self.optimizer = optimizer
        # self.opt_state = self.optimizer.init(self.qvals)
        # self._rng_key = jax.random.PRNGKey(rng.randint(1000000))

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

            return loss_val * self.learning_rate(arm)

        # self._loss = jax.jit(loss)
        self._loss = loss

        def update(params, arm, reward, opt_state, rng_key):
            d_loss_d_params = jax.grad(self._loss)(params, arm, reward, rng_key)
            if self._lr_noise_multiplier is not None:
                lr_noise = (
                    self._lr_noise_multiplier
                    * np.linspace(0.0, 1.0, self._n_ens)[
                        np.random.permutation(self._n_ens)
                    ]
                )
                d_loss_d_params = d_loss_d_params.at[arm].set(
                    jnp.multiply(
                        d_loss_d_params[arm],
                        self._lr_noise_multiplier * np.vstack((lr_noise, lr_noise)).T,
                    )
                )
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            params = optax.apply_updates(params, updates)
            return params, new_opt_state

        # self._update = jax.jit(update)
        self._update = update

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

    def get_epistemic(self):  # Unexpected uncertainty
        # if self._total_steps > 400 or self._total_steps == 0:
        #     import pdb

        #     pdb.set_trace()
        return (
            self._qvals[..., 0].std(axis=1) ** 2
        )  # + jax.nn.softplus(self.qvals[..., 1]).std(axis=1)**2
        # average pairwise KL

    def get_aleatoric(self):  # Expected uncertainty
        # return jax.nn.softplus(self.qvals[..., 1]).mean(axis=1)
        return self.predict_bandits()[1]
        # if self._use_direct:
        #     return jnp.abs(self._qvals[..., 1]).mean(axis=1)

        # return jnp.clip(
        #     self._qvals[..., 1] - self._qvals[..., 0].mean(axis=1, keepdims=True) ** 2,
        #     0.0,
        #     100.0,
        # ).mean(axis=1)

    def learning_rate(self, arm):
        if self._learning_rate is None:
            # lrate = self.get_epistemic() / (self.get_epistemic() + self.get_aleatoric())
            # lrate = np.clip(
            #     self.get_epistemic() / self.get_aleatoric()
            #     + self.get_aleatoric() / self.get_epistemic(),
            #     self.BASELINE,
            #     1,
            # )
            # lrate = np.clip(
            #     self.get_epistemic() / self.get_aleatoric()
            #     + self.get_aleatoric() / self.get_epistemic(),
            #     self.BASELINE,
            #     1,
            # )
            temperature = self.temperature(memory=self._lr_memory)

            # lrate = np.clip(
            lrate = 0.1 * (temperature / (self.get_aleatoric() + temperature))
            #     self.BASELINE,
            #     0.1,
            # )

            if any(np.isnan(lrate)):
                lrate = 0.1 * np.ones(self._n_arms)

            print("Learning RATE", lrate)
            if self._lrate_per_arm:
                # import pdb

                # pdb.set_trace()
                return lrate[arm]
            return lrate.mean()

        return self._learning_rate

    def temperature(self, memory=None):

        memory = memory or self._temperature_memory

        def _likelihood_shift(likelihood_deque):
            list_l = list(likelihood_deque)

            past = np.mean(list_l[memory : 2 * memory])
            pres = np.mean(list_l[:memory])
            if np.isnan(pres) or np.isnan(past):
                import pdb

                pdb.set_trace()
            return past / pres + pres / past - 2

        likelihood_shifts = [
            _likelihood_shift(self._per_arm_likelihood_memory[arm])
            for arm in range(self._n_arms)
        ]

        print("ls", likelihood_shifts)
        logits = np.exp(self.predict_bandits()[0])
        logits /= logits.sum()
        # return np.max(likelihood_shifts)
        return np.average(likelihood_shifts, weights=logits)
        temperature = np.average(self.get_epistemic(), weights=logits)
        # temperature = 0.5 * np.sqrt(temperature)
        if temperature < self._min_epistemic_uncertainty:
            # print(self._min_epistemic_uncertainty)
            self._min_epistemic_uncertainty = temperature - self.BASELINE

        return temperature

    def policy(self):
        if self._adapt_temperature:
            temperature = self.temperature()  # - self._min_epistemic_uncertainty
            # logits = self.predict_bandits()[0]
            # logits -= logits.max()

            logits = np.exp(self.predict_bandits()[0])
            logits /= logits.sum()
            # logits = self._qvals[..., 0].mean(-1) - self._qvals[..., 0].mean(-1).max()
            return rlax.softmax(temperature).probs(logits)

        # Otherwise, acting with thompson sampling
        return np.eye(self._n_arms)[jax.device_get(self._qvals[..., 0].argmax(0))].mean(
            0
        )

    def play(self):
        if not self._arm_seen.all():
            return jax.device_get(self._arm_seen.argmin())

        pi = self.policy()

        # import pdb

        # pdb.set_trace()

        return jax.device_get(np.where(np.random.random() <= pi.cumsum())[0][0])

        # if self._adapt_temperature:
        #     # Alternative:
        #     # epsilon = np.clip(self.get_epistemic(), 0., 1.)
        #     rng_key, self._rng_key = jax.random.split(self._rng_key, 2)
        #     temperature = self.get_epistemic()
        #     logits = self._qvals[..., 0].mean(-1) - self._qvals[..., 0].mean(-1).max()
        #     return jax.device_get(
        #         rlax.softmax(temperature).sample(rng_key, logits)
        #     ).ravel()[0]

        # ind = self.rng.choice(self._n_ens, size=self._n_arms)
        # return jax.device_get(
        #     self._qvals[np.arange(self._n_arms), ind, 0].argmax()
        # ).ravel()[0]

    def update(self, arm, reward):
        self._arm_seen[arm] = True
        self._step += 1
        self._step_arm[arm] += 1
        self._total_steps += 1
        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)

        if self._learning_rate is None:
            self._opt_state.hyperparams["learning_rate"] = self.learning_rate(arm)

        self._qvals, self._opt_state = self._update(
            self._qvals, arm, reward, self._opt_state, rng_key
        )

        if self._use_direct:
            likelihood = np.mean(
                [
                    utils.gaussian_likelihood(mean, np.sqrt(np.exp(logvar)), reward)
                    for (mean, logvar) in self._qvals[arm]
                ]
            )
        else:
            likelihood = np.mean(
                [
                    utils.gaussian_likelihood(
                        mean,
                        np.sqrt(np.clip(sqr_rew - mean**2, self.BASELINE, 100.0)),
                        reward,
                    )
                    for (mean, sqr_rew) in self._qvals[arm]
                ]
            )
        self._likelihood_memory.appendleft(likelihood)
        self._per_arm_likelihood_memory[arm.item()].appendleft(likelihood)
