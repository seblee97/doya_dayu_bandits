from typing import Union

import jax
import numpy as np
import rlax

import utils
from random_bandit import RandomMAB


class DoyaDaYu:

    BASELINE = 0.0001

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

        # If we do not adapt learning rate, then we will just use the optimizer given
        # If we do not adapt temperature, then we will just do thompson sampling
        self._adapt_temperature = adapt_temperature
        self._learning_rate = learning_rate
        self._lrate_per_arm = lrate_per_arm
        self._arm_seen = np.zeros(self._n_arms, dtype=bool)

        self._min_epistemic_uncertainty = np.inf

        if Q0 > 0:
            Q0 = rng.normal(scale=Q0, size=(n_arms, self._n_ens))
        if S0 > 0:
            S0 = np.abs(rng.normal(scale=S0, size=(n_arms, self._n_ens)))

        self._rewards = np.ones((self._n_arms, self._n_ens)) * Q0
        self._sqr_rewards = np.ones((self._n_arms, self._n_ens)) * S0

        # Total and per-arm step count
        self._step = np.zeros(self._n_ens)
        self._step_arm = np.zeros((self._n_arms, self._n_ens))

    @property
    def min_epistemic_uncertainty(self):
        return self._min_epistemic_uncertainty

    @property
    def epistemic_uncertainty(self):
        return self.get_epistemic().mean()

    @property
    def aleatoric_uncertainty(self):
        return self.get_aleatoric().mean()

    def predict_bandits(self):
        mean = self._rewards.mean(-1)
        var = np.clip(self._sqr_rewards.mean(-1) - mean**2, 0.0, 100.0)
        return mean, var

    def get_epistemic(self):  # Unexpected uncertainty
        return np.std(self._rewards, axis=1) ** 2

    def get_aleatoric(self):  # Expected uncertainty
        if self._aleatoric == "variance":
            return self.predict_bandits()[1]
        elif self._aleatoric == "pairwise_kl_mean":
            means, variances = self.predict_bandits()
            return utils.compute_pairwise_kl(means=means, stds=np.sqrt(variances))[0]
        elif self._aleatoric == "pairwise_kl_max":
            means, variances = self.predict_bandits()
            return utils.compute_pairwise_kl(means=means, stds=np.sqrt(variances))[1]
        elif self._aleatoric == "information_radius":
            means, variances = self.predict_bandits()
            return utils.compute_information_radius(
                means=self._rewards, stds=self._sqr_rewards
            )

    def learning_rate(self, arm):
        if self._learning_rate is None:
            lrate = self.get_epistemic() / (self.get_epistemic() + self.get_aleatoric())
            if self._lrate_per_arm:
                return lrate[arm]
            return lrate.mean()

        return self._learning_rate

    def temperature(self):
        # temperature = np.mean(self.get_epistemic())
        logits = np.exp(self.predict_bandits()[0])
        logits /= logits.sum()
        temperature = np.average(self.get_epistemic(), weights=logits)
        # temperature = 0.5 * np.sqrt(temperature)
        if temperature < self._min_epistemic_uncertainty:
            # print(self._min_epistemic_uncertainty)
            self._min_epistemic_uncertainty = temperature - self.BASELINE
        return temperature

    def policy(self):
        if self._adapt_temperature:
            temperature = self.temperature() - self._min_epistemic_uncertainty
            # logits = self.rewards[np.arange(self._n_arms), self.rng.randint(self._n_ens, size=self._n_arms)]
            logits = self.predict_bandits()[0]
            logits -= logits.max()
            # probs = np.ones_like(logits) / self._n_arms
            # import pdb

            # pdb.set_trace()
            # for _ in range(10):
            #     probs = np.exp(logits / np.dot(temperature, probs))
            #     probs /= probs.sum()
            # import pdb

            # pdb.set_trace()
            probs = np.exp(logits / temperature)
            probs /= probs.sum()
            return probs

        # Otherwise, acting with thompson sampling
        return np.eye(self._n_arms)[self._rewards.argmax(0)].mean(0)

    def play(self):
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        pi = self.policy()
        # temperature = self.get_epistemic()
        # temperature = 0.5 * np.sqrt(temperature)

        # logits = self.predict_bandits()[0]
        # logits -= logits.max()

        # return jax.device_get(
        #     rlax.softmax(self.temperature()).sample(self._rng_key, logits)
        # ).ravel()[0]

        # import pdb

        # pdb.set_trace()
        # ucb_values = np.zeros(self._n_arms)
        # for arm in range(self._n_arms):
        #     if self._step_arm[arm, 0] > 0:
        #         ucb_values[arm] = np.sqrt(
        #             1 * np.log(self._step[0]) / self._step_arm[arm][0]
        #         )
        # action = ((self._rewards[:, 0] / self._step_arm[:, 0]) + ucb_values).argmax()
        # return action
        return np.where(np.random.random() <= pi.cumsum())[0][0]

    def update(self, arm, reward):
        self._arm_seen[arm] = True

        alpha = self.learning_rate(arm) * np.ones(self._n_ens)
        alpha *= (
            self._lr_noise_multiplier
            * np.linspace(0.0, 1.0, self._n_ens)[np.random.permutation(self._n_ens)]
        )
        masked = np.where(self._rng.random(self._n_ens) <= self._mask_p)[0]
        for ind in masked:
            self._step[ind] += 1
            self._step_arm[arm, ind] += 1
            delta = reward - self._rewards[arm, ind]
            self._rewards[arm, ind] += alpha[ind] * delta

            if self._use_direct:
                self._sqr_rewards[arm, ind] += alpha[ind] * (
                    delta**2 - self._sqr_rewards[arm, ind]
                )
            else:
                self._sqr_rewards[arm, ind] += alpha[ind] * (
                    reward**2 - self._sqr_rewards[arm, ind]
                )
