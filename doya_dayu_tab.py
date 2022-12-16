import numpy as np

from random_bandit import RandomMAB


class DoyaDaYuTabular(RandomMAB):
    def __init__(
        self,
        n_arms,
        n_ens,
        mask_p=0.5,
        Q0=0.01,
        S0=0.001,
        adapt_lrate=True,
        adapt_temperature=True,
        rng=None,
        lrate_per_arm=True,
        alpha=0.1,
    ):
        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng
        self._n_arms = n_arms
        self._n_ens = n_ens
        self._mask_p = mask_p

        # If we do not adapt learning rate, then we will just use the optimizer given
        # If we do not adapt temperature, then we will just do thompson sampling
        self.adapt_lrate = adapt_lrate
        self.adapt_temperature = adapt_temperature
        self.lrate_per_arm = lrate_per_arm
        self.arm_seen = np.zeros(n_arms, dtype=bool)

        if Q0 > 0:
            Q0 = rng.normal(scale=Q0, size=(n_arms, n_ens))
        if S0 > 0:
            S0 = np.abs(rng.normal(scale=S0, size=(n_arms, n_ens)))

        self.rewards = np.ones((self._n_arms, n_ens)) * Q0
        self.sqr_rewards = np.ones((self._n_arms, n_ens)) * S0

        # Total and per-arm step count
        self.step = np.zeros(n_ens)
        self.step_arm = np.zeros((self._n_arms, n_ens))
        self._alpha = alpha
        # self.gamma = gamma

    def predict_bandits(self):
        mean = self.rewards.mean(-1)
        var = np.clip(self.sqr_rewards.mean(-1) - mean**2, 0.0, 100.0)
        return mean, var

    def get_epistemic(self):  # Unexpected uncertainty
        return np.std(self.rewards, axis=1) ** 2

    def get_aleatoric(self):  # Expected uncertainty
        return self.predict_bandits()[1]

    def learning_rate(self, arm):
        if self.adapt_lrate:
            lrate = self.get_epistemic() / (self.get_epistemic() + self.get_aleatoric())
            if self.lrate_per_arm:
                return lrate[arm]
            return lrate.mean()

        return self._alpha

    def policy(self):
        if self.adapt_temperature:
            temperature = self.get_epistemic()
            temperature = 0.5 * np.sqrt(temperature)
            # logits = self.rewards[np.arange(self._n_arms), self.rng.randint(self._n_ens, size=self._n_arms)]
            logits = self.predict_bandits()[0]
            logits -= logits.max()
            probs = np.ones_like(logits) / self._n_arms
            for _ in range(10):
                probs = np.exp(logits / np.dot(temperature, probs))
                probs /= probs.sum()

        # Otherwise, acting with thompson sampling
        return np.eye(self._n_arms)[self.rewards.argmax(0)].mean(0)

    def play(self):
        if not self.arm_seen.all():
            return self.arm_seen.argmin()

        pi = self.policy()
        return np.where(np.random.random() <= pi.cumsum())[0][0]

    def update(self, arm, reward):
        self.arm_seen[arm] = True

        alpha = self.learning_rate(arm) * np.ones(self._n_ens)
        # alpha *= self.rng.random(self._n_ens)
        alpha *= (
            2 * np.linspace(0.0, 1.0, self._n_ens)[np.random.permutation(self._n_ens)]
        )
        # masked = np.arange(self._n_ens)
        masked = np.where(self.rng.random(self._n_ens) <= self._mask_p)[0]
        for ind in masked:
            self.step[ind] += 1
            self.step_arm[arm, ind] += 1
            self.rewards[arm, ind] += alpha[ind] * (reward - self.rewards[arm, ind])
            self.sqr_rewards[arm, ind] += alpha[ind] * (
                reward**2 - self.sqr_rewards[arm, ind]
            )
