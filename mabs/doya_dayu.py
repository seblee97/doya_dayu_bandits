import numpy as np

from mabs import mab


class DoyaDaYu(mab.MAB):
    def __init__(
        self,
        n_arms,
        n_ens,
        mask_p=0.5,
        learning_rate=None,
        adapt_temperature=True,
        rng=None,
        lrate_per_arm=True,
        lr_noise_multiplier=2,
    ):

        self._n_ens = n_ens
        self._mask_p = mask_p
        self._learning_rate = learning_rate  # None if using adaptive learning rate

        # If we do not adapt learning rate, then we will just use the optimizer given
        # If we do not adapt temperature, then we will just do thompson sampling
        self._adapt_temperature = adapt_temperature
        self._lrate_per_arm = lrate_per_arm

        super().__init__(n_arms=n_arms, rng=rng)

        Q0 = 0.5
        S0 = 0.01

        if Q0 > 0:
            Q0 = rng.normal(scale=Q0, size=(n_arms, n_ens))
        if S0 > 0:
            S0 = np.abs(rng.normal(scale=S0, size=(n_arms, n_ens)))

        self._sqr_rewards = (
            np.ones((self._n_arms, self._n_ens)) * S0  # self._s_initialisation
        )

        # overwrite parent
        self._rewards = (
            np.ones((self._n_arms, self._n_ens)) * Q0
        )  # self._q_initialisation
        self._step = np.zeros(self._n_ens)
        self._step_arm = np.zeros((self._n_arms, self._n_ens))

        self._lr_noise_multiplier = lr_noise_multiplier

    def predict_bandits(self):
        mean = self._rewards.mean(-1)
        var = np.clip(self._sqr_rewards.mean(-1) - mean**2, 0.0, 100.0)
        return mean, var

    def get_epistemic(self):  # Unexpected uncertainty
        return np.std(self._rewards, axis=1) ** 2

    def get_aleatoric(self):  # Expected uncertainty
        return self.predict_bandits()[1]

    def learning_rate(self, arm):
        if self._learning_rate is None:
            lrate = self.get_epistemic() / (self.get_epistemic() + self.get_aleatoric())
            if self._lrate_per_arm:
                return lrate[arm]
            return lrate.mean()

        return self._learning_rate

    def policy(self):
        if self._adapt_temperature:
            temperature = self.get_epistemic()
            temperature = 0.5 * np.sqrt(temperature)
            # logits = self._rewards[np.arange(self._n_arms), self.rng.randint(self._n_ens, size=self._n_arms)]
            logits = self.predict_bandits()[0]
            logits -= logits.max()
            probs = np.ones_like(logits) / self._n_arms
            for _ in range(10):
                probs = np.exp(logits / np.dot(temperature, probs))
                probs /= probs.sum()

        # Otherwise, acting with thompson sampling
        return np.eye(self._n_arms)[self._rewards.argmax(0)].mean(0)

    def play(self):
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        pi = self.policy()

        import pdb

        pdb.set_trace()
        return np.where(np.random.random() <= pi.cumsum())[0][0]

    def _update(self, arm, reward):

        alpha = self.learning_rate(arm) * np.ones(self._n_ens)
        # alpha *= self.rng.random(self._n_ens)
        alpha *= (
            self._lr_noise_multiplier
            * np.linspace(0.0, 1.0, self._n_ens)[np.random.permutation(self._n_ens)]
        )
        masked = np.arange(self._n_ens)
        # masked = np.where(self._rng.random(self._n_ens) <= self._mask_p)[0]
        for ind in masked:
            self._step[ind] += 1
            self._step_arm[arm, ind] += 1
            self._rewards[arm, ind] += alpha[ind] * (reward - self._rewards[arm, ind])
            self._sqr_rewards[arm, ind] += alpha[ind] * (
                reward**2 - self._sqr_rewards[arm, ind]
            )
