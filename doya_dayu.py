import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from random_bandit import RandomMAB


class DoyaDaYu(RandomMAB):
    ### TODO: Think about initialization, for a slow start, we
    # want means close to each other and scales near 1,
    # if we want a faster start, we want scales near zero, and means far apart.

    def __init__(
        self,
        n_arms,
        n_ens,
        optimizer,
        mask_p=0.5,
        Q0=0.01,
        S0=0.001,
        adapt_lrate=True,
        adapt_temperature=True,
        rng=None,
        lrate_per_arm=True,
        use_direct=True,
    ):

        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng
        self._n_arms = n_arms
        self._n_ens = n_ens
        self._mask_p = mask_p
        self.use_direct = use_direct

        # If we do not adapt learning rate, then we will just use the optimizer given
        # If we do not adapt temperature, then we will just do thompson sampling
        self.adapt_lrate = adapt_lrate
        self.adapt_temperature = adapt_temperature
        self.lrate_per_arm = lrate_per_arm

        if Q0 > 0:
            Q0 = rng.normal(scale=Q0, size=(n_arms, n_ens))
        if S0 > 0:
            S0 = np.abs(rng.normal(scale=S0, size=(n_arms, n_ens)))

        self.qvals = np.ones((n_arms, n_ens, 2))
        self.qvals[..., 0] *= Q0
        self.qvals[..., 1] *= S0
        # self.qvals[..., 1] = np.log(np.exp(np.abs(
        #     rng.normal(scale=S0, size=(n_arms, n_ens)))) - 1)

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

            delta = reward - params[arm, :, 0]
            loss_val = 0.5 * jnp.sum(mask * (delta**2))

            if use_direct:
                # 'Direct' method, estimating based on observed squared errors
                # TD version: delta**2 + gamma**2 * Var' - Var
                delta_bar = jax.lax.stop_gradient(delta**2) - params[arm, :, 1]
            else:
                # 'Indirect' method, estimating the expected squared error,
                # then subtract squared expectation later.
                delta_bar = jax.lax.stop_gradient(reward**2) - params[arm, :, 1]

            loss_val += jnp.sum(mask * (delta_bar**2))

            return loss_val * self.learning_rate(arm)

        self._loss = jax.jit(loss)

        def update(params, arm, reward, opt_state, rng_key):
            d_loss_d_params = jax.grad(self._loss)(params, arm, reward, rng_key)
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            params = optax.apply_updates(params, updates)
            return params, new_opt_state

        self._update = jax.jit(update)

    def predict_bandits(self):
        mean = self.qvals[..., 0].mean(axis=1)
        if self.use_direct:
            var = self.qvals[..., 1].mean(axis=1)
            return mean, var

        var = jnp.clip(
            self.qvals[..., 1] - self.qvals[..., 0].mean(axis=1, keepdims=True) ** 2,
            0.0,
            100.0,
        ).mean(axis=1)
        return mean, var

    def get_epistemic(self):  # Unexpected uncertainty
        return (
            self.qvals[..., 0].std(axis=1) ** 2
        )  # + jax.nn.softplus(self.qvals[..., 1]).std(axis=1)**2
        # average pairwise KL

    def get_aleatoric(self):  # Expected uncertainty
        # return jax.nn.softplus(self.qvals[..., 1]).mean(axis=1)
        if self.use_direct:
            return jnp.abs(self.qvals[..., 1]).mean(axis=1)

        return jnp.clip(
            self.qvals[..., 1] - self.qvals[..., 0].mean(axis=1, keepdims=True) ** 2,
            0.0,
            100.0,
        ).mean(axis=1)

    def learning_rate(self, arm):
        if self.adapt_lrate:
            lrate = self.get_epistemic() / (self.get_epistemic() + self.get_aleatoric())
            if self.lrate_per_arm:
                return lrate[arm]
            return lrate.mean()

        return 1.0

    def policy(self):
        if self.adapt_temperature:
            temperature = self.get_epistemic()
            logits = self.qvals[..., 0].mean(-1) - self.qvals[..., 0].mean(-1).max()
            return rlax.softmax(temperature).probs(logits)

        # Otherwise, acting with thompson sampling
        return np.eye(self._n_arms)[jax.device_get(self.qvals[..., 0].argmax(0))].mean(
            0
        )

    def play(self):
        if not self.arm_seen.all():
            return self.arm_seen.argmin()

        if self.adapt_temperature:
            # Alternative:
            # epsilon = np.clip(self.get_epistemic(), 0., 1.)
            rng_key, self._rng_key = jax.random.split(self._rng_key, 2)
            temperature = self.get_epistemic()
            logits = self.qvals[..., 0].mean(-1) - self.qvals[..., 0].mean(-1).max()
            return jax.device_get(
                rlax.softmax(temperature).sample(rng_key, logits)
            ).ravel()[0]

        ind = self.rng.choice(self._n_ens, size=self._n_arms)
        return jax.device_get(
            self.qvals[np.arange(self._n_arms), ind, 0].argmax()
        ).ravel()[0]

    def update(self, arm, reward):
        self.arm_seen[arm] = True
        self.step += 1
        self.step_arm[arm] += 1
        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)

        self.qvals, self.opt_state = self._update(
            self.qvals, arm, reward, self.opt_state, rng_key
        )
