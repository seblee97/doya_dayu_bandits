import numpy as np
import scipy.stats
import abc
import os
import time
import datetime
import multiprocessing as mp
import jax
import jax.numpy as jnp
import optax
import rlax
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from typing import List


NUM_SEEDS = 1
NUM_EPISODES = 20
EPISODE_LENGTH = 5000
NUM_ARMS = 2
MEAN_RANGE = [-2, 2]
SCALE_RANGE = [0.01, 1]
CHANGE_PROBABILITY = 0.75

N_QUANTILES = 31

raw_datetime = datetime.datetime.fromtimestamp(time.time())
exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")

RESULTS_PATH = os.path.join("results", exp_timestamp)
os.mkdir(RESULTS_PATH)


def _setup_bandit(
    num_arms,
    mean_upper,
    mean_lower,
    scale_upper,
    scale_lower,
    num_seeds,
    num_episodes,
    change_probability,
    bernoulli=False,
):
    if bernoulli:
        dist_hist = np.zeros((num_seeds, num_episodes, num_arms))
    else:
        dist_hist = np.zeros((num_seeds, num_episodes, num_arms, 2))

    def _sample_mean(lower, upper):
        return lower + np.random.random() * (upper - lower)

    def _sample_scale(lower, upper):
        return lower + np.random.random() * (upper - lower)

    def _sample_probability():
        return np.random.random()

    dists = [[] for _ in range(num_seeds)]
    best_arms = [[] for _ in range(num_seeds)]

    random_samples = np.random.random(size=(num_seeds, num_episodes))
    changes = [
        [episode == 0 or r < change_probability for episode, r in enumerate(ran)]
        for ran in random_samples
    ]

    for seed in range(num_seeds):
        for episode in range(num_episodes):
            change = changes[seed][episode]
            if not change:
                seed_ep_dists = dists[seed][episode - 1]
                best_arm = best_arms[seed][episode - 1]
                dist_hist[seed, episode] = dist_hist[seed, episode - 1]
            else:
                if bernoulli:
                    probs = np.array([(_sample_probability()) for _ in range(num_arms)])
                    dist_hist[seed, episode] = probs
                    seed_ep_dists = [
                        scipy.stats.bernoulli(probs[a]) for a in range(num_arms)
                    ]
                    best_arm = np.array([d.mean() for d in seed_ep_dists]).argmax()
                else:
                    mean_vars = np.array(
                        [
                            (
                                _sample_mean(mean_lower, mean_upper),
                                _sample_scale(scale_lower, scale_upper),
                            )
                            for _ in range(num_arms)
                        ]
                    )
                    dist_hist[seed, episode] = mean_vars
                    seed_ep_dists = [
                        scipy.stats.norm(mean_vars[a, 0], mean_vars[a, 1])
                        for a in range(num_arms)
                    ]
                    best_arm = np.array([d.mean() for d in seed_ep_dists]).argmax()

            dists[seed].append(seed_ep_dists)
            best_arms[seed].append(best_arm)

    return dists, best_arms


class TDMAB(abc.ABC):
    def __init__(self, num_arms: int, rng):
        if rng is None:
            rng = np.random.RandomState()

        self._rng = rng
        self._rng_key = jax.random.PRNGKey(self._rng.randint(1000000))

        self._num_arms = num_arms

        # Total and per-arm step count
        self._step = 0
        self._step_arm = np.zeros(self._num_arms)
        self._arm_seen = np.zeros(self._num_arms, dtype=bool)

    @abc.abstractmethod
    def play(self):
        pass

    @abc.abstractmethod
    def update(self, arm, reward):
        pass

    @abc.abstractmethod
    def predict_bandits(self):
        pass

    @abc.abstractmethod
    def policy(self):
        pass

    @property
    def min_epistemic_uncertainty(self):
        return None

    @property
    def epistemic_uncertainty(self):
        return None

    @property
    def aleatoric_uncertainty(self):
        return None


class DiscountedUCB(TDMAB):
    def __init__(
        self,
        num_arms: int,
        rho: float,
        gamma: float,
        learning_rate: float,
        temperature: float,
        q_initialisation: float,
        scalar_log_spec: List[str],
        rng=None,
    ):
        self._rho = rho
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._temperature = temperature
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

        # self._update = update
        self._update = jax.jit(update)

    def predict_bandits(self):
        return self._qvals, np.ones(
            self._num_arms
        )  # np.log(self._step) / (self._step_arm + 1.0)

    def policy(self, ep):
        return np.eye(self._num_arms)[self.play(ep)]

    def play(self):
        # ensure all arms are played at least once initially.
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        rng_key, self._rng_key = jax.random.split(self._rng_key, 2)

        ucb_values = np.zeros(self._num_arms)
        for arm in range(self._num_arms):
            if self._step_arm[arm] > 0:
                ucb_values[arm] = np.sqrt(
                    self._rho * np.log(self._step) / self._step_arm[arm]
                )

        # return jax.device_get(
        #     rlax.softmax(self._temperature).sample(rng_key, self._qvals + ucb_values)
        # ).ravel()[0]

        action = (jax.device_get(self._qvals) + ucb_values).argmax()
        return action

    def learning_rate(self, action: int):
        return self._learning_rate

    def temperature(self):
        return None

    def update(self, arm, reward):
        # print(self._epsilon)
        self._arm_seen[arm] = True
        self._step *= self._gamma
        self._step_arm[arm] *= self._gamma
        self._step += 1
        self._step_arm[arm] += 1
        self._qvals, self._opt_state = self._update(
            self._qvals, arm, reward, self._opt_state
        )

    def scalar_log(self):
        log = {}

        # if constants.MEAN_EST in self._scalar_log_spec:
        #     mean = self._qvals.mean()
        #     mean = self._qvals.mean()
        #     log[constants.MEAN_EST] = mean
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


class QR(TDMAB):
    def __init__(
        self,
        num_arms: int,
        n_quantiles: int,
        learning_rate,
        adapt_lr,
        temperature,
        adapt_temp,
        ucb,
        gamma,
        rho,
        init_range,
        scalar_log_spec: List[str],
        true_dists,
        rng=None,
    ):
        self._learning_rate = learning_rate
        self._adapt_lr = adapt_lr
        self._temperature = temperature
        self._adapt_temp = adapt_temp
        self._ucb = ucb
        self._gamma = gamma
        self._rho = rho
        self.dtype = torch.float64

        self._scalar_log_spec = scalar_log_spec
        self._true_dists = true_dists

        self._n_quantiles = n_quantiles
        self._init_range = init_range

        super().__init__(num_arms=num_arms, rng=rng)

        self._init_quantiles()
        self._init_classifiers()
        # self._setup_optimizer()

    def _init_quantiles(self):
        self._q_distr = []
        self._parameters = []
        for a in range(self._num_arms):
            self._q_distr.append(
                QRTD(n_quantiles=self._n_quantiles, init_range=self._init_range)
            )
            self._parameters.append(self._q_distr[a].mean())
        self._taus = np.linspace(0, 1, num=self._n_quantiles + 1, endpoint=True)
        self._tau_hat = np.linspace(
            np.diff(self._taus)[0] / 2,
            1 - np.diff(self._taus)[0] / 2,
            num=self._n_quantiles,
        )

    def _init_classifiers(self):
        self._ineq_estimator = {}
        for a in range(self._num_arms):
            self._ineq_estimator[a] = InequalityEstimator(n_quantiles=self._n_quantiles)

    def predict_bandits(self):
        return [dist.mean() for dist in self._q_distr], [
            dist.var() for dist in self._q_distr
        ]

    def policy(self, ep):
        return np.eye(self._num_arms)[self.play(ep)]

    def play(self, ep):
        # ensure all arms are played at least once initially.
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        values = np.array([dist.mean() for dist in self._q_distr])

        if self._ucb:
            ucb_values = np.zeros(self._num_arms)
            for arm in range(self._num_arms):
                if self._step_arm[arm] > 0:
                    ucb_values[arm] = np.sqrt(
                        self._rho * np.log(self._step) / self._step_arm[arm]
                    )

            # return jax.device_get(
            #     rlax.softmax(self._temperature).sample(rng_key, self._qvals + ucb_values)
            # ).ravel()[0]

            action = (values + ucb_values).argmax()
        else:
            if self._adapt_temp is None:
                pass
            elif self._adapt_temp["type"] == "oracle_epistemic":
                factor = self._adapt_temp.get("factor", 1)
                self._temperature = factor / (
                    np.mean(
                        [
                            (self._q_distr[a].mean() - self._true_dists[ep][a].mean())
                            ** 2
                            for a in range(self._num_arms)
                        ]
                    )
                    + 0.0001
                )
            elif self._adapt_temp["type"] == "min_epistemic":
                factor = self._adapt_temp.get("factor", 1)
                self._temperature = factor / np.min(
                    [self.epistemic_uncertainty(arm) for arm in range(self._num_arms)]
                )
            elif self._adapt_temp["type"] == "max_epistemic":
                factor = self._adapt_temp.get("factor", 1)
                self._temperature = factor / np.max(
                    [self.epistemic_uncertainty(arm) for arm in range(self._num_arms)]
                )
            elif self._adapt_temp["type"] == "epistemic":
                factor = self._adapt_temp.get("factor", 1)
                self._temperature = factor / np.mean(
                    [self.epistemic_uncertainty(arm) for arm in range(self._num_arms)]
                )
            elif self._adapt_temp["type"] == "epistemic_ratio":
                factor = self._adapt_temp.get("factor", 1)
                epistemic = np.mean(
                    [self.epistemic_uncertainty(arm) for arm in range(self._num_arms)]
                )
                aleatoric = np.mean(
                    [self._q_distr[a].var() for a in range(self._num_arms)]
                )
                self._temperature = factor / (epistemic / (epistemic + aleatoric))
            elif self._adapt_temp["type"] == "epistemic_per_arm_bonus":
                factor = self._adapt_temp.get("factor", 1)
                bonus_values = np.zeros(self._num_arms)
                for arm in range(self._num_arms):
                    if self._step_arm[arm] > 0:
                        bonus_values[arm] = self.epistemic_uncertainty(arm)
                action = (values + factor * bonus_values).argmax()
                return action
            # logsumexp
            max_val = max(values)
            scaled_vals = values - max_val
            logits = np.exp(
                self._temperature
                * (values - (max_val + np.log(np.exp(scaled_vals).sum())))
                + 0.0001
            )

            if any(logits == 0):
                action = np.argmax(values)
            elif any(np.isnan(logits)):
                action = np.argmax(values)
            else:
                softmax_values = logits / np.sum(logits)

                if any(np.isnan(softmax_values)):
                    import pdb

                    pdb.set_trace()

                action = np.random.choice(range(self._num_arms), p=softmax_values)

        return action

    def epistemic_uncertainty(self, arm):
        # using calibrated quantiles
        # reward = torch.tensor(reward, dtype=self.dtype, requires_grad=False)[None, None]

        #         if self._step > 10000:
        #             import pdb; pdb.set_trace()

        Fnu = self._ineq_estimator[arm].forward().clone().detach().numpy().flatten()
        qr_grad = self._q_distr[arm].tau_hat - Fnu
        # really propto, scaling possibly needed.
        epistemic = np.sqrt(np.mean(qr_grad**2))

        return epistemic

    def update(self, arm, reward, ep):
        self._arm_seen[arm] = True
        self._step *= self._gamma
        self._step_arm[arm] *= self._gamma
        self._step += 1
        self._step_arm[arm] += 1

        epistemic = self.epistemic_uncertainty(arm)
        aleatoric = self._q_distr[arm].var()

        oracle_epistemic = (
            self._q_distr[arm].mean() - self._true_dists[ep][arm].mean()
        ) ** 2
        oracle_aleatoric = self._true_dists[ep][arm].var()

        all_epistemics = [self.epistemic_uncertainty(a) for a in range(self._num_arms)]
        min_epistemics = np.min(all_epistemics)
        argmin_epistemics = np.argmin(all_epistemics)
        max_epistemics = np.max(all_epistemics)
        argmax_epistemics = np.argmax(all_epistemics)
        mean_epistemic = np.mean(all_epistemics)

        if self._adapt_lr is None:
            pass
        elif self._adapt_lr["type"] == "oracle_epistemic_ratio":
            factor = self._adapt_lr.get("factor", 1)
            self._learning_rate = (
                factor * oracle_epistemic / (oracle_epistemic + oracle_aleatoric)
            )
        elif self._adapt_lr["type"] == "oracle_epistemic_ratio_2":
            factor = self._adapt_lr.get("factor", 1)
            self._learning_rate = (
                factor * oracle_epistemic**2 / (oracle_epistemic + oracle_aleatoric)
            )
        elif self._adapt_lr["type"] == "ratio":
            factor = self._adapt_lr.get("factor", 1)
            self._learning_rate = factor * epistemic / (epistemic + aleatoric)
        elif self._adapt_lr["type"] == "ratio_2":
            factor = self._adapt_lr.get("factor", 1)
            self._learning_rate = factor * epistemic**2 / (epistemic + aleatoric)

        self._q_distr[arm].update(reward, lr=self._learning_rate)
        quantiles = torch.tensor(
            self._q_distr[arm].thetas, dtype=self.dtype, requires_grad=False
        )[:, None]

        reward = torch.tensor(reward, dtype=self.dtype, requires_grad=False)[None, None]

        if self._adapt_lr is None:
            ineq_lr = self._learning_rate
        elif self._adapt_lr.get("ineq_lr") is None:
            ineq_lr = self._learning_rate
        elif self._adapt_lr.get("ineq_lr") == "oracle_epistemic_ratio":
            factor = self._adapt_lr.get("ieq_factor", 1)
            ineq_lr = factor * oracle_epistemic / (oracle_epistemic + oracle_aleatoric)
        elif self._adapt_lr.get("ineq_lr") == "oracle_epistemic_ratio_2":
            factor = self._adapt_lr.get("ieq_factor", 1)
            ineq_lr = (
                factor * oracle_epistemic**2 / (oracle_epistemic + oracle_aleatoric)
            )
        elif self._adapt_lr.get("ineq_lr") == "ratio":
            factor = self._adapt_lr.get("ieq_factor", 1)
            ineq_lr = factor * epistemic / (epistemic + aleatoric)
        elif self._adapt_lr.get("ineq_lr") == "ratio_2":
            factor = self._adapt_lr.get("ieq_factor", 1)
            ineq_lr = factor * epistemic**2 / (epistemic + aleatoric)
        loss = self._ineq_estimator[arm].update(reward, quantiles, lr=ineq_lr)

        return (
            epistemic,
            aleatoric,
            oracle_epistemic,
            oracle_aleatoric,
            self._learning_rate,
            self._temperature,
            min_epistemics,
            argmin_epistemics,
            max_epistemics,
            argmax_epistemics,
            mean_epistemic,
            loss,
        )

    def scalar_log(self):
        log = {}

        # if constants.MEAN_EST in self._scalar_log_spec:
        #     mean = self._qvals.mean()
        #     mean = self._qvals.mean()
        #     log[constants.MEAN_EST] = mean
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


class QRTD(object):
    def __init__(self, n_quantiles=21, init_range=(-1, 1)):
        self.n_quantiles = n_quantiles
        self.init_range = init_range
        self.taus = np.linspace(0, 1, num=self.n_quantiles + 1, endpoint=True)
        self.tau_hat = np.linspace(
            np.diff(self.taus)[0] / 2,
            1 - np.diff(self.taus)[0] / 2,
            num=self.n_quantiles,
        )
        self.thetas = np.linspace(
            self.init_range[0], self.init_range[1], num=self.n_quantiles
        )
        self.thetas = np.sort(self.thetas)

    def update(self, target, lr):
        # import pdb; pdb.set_trace()
        delta_sign = target < self.thetas
        self.thetas = self.thetas + lr * np.clip(
            (self.tau_hat - delta_sign), a_min=-1, a_max=1
        )

    def mean(self):
        return np.mean(self.thetas)

    def var(self):
        return np.var(self.thetas)

    def pdf(self):
        pdf = (
            (1 / (self.n_quantiles - 1))
            * np.ones(self.n_quantiles - 1)
            / np.diff(self.thetas)
        )
        x_quantiles = self.thetas[:-1] + np.diff(self.thetas) / 2
        return x_quantiles, pdf


class InequalityEstimator(object):
    DEFAULT_LR = 1

    def __init__(self, n_quantiles):
        self.dtype = torch.float64
        self.n_quantiles = n_quantiles
        self.weights = torch.tensor(
            np.random.normal(size=(self.n_quantiles, 1)),
            dtype=self.dtype,
            requires_grad=True,
        )
        self.offsets = torch.tensor(
            np.random.normal(size=(self.n_quantiles, 1)),
            dtype=self.dtype,
            requires_grad=True,
        )
        self.optimizer = torch.optim.Adam(
            [self.weights, self.offsets], lr=InequalityEstimator.DEFAULT_LR
        )

    def forward(self):
        # Make sample_val.shape = (1, 1), and (batch, 1, 1) for broadcasting
        # l = self.weights @ sample_val + self.offsets
        l = self.weights
        return torch.sigmoid(l)

    def loss(self, sample_val, quantiles):
        deltas = sample_val <= quantiles
        deltas = deltas.type(self.dtype)
        p_tau = self.forward()
        loss = deltas * torch.log(p_tau + 1e-10) + (1 - deltas) * torch.log(
            1 - p_tau + 1e-10
        )
        return -torch.mean(loss)

    def update(self, sample_val, quantiles, lr):
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        # import pdb; pdb.set_trace()
        # print("updating inequality estimator", lr)
        self.optimizer.zero_grad()
        loss = self.loss(sample_val, quantiles)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


class LinearQL(object):
    # Not really linear, it is tabular

    def __init__(
        self,
        discount_factor,
        learning_rate,
        n_states,
        n_actions,
        verbose=False,
        clip_error=True,
        device=None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # print("Torch device", self.device)
        self.dtype = torch.float64
        self.discount_factor = discount_factor
        self.lr = learning_rate
        self.verbose = verbose
        self.clip_error = clip_error
        self.n_states = n_states
        self.n_actions = n_actions
        self.parameters = torch.normal(
            mean=0,
            std=0.1,
            size=(self.n_states, self.n_actions),
            requires_grad=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.var_params = 0.0 * torch.rand(
            size=self.parameters.shape,
            requires_grad=False,
            device=self.device,
            dtype=self.dtype,
        )

    def act(self, state):
        Q_a = self.parameters[state, :]
        action = torch.argmax(Q_a)
        return action.detach().cpu().numpy()

    def update(self, transitions, importance_weights=None, lr=None, var_lr=None):
        if lr is None:
            lr = self.lr
        td_errors = []
        updates = []
        for transition in transitions:
            td_error = self.compute_td_error(transition)
            td_errors.append(td_error.detach().cpu().numpy())
            update = (
                td_error  # *self.parameters[np.argmax(prev_state), transition.action]
            )
            updates.append(update)
            self.parameters[np.argmax(transition.prev_state), transition.action] += (
                lr * update
            )
        return td_errors

    def compute_td_error(self, transition):
        prev_state = transition.prev_state
        next_state = transition.next_state
        reward = torch.tensor(transition.reward, dtype=self.dtype, device=self.device)

        Q_a = self.parameters[np.argmax(next_state), :]
        if transition.terminate:
            td_error = (
                reward - self.parameters[np.argmax(prev_state), transition.action]
            )
        else:
            td_error = (
                reward
                + self.discount_factor * torch.max(Q_a)
                - self.parameters[np.argmax(prev_state), transition.action]
            )
        if self.clip_error:
            td_error = torch.clamp(td_error, min=-1.0, max=1.0)
        return td_error

    def oracle_transition_index(self, all_transitions, optimal_Q):
        mse_list = []
        optimal_Q = torch.tensor(optimal_Q, dtype=self.dtype, device=self.device)
        for transition in all_transitions:
            td_error = self.compute_td_error(transition).detach().cpu().numpy()
            new_parameters = torch.clone(self.parameters)
            new_parameters[np.argmax(transition.prev_state), transition.action] += (
                self.lr * td_error
            )
            mse = torch.mean((optimal_Q - new_parameters) ** 2)
            mse_list.append(mse.detach().cpu().numpy())

        return np.array(
            [
                np.argmin(mse_list),
            ]
        )

    def get_all_td_errors(self, transitions):
        td_errors = []
        for transition in transitions:
            td_error = self.compute_td_error(transition).detach().cpu().numpy()
            td_errors.append(td_error)
        return np.array(td_errors)


class PrioritizedReplayQL(LinearQL):
    def update(self, transitions, importance_weights=None, lr=None, var_lr=None):
        if lr is None:
            lr = self.lr
        importance_weights = torch.tensor(
            importance_weights, dtype=self.dtype, device=self.device
        )
        td_errors = []
        updates = []
        for i, transition in enumerate(transitions):
            td_error = self.compute_td_error(transition)
            td_errors.append(td_error.detach().cpu().numpy())
            update = (
                td_error  # *self.parameters[np.argmax(prev_state), transition.action]
            )
            updates.append(update)
            self.parameters[np.argmax(transition.prev_state), transition.action] += (
                lr * update * importance_weights[i]
            )
        return np.array(td_errors)


class DistributionalAgent(PrioritizedReplayQL):
    def __init__(
        self,
        discount_factor,
        learning_rate,
        n_states,
        n_actions,
        verbose=False,
        clip_error=True,
        init_range=(-1, 1),
        n_quantiles=21,
        priority_variable="td_error",
    ):
        super().__init__(
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            n_states=n_states,
            n_actions=n_actions,
            verbose=verbose,
            clip_error=clip_error,
        )

        self.init_range = init_range
        self.n_quantiles = n_quantiles
        self.priority_variable = priority_variable
        self.var_params = 0.1 * torch.ones(
            size=self.parameters.shape,
            requires_grad=False,
            device=self.device,
            dtype=self.dtype,
        )
        self._init_quantiles()
        self.ineq_estimator = {}
        for s in range(n_states):
            for a in range(n_actions):
                self.ineq_estimator[(s, a)] = InequalityEstimator(
                    n_quantiles=self.n_quantiles,
                    learning_rate=self.lr,
                    device=self.device,
                )

    def _init_quantiles(self):
        self.q_distr = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.q_distr[(s, a)] = QRTD(
                    n_quantiles=self.n_quantiles,
                    learning_rate=self.lr,
                    init_range=self.init_range,
                )
                self.parameters[s, a] = self.q_distr[(s, a)].mean()
        self.taus = np.linspace(0, 1, num=self.n_quantiles + 1, endpoint=True)
        self.tau_hat = np.linspace(
            np.diff(self.taus)[0] / 2,
            1 - np.diff(self.taus)[0] / 2,
            num=self.n_quantiles,
        )

    def update(self, transitions, importance_weights=None, lr=None, var_lr=None):
        if lr is None:
            lr = self.lr
        if importance_weights is None:
            importance_weights = np.ones(len(transitions))
        importance_weights = torch.tensor(
            importance_weights, dtype=self.dtype, device=self.device
        )
        td_errors = []
        updates = []
        for i, transition in enumerate(transitions):
            td_error = self.compute_td_error(transition)
            td_errors.append(td_error.detach().cpu().numpy())
            update = (
                td_error  # *self.parameters[np.argmax(prev_state), transition.action]
            )
            updates.append(update)
            self.q_distr[(np.argmax(transition.prev_state), transition.action)].lr = lr
            self.q_distr[(np.argmax(transition.prev_state), transition.action)].update(
                transition.reward,
            )
            h = self.q_distr[(np.argmax(transition.prev_state), transition.action)]
            self.parameters[
                np.argmax(transition.prev_state), transition.action
            ] = h.mean()
            self.var_params[
                np.argmax(transition.prev_state), transition.action
            ] = h.var()
        if self.priority_variable == "td_error":
            return np.array(td_errors)
        else:
            # warnings.warn("Unknown priority variable, returning td_error")
            return np.array(td_errors)

    def get_quantile_gradient(self, transitions):
        gradients = []
        for i, transition in enumerate(transitions):
            reward = torch.tensor(
                transition.reward,
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )[None, None]
            ineq_estimator = self.ineq_estimator[
                (np.argmax(transition.prev_state), transition.action)
            ]
            Fv = ineq_estimator.forward(reward)
            torch_tau_hat = torch.tensor(
                self.tau_hat, dtype=self.dtype, device=self.device, requires_grad=False
            )
            grad_norm = torch.sqrt(torch.mean((torch_tau_hat - Fv) ** 2))
            gradients.append(grad_norm.detach().cpu().numpy())
        return np.array(gradients)

    def update_ineq_estimator(self, transitions, lr=None):
        if lr is None:
            lr = self.lr
        losses = []
        for i, transition in enumerate(transitions):
            quantiles = torch.tensor(
                self.get_quantiles(state=0, action=transition.action),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )[:, None]
            reward = torch.tensor(
                transition.reward,
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )[None, None]
            loss = self.ineq_estimator[
                (np.argmax(transition.prev_state), transition.action)
            ].update(reward, quantiles, lr=lr)
            losses.append(loss)
        return np.array(losses)

    def get_quantiles(self, state, action):
        return self.q_distr[(state, action)].thetas

    def quantile_regression_loss(self, state, action, target):
        all_quantiles = self.get_quantiles(state, action)
        u = target - all_quantiles
        scaling = np.abs(self.tau_hat[np.newaxis, :] - (u < 0))
        loss = scaling * huber(1, u)
        return loss

    def get_quantile_loss(self, transitions):
        losses = []
        for i, transition in enumerate(transitions):
            loss = self.quantile_regression_loss(
                np.argmax(transition.prev_state), transition.action, transition.reward
            )
            losses.append(np.mean(loss))
        return np.array(losses)

    def get_all_quantiles(self):
        state = 0
        return np.stack(
            [self.get_quantiles(state, action) for action in range(self.n_actions)],
            axis=-1,
        )

    def get_pdfs(self):
        state = 0
        return np.stack(
            [self.q_distr[(state, action)].pdf() for action in range(self.n_actions)],
            axis=-1,
        )


def _setup_data_log(agents, num_episodes, change_frequency, bernoulli, num_arms):
    scalar_data_shape = (
        len(agents),
        num_episodes,
        change_frequency,
    )

    if bernoulli:
        dist_hist = np.zeros((num_episodes, num_arms))
    else:
        dist_hist = np.zeros((num_episodes, num_arms, 2))

    regret = np.zeros(scalar_data_shape)
    emp_regret = np.zeros(scalar_data_shape)
    correct_arm = np.zeros(scalar_data_shape)
    learning_rate = np.zeros(scalar_data_shape)
    temperature = np.zeros(scalar_data_shape)
    epistemic_uncertainty = np.zeros(scalar_data_shape)
    aleatoric_uncertainty = np.zeros(scalar_data_shape)
    oracle_epistemic_uncertainty = np.zeros(scalar_data_shape)
    oracle_aleatoric_uncertainty = np.zeros(scalar_data_shape)
    min_uncertainty = np.zeros(scalar_data_shape)
    argmin_uncertainty = np.zeros(scalar_data_shape)
    max_uncertainty = np.zeros(scalar_data_shape)
    argmax_uncertainty = np.zeros(scalar_data_shape)
    mean_uncertainty = np.zeros(scalar_data_shape)
    classifier_loss_log = np.zeros(scalar_data_shape)
    actions = np.zeros(scalar_data_shape)
    policy = np.zeros(scalar_data_shape + (num_arms,))
    moment_error = np.zeros(
        scalar_data_shape
        + (
            num_arms,
            2,
        )
    )

    scalar_logs = {}
    # for agent_arg in config.agents:
    #     for scalar in agent_arg.get(constants.SCALAR_LOG_SPEC, []):
    #         scalar_logs[scalar] = np.zeros(scalar_data_shape)
    # agent_spec = {**agent_spec, **agent_arg}

    # scalar_logs = {
    #     scalar: np.zeros(scalar_data_shape)
    #     for scalar in agent_spec.get(constants.SCALAR_LOG_SPEC, [])
    # }

    return (
        dist_hist,
        regret,
        emp_regret,
        correct_arm,
        learning_rate,
        temperature,
        epistemic_uncertainty,
        aleatoric_uncertainty,
        oracle_epistemic_uncertainty,
        oracle_aleatoric_uncertainty,
        min_uncertainty,
        argmin_uncertainty,
        max_uncertainty,
        argmax_uncertainty,
        mean_uncertainty,
        classifier_loss_log,
        actions,
        policy,
        moment_error,
        scalar_logs,
    )


def train(
    seed,
    agents,
    agent_order,
    change_frequency,
    num_episodes,
    dists,
    bernoulli,
    best_arms,
):
    (
        dist_hist_log,
        regret_log,
        emp_regret_log,
        correct_arm_log,
        learning_rate_log,
        temperature_log,
        epistemic_uncertainty_log,
        aleatoric_uncertainty_log,
        oracle_epistemic_uncertainty_log,
        oracle_aleatoric_uncertainty_log,
        min_uncertainty_log,
        argmin_uncertainty_log,
        max_uncertainty_log,
        argmax_uncertainty_log,
        mean_uncertainty_log,
        classifier_loss_log,
        actions_log,
        policy_log,
        moment_error_log,
        scalar_logs,
    ) = _setup_data_log(
        agents=agents.keys(),
        num_episodes=num_episodes,
        change_frequency=change_frequency,
        bernoulli=bernoulli,
        num_arms=NUM_ARMS,
    )

    rng = np.random.RandomState(seed)

    for episode in range(num_episodes):
        if bernoulli:
            dist_hist_log[episode] = [d.mean() for d in dists[seed][episode]]
        else:
            dist_hist_log[episode][:, 0] = [d.mean() for d in dists[seed][episode]]
            dist_hist_log[episode][:, 1] = [d.std() for d in dists[seed][episode]]

        for i, name in enumerate(agent_order):
            agent = agents[name][seed]
            dist = dists[seed][episode]
            best_arm = best_arms[seed][episode]

            # oracle to agents
            try:
                agent.dist = dist
            except:
                pass

            for trial in range(change_frequency):
                action = agent.play(episode)
                arm_sample = dist[action].rvs(random_state=rng)

                emp_regret_log[i, episode, trial] = dist[best_arm].mean() - arm_sample
                regret_log[i, episode, trial] = (
                    dist[best_arm].mean() - dist[action].mean()
                )
                correct_arm_log[i, episode, trial] = best_arm == action

                (
                    epi,
                    ale,
                    oracle_epi,
                    oracle_ale,
                    lr,
                    temp,
                    min_epi,
                    argmin_epi,
                    max_epi,
                    argmax_epi,
                    mean_epi,
                    c_loss,
                ) = agent.update(action, arm_sample, episode)

                learning_rate_log[i, episode, trial] = lr
                temperature_log[i, episode, trial] = temp
                actions_log[i, episode, trial] = action
                policy_log[i, episode, trial] = agent.policy(episode)
                epistemic_uncertainty_log[i, episode, trial] = epi
                oracle_epistemic_uncertainty_log[i, episode, trial] = oracle_epi
                oracle_aleatoric_uncertainty_log[i, episode, trial] = oracle_ale
                min_uncertainty_log[i, episode, trial] = min_epi
                argmin_uncertainty_log[i, episode, trial] = argmin_epi
                max_uncertainty_log[i, episode, trial] = max_epi
                argmax_uncertainty_log[i, episode, trial] = argmax_epi
                mean_uncertainty_log[i, episode, trial] = mean_epi
                aleatoric_uncertainty_log[i, episode, trial] = ale
                classifier_loss_log[i, episode, trial] = c_loss

                means, vars = agent.predict_bandits()

                if bernoulli:
                    moment_error_log[i, episode, trial, 0] = (
                        dist_hist_log[episode] - means
                    ) ** 2
                else:
                    moment_error_log[i, episode, trial, :, 0] = (
                        dist_hist_log[episode, :, 0] - means
                    ) ** 2
                    moment_error_log[i, episode, trial, :, 1] = (
                        dist_hist_log[episode, :, 1] ** 2 - vars
                    ) ** 2

                log_scalars = agent.scalar_log()

                for scalar_name, scalar_value in log_scalars.items():
                    scalar_logs[scalar_name][i, episode, trial] = scalar_value

    # save data
    data = {
        "emp_regret": emp_regret_log,
        "regret": regret_log,
        "actions": actions_log,
        "policy": policy_log,
        "temperature": temperature_log,
        "learning_rate": learning_rate_log,
        "epistemic_uncertainty": epistemic_uncertainty_log,
        "aleatoric_uncertainty": aleatoric_uncertainty_log,
        "oracle_epistemic_uncertainty": oracle_epistemic_uncertainty_log,
        "oracle_aleatoric_uncertainty": oracle_aleatoric_uncertainty_log,
        "min_uncertainty": min_uncertainty_log,
        "argmin_uncertainty": argmin_uncertainty_log,
        "max_uncertainty": max_uncertainty_log,
        "argmax_uncertainty": argmax_uncertainty_log,
        "mean_uncertainty": mean_uncertainty_log,
        "classifier_loss": classifier_loss_log,
        "best_arms": best_arms,
        "correct_arm": correct_arm_log,
        "scalar_logs": scalar_logs,
        "moment_error": moment_error_log,
        "agent_order": agent_order,
        "dists": dist_hist_log,
    }
    np.savez(os.path.join(RESULTS_PATH, f"seed_{seed}.npz"), data)


if __name__ == "__main__":
    dists, best_arms = _setup_bandit(
        num_arms=NUM_ARMS,
        mean_upper=MEAN_RANGE[1],
        mean_lower=MEAN_RANGE[0],
        scale_upper=SCALE_RANGE[1],
        scale_lower=SCALE_RANGE[0],
        num_seeds=NUM_SEEDS,
        num_episodes=NUM_EPISODES,
        change_probability=CHANGE_PROBABILITY,
    )

    agents = {}
    agents[f"baseline"] = [
        QR(
            num_arms=NUM_ARMS,
            rho=1.0,
            gamma=1.0,
            ucb=False,
            n_quantiles=N_QUANTILES,
            adapt_lr=None,
            adapt_temp=None,
            learning_rate=0.25,
            temperature=5,
            init_range=(-1, 1),
            true_dists=dists[d],
            scalar_log_spec=[],
        )
        for d in range(NUM_SEEDS)
    ]
    agents["oracle"] = [
        QR(
            num_arms=NUM_ARMS,
            rho=1.0,
            gamma=1,
            ucb=False,
            n_quantiles=N_QUANTILES,
            adapt_lr={"type": "oracle_epistemic_ratio", "factor": 1},
            adapt_temp={"type": "oracle_epistemic", "factor": 0.5},
            learning_rate=None,
            temperature=None,
            init_range=(-1, 1),
            true_dists=dists[d],
            scalar_log_spec=[],
        )
        for d in range(NUM_SEEDS)
    ]
    agents["oracle_2"] = [
        QR(
            num_arms=NUM_ARMS,
            rho=1.0,
            gamma=1,
            ucb=False,
            n_quantiles=N_QUANTILES,
            adapt_lr={"type": "oracle_epistemic_ratio_2", "factor": 1},
            adapt_temp={"type": "oracle_epistemic", "factor": 1},
            learning_rate=None,
            temperature=None,
            init_range=(-1, 1),
            true_dists=dists[d],
            scalar_log_spec=[],
        )
        for d in range(NUM_SEEDS)
    ]
    # for lr in [0.5, 0.25, 0.1, 0.01, 0.001]:
    #     for gamma in [0.99, 0.999, 0.9999]:
    #         agents[f"ducb_{lr}_{gamma}"] = [
    #             QR(
    #                 num_arms=NUM_ARMS,
    #                 rho=1.0,
    #                 gamma=gamma,
    #                 ucb=True,
    #                 n_quantiles=N_QUANTILES,
    #                 adapt_lr=None,
    #                 adapt_temp=None,
    #                 learning_rate=lr,
    #                 temperature=None,
    #                 init_range=(-1, 1),
    #                 scalar_log_spec=[],
    #             )
    #             for _ in range(NUM_SEEDS)
    #         ]

    #     for temperature in [0.1, 1, 5, 10]:
    #         agents[f"qr_{lr}_{temperature}"] = [
    #             QR(
    #                 num_arms=NUM_ARMS,
    #                 rho=1.0,
    #                 gamma=1.0,
    #                 ucb=False,
    #                 n_quantiles=N_QUANTILES,
    #                 adapt_lr=None,
    #                 adapt_temp=None,
    #                 learning_rate=lr,
    #                 temperature=temperature,
    #                 init_range=(-1, 1),
    #                 scalar_log_spec=[],
    #             )
    #             for _ in range(NUM_SEEDS)
    #         ]

    for factor_1 in [0.01]:
        # agents[f"qr_adapt_lr_{factor_1}"] = [
        #     QR(
        #         num_arms=NUM_ARMS,
        #         rho=1.0,
        #         gamma=gamma,
        #         ucb=False,
        #         n_quantiles=N_QUANTILES,
        #         adapt_lr={"type": "ratio", "factor": factor_1},
        #         adapt_temp=None,
        #         learning_rate=None,
        #         temperature=temperature,
        #         init_range=(-1, 1),
        #         scalar_log_spec=[],
        #     )
        #     for _ in range(NUM_SEEDS)
        # ]
        # agents[f"qr_adapt_lr2_{factor_1}"] = [
        #     QR(
        #         num_arms=NUM_ARMS,
        #         rho=1.0,
        #         gamma=gamma,
        #         ucb=False,
        #         n_quantiles=N_QUANTILES,
        #         adapt_lr={"type": "ratio_2", "factor": factor_1},
        #         adapt_temp=None,
        #         learning_rate=None,
        #         temperature=temperature,
        #         init_range=(-1, 1),
        #         scalar_log_spec=[],
        #     )
        #     for _ in range(NUM_SEEDS)
        # ]
        for factor_2 in [0.1, 0.5, 1, 5]:
            # agents[f"qr_adapt_lr_{factor_1}_temp_{factor_2}_oracle"] = [
            #     QR(
            #         num_arms=NUM_ARMS,
            #         rho=1.0,
            #         gamma=1,
            #         ucb=False,
            #         n_quantiles=N_QUANTILES,
            #         adapt_lr={"type": "oracle_epistemic_ratio", "factor": factor_1},
            #         adapt_temp={"type": "oracle_epistemic", "factor": factor_2},
            #         learning_rate=None,
            #         temperature=None,
            #         init_range=(-1, 1),
            #         true_dists=dists[d],
            #         scalar_log_spec=[],
            #     )
            #     for d in range(NUM_SEEDS)
            # ]
            # agents[f"qr_adapt_lr2_{factor_1}_temp_{factor_2}"] = [
            #     QR(
            #         num_arms=NUM_ARMS,
            #         rho=1.0,
            #         gamma=1,
            #         ucb=False,
            #         n_quantiles=N_QUANTILES,
            #         adapt_lr={"type": "oracle_epistemic_ratio_2", "factor": factor_1},
            #         adapt_temp={"type": "epistemic", "factor": factor_2},
            #         learning_rate=None,
            #         temperature=None,
            #         init_range=(-1, 1),
            #         true_dists=dists[d],
            #         scalar_log_spec=[],
            #     )
            #     for d in range(NUM_SEEDS)
            # ]
            agents[f"qr_adapt_lr2_{factor_1}_temp_{factor_2}"] = [
                QR(
                    num_arms=NUM_ARMS,
                    rho=1.0,
                    gamma=1,
                    ucb=False,
                    n_quantiles=N_QUANTILES,
                    adapt_lr={
                        "type": "ratio_2",
                        "factor": factor_1,
                        "ineq_lr": "oracle_epistemic_ratio",
                    },
                    adapt_temp={"type": "epistemic", "factor": factor_2},
                    learning_rate=None,
                    temperature=None,
                    init_range=(-1, 1),
                    true_dists=dists[d],
                    scalar_log_spec=[],
                )
                for d in range(NUM_SEEDS)
            ]
            # agents[f"qr_adapt_lr2_{factor_1}_temp_{factor_2}_min"] = [
            #     QR(
            #         num_arms=NUM_ARMS,
            #         rho=1.0,
            #         gamma=1,
            #         ucb=False,
            #         n_quantiles=N_QUANTILES,
            #         adapt_lr={"type": "ratio_2", "factor": factor_1},
            #         adapt_temp={"type": "min_epistemic", "factor": factor_2},
            #         learning_rate=None,
            #         temperature=None,
            #         init_range=(-1, 1),
            #         scalar_log_spec=[],
            #     )
            #     for _ in range(NUM_SEEDS)
            # ]
            # agents[f"qr_adapt_lr2_{factor_1}_temp_{factor_2}_max"] = [
            #     QR(
            #         num_arms=NUM_ARMS,
            #         rho=1.0,
            #         gamma=1,
            #         ucb=False,
            #         n_quantiles=N_QUANTILES,
            #         adapt_lr={"type": "ratio_2", "factor": factor_1},
            #         adapt_temp={"type": "max_epistemic", "factor": factor_2},
            #         learning_rate=None,
            #         temperature=None,
            #         init_range=(-1, 1),
            #         scalar_log_spec=[],
            #     )
            #     for _ in range(NUM_SEEDS)
            # ]
            # agents[f"qr_adapt_lr2_{factor_1}_temp_{factor_2}_ucb"] = [
            #     QR(
            #         num_arms=NUM_ARMS,
            #         rho=1.0,
            #         gamma=1,
            #         ucb=False,
            #         n_quantiles=N_QUANTILES,
            #         adapt_lr={"type": "ratio_2", "factor": factor_1},
            #         adapt_temp={"type": "epistemic_per_arm_bonus", "factor": factor_2},
            #         learning_rate=None,
            #         temperature=None,
            #         init_range=(-1, 1),
            #         scalar_log_spec=[],
            #     )
            #     for _ in range(NUM_SEEDS)
            # ]

    # agents["qr_adapt_temp"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr=None,
    #         adapt_temp={"type": "epistemic"},
    #         learning_rate=lr,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_lr"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "ratio"},
    #         adapt_temp={"type": "epistemic"},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_lr2"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "ratio_2"},
    #         adapt_temp={"type": "epistemic"},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_10"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr=None,
    #         adapt_temp={"type": "epistemic", "factor": 10},
    #         learning_rate=lr,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_lr_10"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "ratio"},
    #         adapt_temp={"type": "epistemic", "factor": 10},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_lr2_10"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "ratio_2"},
    #         adapt_temp={"type": "epistemic", "factor": 10},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_100"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr=None,
    #         adapt_temp={"type": "epistemic", "factor": 100},
    #         learning_rate=lr,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_lr_100"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "ratio"},
    #         adapt_temp={"type": "epistemic", "factor": 100},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]
    # agents["qr_adapt_temp_lr2_100"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=gamma,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "ratio_2"},
    #         adapt_temp={"type": "epistemic", "factor": 100},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         scalar_log_spec=[],
    #     )
    #     for _ in range(NUM_SEEDS)
    # ]

    seed_runs = []

    # agents = {}
    # agents["test"] = [
    #     QR(
    #         num_arms=NUM_ARMS,
    #         rho=1.0,
    #         gamma=1,
    #         ucb=False,
    #         n_quantiles=N_QUANTILES,
    #         adapt_lr={"type": "oracle_epistemic_ratio", "factor": 100},
    #         adapt_temp={"type": "oracle_epistemic", "factor": 100},
    #         learning_rate=None,
    #         temperature=None,
    #         init_range=(-1, 1),
    #         true_dists=dists[d],
    #         scalar_log_spec=[],
    #     )
    #     for d in range(NUM_SEEDS)
    # ]

    # train(
    #     0,
    #     agents,
    #     list(agents.keys()),
    #     EPISODE_LENGTH,
    #     NUM_EPISODES,
    #     dists,
    #     False,
    #     best_arms,
    # )

    for seed in range(NUM_SEEDS):
        process = mp.Process(
            target=train,
            args=(
                seed,
                agents,
                list(agents.keys()),
                EPISODE_LENGTH,
                NUM_EPISODES,
                dists,
                False,
                best_arms,
            ),
        )
        process.start()
        seed_runs.append(process)

    for process in seed_runs:
        process.join()
