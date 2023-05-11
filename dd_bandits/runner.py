import functools
import multiprocessing as mp
import os
from typing import List

import numpy as np
import scipy
from run_modes import base_runner

from dd_bandits import constants
from dd_bandits.dd_mabs import td_doya_dayu_, td_doya_dayu_memory
from dd_bandits.mabs import boltzmann, discounted_ucb
from dd_bandits.td_mabs import td_boltzmann, td_discounted_ucb
from dd_bandits.utils import plotting_functions


class Runner(base_runner.BaseRunner):
    def __init__(self, config, unique_id) -> None:

        self._rng = np.random.RandomState(config.seed)

        self._num_arms = config.num_arms
        self._seed_start = config.seed_start
        self._num_seeds = config.num_seeds
        self._num_episodes = config.num_episodes
        self._change_frequency = config.change_frequency
        self._bernoulli = config.bernoulli

        super().__init__(config=config, unique_id=unique_id)

        self._agents = self._setup_agents(config=config)
        self._agent_order = sorted(self._agents)

        (
            self._dist_hist,
            self._regret,
            self._correct_arm,
            self._learning_rate,
            self._temperature,
            self._epistemic_uncertainty,
            self._aleatoric_uncertainty,
            self._min_uncertainty,
            self._policy,
            self._moment_error,
            self._scalar_logs,
        ) = self._setup_data_log(config=config)

        self._dists, self._best_arms = self._setup_bandit(config=config)

        self._array_path = os.path.join(self._checkpoint_path, constants.ARRAYS)
        self._plot_path = os.path.join(self._checkpoint_path, constants.PLOTS)
        os.mkdir(self._array_path)
        os.mkdir(self._plot_path)
        np.save(
            os.path.join(self._array_path, constants.AGENT_ORDER), self._agent_order
        )

    def _setup_data_log(self, config):
        scalar_data_shape = (
            len(self._agents),
            config.num_episodes,
            config.change_frequency,
        )

        if config.bernoulli:
            dist_hist = np.zeros(
                (config.num_seeds, config.num_episodes, config.num_arms)
            )
        else:
            dist_hist = np.zeros(
                (config.num_seeds, config.num_episodes, config.num_arms, 2)
            )

        regret = np.zeros(scalar_data_shape)
        correct_arm = np.zeros(scalar_data_shape)
        learning_rate = np.zeros(scalar_data_shape)
        temperature = np.zeros(scalar_data_shape)
        epistemic_uncertainty = np.zeros(scalar_data_shape)
        aleatoric_uncertainty = np.zeros(scalar_data_shape)
        min_uncertainty = np.zeros(scalar_data_shape)
        policy = np.zeros(scalar_data_shape + (config.num_arms,))
        moment_error = np.zeros(scalar_data_shape + (2,))

        scalar_logs = {}
        for agent_arg in config.agents:
            for scalar in agent_arg.get(constants.SCALAR_LOG_SPEC, []):
                scalar_logs[scalar] = np.zeros(scalar_data_shape)
            # agent_spec = {**agent_spec, **agent_arg}

        # scalar_logs = {
        #     scalar: np.zeros(scalar_data_shape)
        #     for scalar in agent_spec.get(constants.SCALAR_LOG_SPEC, [])
        # }

        return (
            dist_hist,
            regret,
            correct_arm,
            learning_rate,
            temperature,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            min_uncertainty,
            policy,
            moment_error,
            scalar_logs,
        )

    def _get_data_columns(self) -> List[str]:
        return []

    def _setup_agents(self, config):
        agent_specs = np.load(
            os.path.join(self._checkpoint_path, "agents.npy"), allow_pickle=True
        )[()]

        agents = {}

        for agent_spec in agent_specs:
            agent_type = agent_spec.pop(constants.AGENT)
            agent_name = agent_spec.pop(constants.NAME)
            if agent_type == constants.TD_DOYA_DAYU:
                agent_class = td_doya_dayu_.DoyaDayu
            elif agent_type == constants.TD_DOYA_DAYU_MEMORY:
                agent_class = td_doya_dayu_memory.DoyaDaYu
            elif agent_type == constants.BOLTZMANN:
                agent_class = boltzmann.Boltzmann
            elif agent_type == constants.TD_BOLTZMANN:
                agent_class = td_boltzmann.Boltzmann
            elif agent_type == constants.DUCB:
                agent_class = discounted_ucb.DiscountedUCB
            elif agent_type == constants.TD_DUCB:
                agent_class = td_discounted_ucb.DiscountedUCB
            else:
                raise ValueError(f"Agent class {agent_type} not recognised.")

            base_args = {constants.NUM_ARMS: config.num_arms}
            agent_args = {**base_args, **agent_spec}

            agents[agent_name] = functools.partial(agent_class, **agent_args)

        return agents

    def _setup_bandit(self, config):
        def _sample_mean(lower, upper):
            return self._rng.uniform(lower, upper)

        def _sample_scale(lower, upper):
            return self._rng.uniform(lower, upper)

        def _sample_probability():
            return self._rng.uniform(0, 1)

        dists = [[] for _ in range(config.num_seeds)]
        best_arms = [[] for _ in range(config.num_seeds)]

        random_samples = np.random.random(size=(config.num_seeds, config.num_episodes))
        changes = [
            [
                episode == 0 or r < config.change_probability
                for episode, r in enumerate(ran)
            ]
            for ran in random_samples
        ]

        for seed in range(config.num_seeds):
            for episode in range(config.num_episodes):
                change = changes[seed][episode]
                if not change:
                    seed_ep_dists = dists[seed][episode - 1]
                    best_arm = best_arms[seed][episode - 1]
                    self._dist_hist[seed, episode] = self._dist_hist[seed, episode - 1]
                else:
                    if config.bernoulli:
                        probs = np.array(
                            [(_sample_probability()) for _ in range(config.num_arms)]
                        )
                        self._dist_hist[seed, episode] = probs
                        seed_ep_dists = [
                            scipy.stats.bernoulli(probs[a])
                            for a in range(config.num_arms)
                        ]
                        best_arm = np.array([d.mean() for d in seed_ep_dists]).argmax()
                    else:
                        mean_vars = np.array(
                            [
                                (
                                    _sample_mean(config.mean_lower, config.mean_upper),
                                    _sample_scale(
                                        config.scale_lower, config.scale_upper
                                    ),
                                )
                                for _ in range(config.num_arms)
                            ]
                        )
                        self._dist_hist[seed, episode] = mean_vars
                        seed_ep_dists = [
                            scipy.stats.norm(mean_vars[a, 0], mean_vars[a, 1])
                            for a in range(config.num_arms)
                        ]
                        best_arm = np.array([d.mean() for d in seed_ep_dists]).argmax()

                dists[seed].append(seed_ep_dists)
                best_arms[seed].append(best_arm)

        return dists, best_arms

    @staticmethod
    def single_seed_train(
        seed,
        agents,
        agent_order,
        change_frequency,
        num_episodes,
        dists,
        regret,
        correct_arm,
        learning_rate,
        temperature,
        bernoulli,
        rng,
        moment_error,
        dist_hist,
        scalar_logs,
        best_arms,
        checkpoint_path,
    ):

        agent_instances = {name: agent() for name, agent in agents.items()}

        for episode in range(num_episodes):

            print(f"Seed: {seed}, Episode: {episode}")

            for i, name in enumerate(agent_order):

                # agent = experiment_agents[name][1]()
                agent = agent_instances[name]
                dist = dists[episode]
                best_arm = best_arms[episode]

                # oracle to agents
                try:
                    agent.dist = dist
                except:
                    pass

                for trial in range(change_frequency):

                    action = agent.play()
                    regret[i, episode, trial] = (
                        dist[best_arm].mean() - dist[action].mean()
                    )
                    correct_arm[i, episode, trial] = best_arm == action

                    lr = agent.learning_rate(action)
                    learning_rate[i, episode, trial] = lr
                    temperature[i, episode, trial] = agent.temperature()
                    # self._min_uncertainty[
                    #     i, seed, episode, trial
                    # ] = agent.min_epistemic_uncertainty
                    # self._aleatoric_uncertainty[
                    #     i, seed, episode, trial
                    # ] = agent.aleatoric_uncertainty
                    # self._epistemic_uncertainty[
                    #     i, seed, episode, trial
                    # ] = agent.epistemic_uncertainty
                    # self._policy[i, seed, episode, trial] = agent.policy()

                    agent.update(action, dist[action].rvs(random_state=rng))
                    means, vars = agent.predict_bandits()

                    if bernoulli:
                        moment_error[i, episode, trial, 0] = (
                            dist_hist[episode, best_arm] - means[best_arm]
                        ) ** 2
                    else:
                        moment_error[i, episode, trial, 0] = (
                            dist_hist[episode, best_arm, 0] - means[best_arm]
                        ) ** 2
                        moment_error[i, episode, trial, 1] = (
                            dist_hist[episode, best_arm, 1] ** 2 - vars[best_arm]
                        ) ** 2

                    log_scalars = agent.scalar_log()

                    # import pdb

                    # pdb.set_trace()

                    for scalar_name, scalar_value in log_scalars.items():
                        scalar_logs[scalar_name][i, episode, trial] = scalar_value

        data = {
            "regret": regret,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "best_arms": best_arms,
            "correct_arm": correct_arm,
            "scalar_logs": scalar_logs,
            "moment_error": moment_error,
        }
        np.save(os.path.join(checkpoint_path, f"seed_{seed}_data"), data)

        # return (
        #     regret,
        #     temperature,
        #     learning_rate,
        #     best_arms,
        #     correct_arm,
        #     scalar_logs,
        #     moment_error,
        # )

    def train(self):

        seed_runs = []

        # manager = mp.Manager()
        # return_dict = manager.dict()
        # queues = [mp.Queue() for _ in range(self._num_seeds)]

        for seed in range(self._seed_start, self._seed_start + self._num_seeds):

            # Runner.single_seed_train(
            #     seed,
            #     self._agents,
            #     self._agent_order,
            #     self._change_frequency,
            #     self._num_episodes,
            #     self._dists[seed],
            #     self._regret,
            #     self._correct_arm,
            #     self._learning_rate,
            #     self._temperature,
            #     self._bernoulli,
            #     self._rng,
            #     self._moment_error,
            #     self._dist_hist[seed],
            #     self._scalar_logs,
            #     self._best_arms[seed],
            #     self._checkpoint_path,
            # )

            process = mp.Process(
                target=Runner.single_seed_train,
                args=(
                    seed,
                    self._agents,
                    self._agent_order,
                    self._change_frequency,
                    self._num_episodes,
                    self._dists[seed],
                    self._regret,
                    self._correct_arm,
                    self._learning_rate,
                    self._temperature,
                    self._bernoulli,
                    self._rng,
                    self._moment_error,
                    self._dist_hist[seed],
                    self._scalar_logs,
                    self._best_arms[seed],
                    self._checkpoint_path,
                ),
            )
            process.start()
            seed_runs.append(process)

        for process in seed_runs:
            process.join()

        regrets = []
        temperatures = []
        learning_rates = []
        best_arms = []
        correct_arms = []
        moment_errors = []
        scalar_logs = {k: [] for k in self._scalar_logs.keys()}

        for seed in range(self._num_seeds):
            data = np.load(
                os.path.join(self._checkpoint_path, f"seed_{seed}_data.npy"),
                allow_pickle=True,
            )[()]
            regrets.append(data["regret"])
            temperatures.append(data["temperature"])
            learning_rates.append(data["learning_rate"])
            best_arms.append(data["best_arms"])
            correct_arms.append(data["correct_arm"])
            moment_errors.append(data["moment_error"])
            for k, v in data["scalar_logs"].items():
                scalar_logs[k].append(v)

        self._regret = np.swapaxes(np.stack(regrets), 0, 1)
        self._temperature = np.swapaxes(np.stack(temperatures), 0, 1)
        self._correct_arm = np.swapaxes(np.stack(correct_arms), 0, 1)
        self._moment_error = np.swapaxes(np.stack(moment_errors), 0, 1)
        self._best_arms = np.swapaxes(np.stack(best_arms), 0, 1)
        self._learning_rate = np.swapaxes(np.stack(learning_rates), 0, 1)

        self._scalar_logs = {
            k: np.swapaxes(np.stack(v), 0, 1) for k, v in scalar_logs.items()
        }

    def post_process(self):

        for array, array_name in [
            (self._dist_hist, constants.DIST_HIST),
            (self._regret, constants.REGRET),
            (self._correct_arm, constants.CORRECT_ARM),
            (self._learning_rate, constants.LEARNING_RATE),
            (self._temperature, constants.TEMPERATURE),
            (self._epistemic_uncertainty, constants.EPISTEMIC_UNCERTAINTY),
            (self._aleatoric_uncertainty, constants.ALEATORIC_UNCERTAINTY),
            (self._min_uncertainty, constants.MIN_UNCERTAINTY),
            (self._policy, constants.POLICY),
            (self._moment_error, constants.MOMENT_ERROR),
        ]:
            np.save(file=os.path.join(self._array_path, array_name), arr=array)

        for array_name, array in self._scalar_logs.items():
            np.save(file=os.path.join(self._array_path, array_name), arr=array)

        plotting_functions.average_best_arm_plot(
            agent_order=self._agent_order,
            correct_arm=self._correct_arm,
            change_freq=self._change_frequency,
            save_path=self._plot_path,
        )
        plotting_functions.average_regret_plot(
            agent_order=self._agent_order,
            regret=self._regret,
            change_freq=self._change_frequency,
            save_path=self._plot_path,
        )
        plotting_functions.cumulative_regret(
            agent_order=self._agent_order,
            regret=self._regret,
            save_path=self._plot_path,
        )
        plotting_functions.policies(
            agent_order=self._agent_order,
            policies=self._policy,
            save_path=self._plot_path,
        )
        plotting_functions.mean_mses(
            agent_order=self._agent_order,
            mses=self._moment_error,
            save_path=self._plot_path,
        )
        plotting_functions.variance_mses(
            agent_order=self._agent_order,
            mses=self._moment_error,
            save_path=self._plot_path,
        )
        for (scalar_label, scalar_data, plot_label, logscale) in [
            (
                "aleatoric_uncertainty",
                self._aleatoric_uncertainty,
                "Aleatoric Uncertainty",
                False,
            ),
            (
                "epistemic_uncertainty",
                self._epistemic_uncertainty,
                "Epistemic Uncertainty",
                False,
            ),
            (
                "min_uncertainty",
                self._min_uncertainty,
                "Minimum Epistemic Uncertainty",
                False,
            ),
            ("temperature", self._temperature, "Temperature", False),
            ("learning_rate", self._learning_rate, "Learning Rate", False),
        ] + [(name, arr, name, False) for name, arr in self._scalar_logs.items()]:
            plotting_functions.scalar_plot(
                agent_order=self._agent_order,
                scalar_data=scalar_data,
                label=plot_label,
                save_path=os.path.join(self._plot_path, f"{scalar_label}.pdf"),
                logscale=logscale,
            )
