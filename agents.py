import functools

import numpy as np
import optax

import dd_mabs.doya_dayu
import dd_mabs.td_doya_dayu
import doya_dayu_tab
import ducb
import mabs.boltzmann
import mabs.discounted_ucb
import mabs.doya_dayu
import mabs.epsilon_greedy
import mabs.thompson_gaussian
import td_mabs.boltzmann
import td_mabs.discounted_ucb
import td_mabs.epsilon_greedy


class Agents:
    def __init__(self, rng, num_actions: int, learning_rate: float) -> None:
        self._rng = rng
        self._num_actions = num_actions
        self._learning_rate = learning_rate

        self._agents = self._setup_agents()

    def _setup_agents(self):
        # agents = {}
        # for i in [5, 10, 20, 100]:
        #     for j in np.linspace(0, 1, 11):
        #         for k in [0.5, 1, 1.5, 2]:
        #             agents[f"dd_{i}_{j}_{k}"] = (
        #                 f"DD {i} {j} {k}",
        #                 functools.partial(
        #                     doya_dayu_tab.DoyaDaYuTabular,
        #                     self._num_actions,
        #                     i,
        #                     adapt_lrate=True,
        #                     adapt_temperature=True,
        #                     rng=self._rng,
        #                     mask_p=j,
        #                     Q0=0.5,
        #                     S0=0.01,
        #                     lrate_per_arm=True,
        #                     lr_noise_multiplier=k,
        #                 ),
        #             )

        # agents["ducb"] = (
        #     "Discounted UCB 99",
        #     functools.partial(
        #         mabs.discounted_ucb.DiscountedUCB,
        #         n_arms=self._num_actions,
        #         rho=1.0,
        #         gamma=0.99,
        #         rng=self._rng,
        #     ),
        # )

        # return agents

        return dict(
            # egreedy=(
            #     "Epsilon Greedy",
            #     functools.partial(
            #         mabs.epsilon_greedy.EpsilonGreedy,
            #         n_arms=self._num_actions,
            #         epsilon=0.05,
            #         rng=self._rng,
            #     ),
            # ),
            # td_egreedy=(
            #     "TD Epsilon Greedy",
            #     functools.partial(
            #         td_mabs.epsilon_greedy.EpsilonGreedy,
            #         n_arms=self._num_actions,
            #         epsilon=0.05,
            #         learning_rate=self._learning_rate,
            #         rng=self._rng,
            #     ),
            # ),
            # boltzmann=(
            #     "Boltzmann",
            #     functools.partial(
            #         mabs.boltzmann.Boltzmann,
            #         n_arms=self._num_actions,
            #         temperature=0.15,
            #         rng=self._rng,
            #     ),
            # ),
            taddd=(
                "TD DoyaDaYu 1.25",
                functools.partial(
                    dd_mabs.td_doya_dayu.DoyaDaYu,
                    n_arms=self._num_actions,
                    n_ens=20,
                    learning_rate=None,
                    adapt_temperature=True,
                    rng=self._rng,
                    mask_p=1.0,
                    Q0=0.0,
                    S0=0.5,
                    lrate_per_arm=True,
                    lr_noise_multiplier=None,
                    use_direct=False,
                    aleatoric="variance",
                ),
            ),
            td_boltzmann=(
                "TD Boltzmann",
                functools.partial(
                    td_mabs.boltzmann.Boltzmann,
                    n_arms=self._num_actions,
                    temperature=0.25,
                    learning_rate=0.1,
                    rng=self._rng,
                ),
            ),
            # ucb=(
            #     "UCB",
            #     functools.partial(
            #         mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=1,
            #         rng=self._rng,
            #     ),
            # ),
            # thompson=(
            #     "Thompson",
            #     functools.partial(
            #         mabs.thompson_gaussian.ThompsonSamplingGaussian,
            #         n_arms=self._num_actions,
            #         rng=self._rng,
            #     ),
            # ),
            td_ucb=(
                "TD UCB",
                functools.partial(
                    td_mabs.discounted_ucb.DiscountedUCB,
                    n_arms=self._num_actions,
                    rho=1.0,
                    gamma=1.0,
                    learning_rate=0.1,
                    rng=self._rng,
                ),
            ),
            # ducb2=(
            #     "Discounted UCB 999",
            #     functools.partial(
            #         mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.999,
            #         rng=self._rng,
            #     ),
            # ),
            # ducb3=(
            #     "Discounted UCB 9999",
            #     functools.partial(
            #         mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.9999,
            #         rng=self._rng,
            #     ),
            # ),
            # ddtab=(
            #     "DoyaDaYu 1.5",
            #     functools.partial(
            #         dd_mabs.doya_dayu.DoyaDaYu,
            #         self._num_actions,
            #         40,
            #         learning_rate=None,
            #         adapt_temperature=True,
            #         rng=self._rng,
            #         mask_p=1.0,
            #         Q0=0.5,
            #         S0=0.01,
            #         lrate_per_arm=True,
            #         lr_noise_multiplier=1.5,
            #         use_direct=False,
            #         aleatoric="variance",
            #     ),
            # ),
            td_ducb=(
                "TD Discounted UCB 0.1 0.99",
                functools.partial(
                    td_mabs.discounted_ucb.DiscountedUCB,
                    n_arms=self._num_actions,
                    rho=1.0,
                    gamma=0.99,
                    learning_rate=0.1,
                    rng=self._rng,
                ),
            ),
            # td_ducb2=(
            #     "TD Discounted UCB 0.1 0.999",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.999,
            #         learning_rate=0.1,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb3=(
            #     "TD Discounted UCB 0.1 0.9999",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.9999,
            #         learning_rate=0.1,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb4=(
            #     "TD Discounted UCB 0.2 0.99",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.99,
            #         learning_rate=0.2,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb5=(
            #     "TD Discounted UCB 0.2 0.999",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.999,
            #         learning_rate=0.2,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb6=(
            #     "TD Discounted UCB 0.2 0.9999",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.9999,
            #         learning_rate=0.2,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb7=(
            #     "TD Discounted UCB 0.5 0.99",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.99,
            #         learning_rate=0.5,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb8=(
            #     "TD Discounted UCB 0.5 0.999",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.999,
            #         learning_rate=0.5,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb9=(
            #     "TD Discounted UCB 0.5 0.9999",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.9999,
            #         learning_rate=0.5,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ducb10=(
            #     "TD Discounted UCB 0.5 0.9",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=0.9,
            #         learning_rate=0.5,
            #         rng=self._rng,
            #     ),
            # ),
            # td_ucb=(
            #     "TD UCB 0.1",
            #     functools.partial(
            #         td_mabs.discounted_ucb.DiscountedUCB,
            #         n_arms=self._num_actions,
            #         rho=1.0,
            #         gamma=1,
            #         learning_rate=0.1,
            #         rng=self._rng,
            #     ),
            # ),
            # tddd2=(
            #     "TD DoyaDaYu 1.25 No Noise",
            #     functools.partial(
            #         dd_mabs.td_doya_dayu.DoyaDaYu,
            #         self._num_actions,
            #         20,
            #         learning_rate=None,
            #         adapt_temperature=True,
            #         rng=self._rng,
            #         mask_p=1.0,
            #         Q0=0.5,
            #         S0=0.01,
            #         lrate_per_arm=True,
            #         lr_noise_multiplier=None,
            #         use_direct=False,
            #         aleatoric="variance",
            #     ),
            # ),
            # ddtab3=(
            #     "DoyaDaYu",
            #     functools.partial(
            #         dd_mabs.doya_dayu.DoyaDaYu,
            #         self._num_actions,
            #         20,
            #         learning_rate=None,
            #         adapt_temperature=True,
            #         rng=self._rng,
            #         mask_p=1,
            #         Q0=0.5,
            #         S0=0.01,
            #         lrate_per_arm=True,
            #         lr_noise_multiplier=1.25,
            #         use_direct=False,
            #         aleatoric="variance",
            #     ),
            # ),
            # ddtab4=(
            #     "DoyaDaYu 1",
            #     functools.partial(
            #         dd_mabs.doya_dayu.DoyaDaYu,
            #         self._num_actions,
            #         20,
            #         learning_rate=None,
            #         adapt_temperature=True,
            #         rng=self._rng,
            #         mask_p=1,
            #         Q0=0.5,
            #         S0=0.01,
            #         lrate_per_arm=True,
            #         lr_noise_multiplier=2,
            #         use_direct=False,
            #         aleatoric="variance",
            #     ),
            # ),
            # ddtab4=(
            #     # "DoyaDaYu (Tab)",
            #     "DoyaDaYu 2.5",
            #     functools.partial(
            #         doya_dayu_tab.DoyaDaYuTabular,
            #         self._num_actions,
            #         30,
            #         adapt_lrate=True,
            #         adapt_temperature=True,
            #         rng=self._rng,
            #         mask_p=0.5,
            #         Q0=0.5,
            #         S0=0.01,
            #         lrate_per_arm=True,
            #         lr_noise_multiplier=2.5,
            #     ),
            # ),
        )

    @property
    def agents(self):
        return self._agents
