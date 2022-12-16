import functools

import optax

import boltzmann
import discounted_ucb
import doya_dayu
import doya_dayu_tab
import epsilon_greedy
import thompson_ensemble
import thompson_gaussian
import ucb


class Agents:
    def __init__(self, rng, num_actions: int, learning_rate: float) -> None:
        self._rng = rng
        self._num_actions = num_actions
        self._learning_rate = learning_rate

        self._agents = self._setup_agents()

    def _setup_agents(self):
        return dict(
            # egreedy=(
            #     "Epsilon Greedy",
            #     functools.partial(
            #         epsilon_greedy.EpsilonGreedy,
            #         self._num_actions,
            #         0.05,
            #         optax.sgd(self._learning_rate),
            #         rng=self._rng,
            #     ),
            # ),
            # boltzmann=(
            #     "Boltzmann",
            #     functools.partial(
            #         boltzmann.Boltzmann,
            #         self._num_actions,
            #         0.25,
            #         optax.sgd(self._learning_rate),
            #         rng=self._rng,
            #     ),
            # ),
            # ucb=(
            #     "UCB",
            #     functools.partial(
            #         ucb.UCB,
            #         self._num_actions,
            #         1.0,
            #         optax.sgd(self._learning_rate),
            #         rng=self._rng,
            #     ),
            # ),
            ducb=(
                "Discounted UCB",
                functools.partial(
                    discounted_ucb.DiscountedUCB,
                    self._num_actions,
                    1.0,
                    0.99,
                    rng=self._rng,
                ),
            ),
            # dd=(
            #     "DoyaDaYu",
            #     functools.partial(
            #         doya_dayu.DoyaDaYu,
            #         self._num_actions,
            #         30,
            #         optax.sgd(1.0),
            #         use_direct=True,
            #         adapt_lrate=True,
            #         adapt_temperature=True,
            #         Q0=0.1,
            #         S0=0.001,
            #         rng=self._rng,
            #     ),
            # ),
            ddtab=(
                # "DoyaDaYu (Tab)",
                "DoyaDaYu",
                functools.partial(
                    doya_dayu_tab.DoyaDaYuTabular,
                    self._num_actions,
                    30,
                    adapt_lrate=True,
                    adapt_temperature=True,
                    rng=self._rng,
                    mask_p=0.5,
                    Q0=0.5,
                    S0=0.01,
                    lrate_per_arm=True,
                ),
            ),
            ddtab2=(
                # "DoyaDaYu (Tab)",
                "DoyaDaYu2",
                functools.partial(
                    doya_dayu_tab.DoyaDaYuTabular,
                    self._num_actions,
                    30,
                    adapt_lrate=True,
                    adapt_temperature=True,
                    rng=self._rng,
                    mask_p=1,
                    Q0=0.5,
                    S0=0.01,
                    lrate_per_arm=True,
                ),
            ),
        )

    @property
    def agents(self):
        return self._agents
