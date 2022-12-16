import argparse
import datetime
import os
import time

import numpy as np
import scipy

import plotting_functions
from agents import Agents

parser = argparse.ArgumentParser()

MASTER_SEED = 321
rng = np.random.RandomState(MASTER_SEED)

parser.add_argument(
    "--change_frequency",
    type=int,
    default=200,
    help="number of steps before bandits change.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=5,
    help="number of bandit contexts in experiment.",
)
parser.add_argument(
    "--num_actions",
    type=int,
    default=10,
    help="number of arms in bandit.",
)
parser.add_argument(
    "--num_seeds",
    type=int,
    default=80,
    help="number of different experiment repeats.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="learning rate.",
)


def _sample_mean():
    return rng.uniform(-3, 3)


def _sample_scale():
    return rng.uniform(1e-3, 2)


if __name__ == "__main__":

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    exp_path = os.path.join("results", exp_timestamp)
    os.makedirs(exp_path, exist_ok=True)

    args = parser.parse_args()

    agent_class = Agents(rng, args.num_actions, args.learning_rate)
    experiment_agents = agent_class.agents

    scalar_data_shape = (
        len(experiment_agents),
        args.num_seeds,
        args.num_episodes,
        args.change_frequency,
    )

    agent_order = sorted(experiment_agents)

    regret = np.zeros(scalar_data_shape)
    correct_arm = np.zeros(scalar_data_shape)
    learning_rate = np.zeros(scalar_data_shape)
    policy = np.zeros(scalar_data_shape + (args.num_actions,))
    moment_error = np.zeros(scalar_data_shape + (2,))
    dist_hist = np.zeros((args.num_seeds, args.num_episodes, args.num_actions, 2))

    for seed in range(args.num_seeds):
        for episode in range(args.num_episodes):
            dist_hist[seed, episode] = np.array(
                [(_sample_mean(), _sample_scale()) for _ in range(args.num_actions)]
            )

    for seed in range(args.num_seeds):
        for episode in range(args.num_episodes):
            dists = [
                scipy.stats.norm(
                    dist_hist[seed, episode, a, 0], scale=dist_hist[seed, episode, a, 1]
                )
                for a in range(args.num_actions)
            ]
            best_arm = np.array([d.mean() for d in dists]).argmax()

            for i, name in enumerate(agent_order):
                agent = experiment_agents[name][1]()

                for trial in range(args.change_frequency):
                    action = agent.play()
                    regret[i, seed, episode, trial] = (
                        dists[best_arm].mean() - dists[action].mean()
                    )
                    correct_arm[i, seed, episode, trial] = best_arm == action

                    learning_rate[i, seed, episode, trial] = agent.learning_rate(action)
                    policy[i, seed, episode, trial] = agent.policy()

                    agent.update(action, dists[action].rvs(random_state=rng))
                    means, vars = agent.predict_bandits()
                    moment_error[i, seed, episode, trial, 0] = (
                        dist_hist[seed, episode, best_arm, 0] - means[best_arm]
                    ) ** 2
                    moment_error[i, seed, episode, trial, 1] = (
                        dist_hist[seed, episode, best_arm, 1] ** 2 - vars[best_arm]
                    ) ** 2

    np.save(file=os.path.join(exp_path, "regret"), arr=regret)
    np.save(file=os.path.join(exp_path, "correct_arm"), arr=correct_arm)

    plotting_functions.average_best_arm_plot(
        agents=experiment_agents,
        agent_order=agent_order,
        correct_arm=correct_arm,
        change_freq=args.change_frequency,
        save_path=exp_path,
    )
    plotting_functions.average_regret_plot(
        agents=experiment_agents,
        agent_order=agent_order,
        regret=regret,
        change_freq=args.change_frequency,
        save_path=exp_path,
    )
    plotting_functions.cumulative_regret(
        agents=experiment_agents,
        agent_order=agent_order,
        regret=regret,
        save_path=exp_path,
    )
