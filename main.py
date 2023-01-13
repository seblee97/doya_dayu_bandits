import argparse
import datetime
import json
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
    default=400,
    help="number of steps before bandits change.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=8,
    help="number of bandit contexts in experiment.",
)
parser.add_argument(
    "--num_actions",
    type=int,
    default=8,
    help="number of arms in bandit.",
)
parser.add_argument(
    "--num_seeds",
    type=int,
    default=1,
    help="number of different experiment repeats.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="learning rate.",
)
parser.add_argument("--mean_lower", default=-3, help="lower bound of sample for means")
parser.add_argument("--mean_upper", default=3, help="upper bound of sample for means")
parser.add_argument(
    "--scale_lower", default=1e-3, help="lower bound of sample for scale"
)
parser.add_argument("--scale_upper", default=1, help="upper bound of sample for scale")
parser.add_argument("--bernoulli", action="store_true", default=False)


def _sample_mean(lower, upper):
    return rng.uniform(lower, upper)


def _sample_scale(lower, upper):
    return rng.uniform(lower, upper)


def _sample_probability():
    return rng.uniform(0, 1)


if __name__ == "__main__":

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    exp_path = os.path.join("results", exp_timestamp)
    os.makedirs(exp_path, exist_ok=True)

    plot_path = os.path.join(exp_path, "plots")
    array_path = os.path.join(exp_path, "arrays")
    os.makedirs(plot_path)
    os.makedirs(array_path)

    args = parser.parse_args()

    agent_class = Agents(rng, args.num_actions, args.learning_rate)
    experiment_agents = agent_class.agents

    experiment_agents_spec = {
        name: (
            agent_spec[0],
            {k: v for k, v in agent_spec[1].keywords.items() if k != "rng"},
        )
        for name, agent_spec in experiment_agents.items()
    }

    with open(os.path.join(exp_path, "agents.json"), "+w") as json_file:
        json.dump(experiment_agents_spec, json_file)

    with open(os.path.join(exp_path, "args.json"), "+w") as json_file:
        json.dump(vars(args), json_file)

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
    temperature = np.zeros(scalar_data_shape)
    epistemic_uncertainty = np.zeros(scalar_data_shape)
    aleatoric_uncertainty = np.zeros(scalar_data_shape)
    min_uncertainty = np.zeros(scalar_data_shape)
    policy = np.zeros(scalar_data_shape + (args.num_actions,))
    moment_error = np.zeros(scalar_data_shape + (2,))

    if args.bernoulli:
        dist_hist = np.zeros((args.num_seeds, args.num_episodes, args.num_actions))
    else:
        dist_hist = np.zeros((args.num_seeds, args.num_episodes, args.num_actions, 2))

    for seed in range(args.num_seeds):
        for episode in range(args.num_episodes):
            if args.bernoulli:
                dist_hist[seed, episode] = np.array(
                    [(_sample_probability()) for _ in range(args.num_actions)]
                )
            else:
                dist_hist[seed, episode] = np.array(
                    [
                        (
                            _sample_mean(args.mean_lower, args.mean_upper),
                            _sample_scale(args.scale_lower, args.scale_upper),
                        )
                        for _ in range(args.num_actions)
                    ]
                )

    for seed in range(args.num_seeds):

        agent_instances = {
            name: agent[1]() for name, agent in experiment_agents.items()
        }

        for episode in range(args.num_episodes):
            if args.bernoulli:
                dists = [
                    scipy.stats.bernoulli(dist_hist[seed, episode, a])
                    for a in range(args.num_actions)
                ]
                best_arm = np.array([d.mean() for d in dists]).argmax()
            else:
                dists = [
                    scipy.stats.norm(
                        dist_hist[seed, episode, a, 0],
                        scale=dist_hist[seed, episode, a, 1],
                    )
                    for a in range(args.num_actions)
                ]
                best_arm = np.array([d.mean() for d in dists]).argmax()

            for i, name in enumerate(agent_order):
                # agent = experiment_agents[name][1]()
                agent = agent_instances[name]

                for trial in range(args.change_frequency):
                    action = agent.play()
                    regret[i, seed, episode, trial] = (
                        dists[best_arm].mean() - dists[action].mean()
                    )
                    correct_arm[i, seed, episode, trial] = best_arm == action

                    learning_rate[i, seed, episode, trial] = agent.learning_rate(action)
                    temperature[i, seed, episode, trial] = agent.temperature()
                    min_uncertainty[
                        i, seed, episode, trial
                    ] = agent.min_epistemic_uncertainty
                    aleatoric_uncertainty[
                        i, seed, episode, trial
                    ] = agent.aleatoric_uncertainty
                    epistemic_uncertainty[
                        i, seed, episode, trial
                    ] = agent.epistemic_uncertainty
                    policy[i, seed, episode, trial] = agent.policy()

                    agent.update(action, dists[action].rvs(random_state=rng))
                    means, vars = agent.predict_bandits()

                    if args.bernoulli:
                        moment_error[i, seed, episode, trial, 0] = (
                            dist_hist[seed, episode, best_arm] - means[best_arm]
                        ) ** 2
                    else:
                        moment_error[i, seed, episode, trial, 0] = (
                            dist_hist[seed, episode, best_arm, 0] - means[best_arm]
                        ) ** 2
                        moment_error[i, seed, episode, trial, 1] = (
                            dist_hist[seed, episode, best_arm, 1] ** 2 - vars[best_arm]
                        ) ** 2

    np.save(file=os.path.join(array_path, "regret"), arr=regret)
    np.save(file=os.path.join(array_path, "correct_arm"), arr=correct_arm)
    np.save(file=os.path.join(array_path, "learning_rate"), arr=learning_rate)
    np.save(file=os.path.join(array_path, "temperature"), arr=temperature)
    np.save(file=os.path.join(array_path, "min_uncertainty"), arr=min_uncertainty)
    np.save(
        file=os.path.join(array_path, "epistemic_uncertainty"),
        arr=epistemic_uncertainty,
    )
    np.save(
        file=os.path.join(array_path, "aleatoric_uncertainty"),
        arr=aleatoric_uncertainty,
    )
    np.save(file=os.path.join(array_path, "policy"), arr=policy)
    np.save(file=os.path.join(array_path, "moment_error"), arr=moment_error)
    np.save(file=os.path.join(array_path, "dist_hist"), arr=dist_hist)

    plotting_functions.average_best_arm_plot(
        agents=experiment_agents,
        agent_order=agent_order,
        correct_arm=correct_arm,
        change_freq=args.change_frequency,
        save_path=plot_path,
    )
    plotting_functions.average_regret_plot(
        agents=experiment_agents,
        agent_order=agent_order,
        regret=regret,
        change_freq=args.change_frequency,
        save_path=plot_path,
    )
    plotting_functions.cumulative_regret(
        agents=experiment_agents,
        agent_order=agent_order,
        regret=regret,
        save_path=plot_path,
    )
    plotting_functions.policies(
        agents=experiment_agents,
        agent_order=agent_order,
        policies=policy,
        save_path=plot_path,
    )
    plotting_functions.mean_mses(
        agents=experiment_agents,
        agent_order=agent_order,
        mses=moment_error,
        save_path=plot_path,
    )
    plotting_functions.variance_mses(
        agents=experiment_agents,
        agent_order=agent_order,
        mses=moment_error,
        save_path=plot_path,
    )
    for (scalar_label, scalar_data, plot_label, logscale) in [
        (
            "aleatoric_uncertainty",
            aleatoric_uncertainty,
            "Aleatoric Uncertainty",
            False,
        ),
        (
            "epistemic_uncertainty",
            epistemic_uncertainty,
            "Epistemic Uncertainty",
            False,
        ),
        ("min_uncertainty", min_uncertainty, "Minimum Epistemic Uncertainty", False),
        ("temperature", temperature, "Temperature", False),
        ("learning_rate", learning_rate, "Learning Rate", False),
    ]:
        plotting_functions.scalar_plot(
            agents=experiment_agents,
            agent_order=agent_order,
            scalar_data=scalar_data,
            label=plot_label,
            save_path=os.path.join(plot_path, f"{scalar_label}.pdf"),
            logscale=logscale,
        )
    # plotting_functions.distributions(
    #     distributions=dist_hist,
    #     change_frequency=args.change_frequency,
    #     save_path=plot_path,
    # )
