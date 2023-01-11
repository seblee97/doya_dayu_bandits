import os

import matplotlib.pyplot as plt
import numpy as np


def average_best_arm_plot(agents, agent_order, correct_arm, change_freq, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        label = agents[name][0]
        plt.plot(
            (correct_arm[i].mean((0, 1)).cumsum(-1) / np.arange(1, change_freq + 1)),
            lw=4,
            label=label,
        )

    plt.legend(loc="best")

    plt.xlabel("Steps (per context)")
    plt.ylabel("Average best arm (over context switches)")

    fig.savefig(os.path.join(save_path, "average_best_arm.pdf"))


def average_regret_plot(agents, agent_order, regret, change_freq, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        label = agents[name][0]
        plt.plot(
            regret[i].mean((0, 1)).cumsum(-1) / np.arange(1, change_freq + 1),
            lw=4,
            label=label,
        )

    plt.legend(loc="best")

    plt.xlabel("Steps (per context)")
    plt.ylabel("Average regret (over context switches)")

    fig.savefig(os.path.join(save_path, "average_regret.pdf"))


def cumulative_regret(agents, agent_order, regret, save_path):

    fig = plt.figure()

    for i, name in enumerate(agent_order):
        label = agents[name][0]
        flattened_regrets = np.reshape(regret[i], (regret[i].shape[0], -1))
        cum_regret = np.cumsum(flattened_regrets, axis=1)
        means = cum_regret.mean(0)
        stds = cum_regret.std(0)
        # means = regret[i].mean(0).ravel().cumsum(-1)
        # stds = regret[i].std(0).ravel()
        plt.plot(means, lw=2, label=label)
        plt.fill_between(range(len(stds)), means - stds, means + stds, alpha=0.25)

    plt.legend(loc="best")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative regret")

    fig.savefig(os.path.join(save_path, "cumulative_regret.pdf"))


def learning_rates(agents, agent_order, learning_rates, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        plt.plot(learning_rates[i].mean(0).ravel(), label=agents[name][0])
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, "learning_rates.pdf"))


def temperatures(agents, agent_order, temperatures, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        plt.plot(temperatures[i].mean(0).ravel(), label=agents[name][0])
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, "temperatures.pdf"))


def min_uncertainties(agents, agent_order, min_uncertainties, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        plt.plot(min_uncertainties[i].mean(0).ravel(), label=agents[name][0])
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Minimum Epistemic Uncertainty")
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, "min_uncertainties.pdf"))


def policies(agents, agent_order, policies, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        plt.plot(policies[i].max(-1).mean(0).ravel(), label=agents[name][0])
    plt.xlabel("Steps")
    plt.ylabel("Policy")
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, "policies.pdf"))


def mean_mses(agents, agent_order, mses, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        plt.plot(mses[i, ..., 0].mean(0).ravel(), label=agents[name][0])
    plt.xlabel("Steps")
    plt.ylabel("Mean MSEs")
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, "mean_mses.pdf"))


def variance_mses(agents, agent_order, mses, save_path):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        plt.plot(mses[i, ..., 1].mean(0).ravel(), label=agents[name][0])
    plt.xlabel("Steps")
    plt.ylabel("Variance MSEs")
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, "variance_mses.pdf"))


def scalar_plot(agents, agent_order, scalar_data, save_path, label, logscale):
    fig = plt.figure()
    for i, name in enumerate(agent_order):
        means = scalar_data[i].mean(0).ravel()

        non_inf_indices = np.where(means < np.inf)
        if len(non_inf_indices[0]):
            non_inf_index = non_inf_indices[0][0] + 1
        else:
            non_inf_index = len(means)

        means = means[non_inf_index:]
        stds = scalar_data[i].std(0).ravel()[non_inf_index:]
        # means = regret[i].mean(0).ravel().cumsum(-1)
        # stds = regret[i].std(0).ravel()
        plt.plot(
            np.arange(non_inf_index, len(means) + non_inf_index),
            means,
            lw=2,
            label=agents[name][0],
        )
        plt.fill_between(
            np.arange(non_inf_index, len(means) + non_inf_index),
            means - stds,
            means + stds,
            alpha=0.25,
        )
    if logscale:
        plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel(label)
    plt.legend(loc="best")
    fig.savefig(save_path)


# def distributions(distributions, change_frequency, save_path):
#     fig, axes = plt.subplots(1, 2)

#     import pdb

#     pdb.set_trace()

#     axes[0].plot(np.repeat(distributions, change_frequency, axis=0)[:, 0], lw=4)
#     axes[0].set_title("Mean")

#     axes[1].plot(np.repeat(distributions, change_frequency, axis=0)[:, 1], lw=4)
#     axes[1].set_title("Standard Deviation")

#     fig.savefig(os.path.join(save_path, "distributions.pdf"))
