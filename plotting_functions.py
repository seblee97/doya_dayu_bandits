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
        means = regret[i].mean(0).ravel().cumsum(-1)
        # stds = regret[i].std(0).ravel()
        h = plt.plot(means, lw=2, label=label)
        # plt.fill_between(range(len(stds)), means - stds, means + stds, alpha=0.5)

        plt.legend(loc="best")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative regret")

    fig.savefig(os.path.join(save_path, "cumulative_regret.pdf"))
