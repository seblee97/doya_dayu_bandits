import itertools

import numpy as np


def kl_div(mu_1, mu_2, sigma_1, sigma_2):
    return (
        np.log(sigma_2 / sigma_1)
        + (sigma_1**2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2**2)
        - 0.5
    )


def compute_information_radius(means, stds):
    average_mean = np.mean(means)
    average_var = np.mean(np.array(stds) ** 2)

    kls = []

    for (mean, std) in zip(means, stds):
        kls.append(kl_div(mean, average_mean, std, np.sqrt(average_var)))

    return np.mean(kls)


def compute_pairwise_kl(means, stds):
    kls = []

    dists = list(zip(means, stds))

    for ((mu_1, sigma_1), (mu_2, sigma_2)) in itertools.product(dists, dists):
        kl_12 = kl_div(mu_1, mu_2, sigma_1, sigma_2)
        kls.append(kl_12)

    mean_kls = np.mean(kls)
    max_kls = np.max(kls)

    return mean_kls, max_kls


def gaussian_likelihood(mean, std, x):
    lik = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    if np.isnan(lik):
        import pdb

        pdb.set_trace()
    return lik
