import itertools

import numpy as n

NUM_ARMS = 5

constant_agents = []
for lr in []:
    for t in []:
        for ud in [True, False]:
            constant_agents.append(
                {
                    "name": f"constant_{t}_{lr}",
                    "agent": "td_doya_dayu",
                    "n_ens": 10,
                    "adaptation_modules": {
                        "constant": [{"type": "constant"}, {"value": lr}],
                        "constant_t": [{"type": "constant"}, {"value": t}],
                    },
                    "learning_rate": "constant",
                    "temperature": "constant_t",
                    "mask_p": 1.0,
                    "q_initialisation": 0.01,
                    "s_initialisation": 0.01,
                    "lr_per_arm": True,
                    "scalar_log_spec": ["mean_mean", "mean_var", "var_mean", "var_var"],
                    "use_direct": ud,
                }
            )

# doya-dayu agents
dd_agents = []
for ens in [5, 10, 20]:
    for m in [0.1, 0.5, 1, 2]:
        for lr in [0.1, 0.25, 1.0]:
            for q_init in [0.01, 0.0, 0.1]:
                for s_init in [0.01, 0.1]:
                    for mask in [1.0, 0.25, 0.5, 0.75]:
                        for ud in [True, False]:
                            dd_agents.append(
                                {
                                    "name": f"dd_{ens}_{m}_{q_init}_{s_init}_{mask}",
                                    "agent": "td_doya_dayu",
                                    "n_ens": ens,
                                    "adaptation_modules": {
                                        "constant": [
                                            {"type": "constant"},
                                            {"value": 0.1},
                                        ],
                                        "reliability_index": [
                                            {"type": "reliability_index"},
                                            {"learning_rate": lr},
                                            {"num_arms": NUM_ARMS},
                                        ],
                                        "reliability_index_frac": [
                                            {"type": "reliability_index"},
                                            {"multiple": m},
                                            {"learning_rate": lr},
                                            {"num_arms": NUM_ARMS},
                                        ],
                                    },
                                    "learning_rate": {
                                        "learning_rate_operation": "ratio",
                                        "learning_rate_operands": [
                                            "reliability_index_frac",
                                            "var_mean",
                                        ],
                                    },
                                    "temperature": "reliability_index_frac",
                                    "mask_p": mask,
                                    "q_initialisation": q_init,
                                    "s_initialisation": s_init,
                                    "lr_per_arm": True,
                                    "scalar_log_spec": [
                                        "mean_mean",
                                        "mean_var",
                                        "var_mean",
                                        "var_var",
                                        "reliability_index",
                                        "reliability_index_frac",
                                        "likelihood_shift",
                                    ],
                                    "use_direct": ud,
                                }
                            )

# boltzmann agents
boltzmann_agents = []
for t in [0.01, 0.05, 0.1, 0.25]:
    for lr in [0.1, 0.25, 0.5]:
        for q_init in [0.01]:
            boltzmann_agents.append(
                {
                    "name": f"boltzmann_{t}_{lr}_{q_init}",
                    "agent": "td_boltzmann",
                    "temperature": t,
                    "learning_rate": lr,
                    "q_initialisation": q_init,
                    "scalar_log_spec": ["mean_mean", "mean_var", "var_mean", "var_var"],
                }
            )

# ducb agents
ducb_agents = []
for r in [1.0]:
    for g in [0.99, 0.999, 0.9999]:
        for lr in [0.01, 0.05, 0.2]:
            for q_init in [0.01]:
                ducb_agents.append(
                    {
                        "name": f"ducb_{r}_{g}_{lr}_{q_init}",
                        "agent": "td_ducb",
                        "rho": r,
                        "gamma": g,
                        "learning_rate": lr,
                        "q_initialisation": q_init,
                        "scalar_log_spec": [
                            "mean_mean",
                            "mean_var",
                            "var_mean",
                            "var_var",
                        ],
                    }
                )

agents = constant_agents  # + dd_agents + boltzmann_agents + ducb_agents

# CONFIG_CHANGES = {"agent_ablation": [{"agents": agents}]}
