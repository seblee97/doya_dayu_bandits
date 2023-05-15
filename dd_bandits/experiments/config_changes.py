import itertools

import numpy as np

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
for ens in [20]:
    for offset in [0.0]:
        for m in [0.1]:
            for lr in [0.1]:
                for sc in [1, 1.5, 2]:
                    for typ in ["full_oracle", "oracle", "ratio"]:
                        for q_init in [0.01]:
                            for s_init in [0.1]:
                                for mask in [0.5]:
                                    for ud in [False]:
                                        dd_agents.append(
                                            {
                                                "name": f"dd_{ens}_{offset}_{m}_{lr}_{q_init}_{sc}_{s_init}_{mask}_{typ}",
                                                "agent": "td_doya_dayu",
                                                "n_ens": ens,
                                                "adaptation_modules": {
                                                    "constant": [
                                                        {"type": "constant"},
                                                        {"value": 0.25},
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
                                                    "learning_rate_operation": typ,
                                                    "learning_rate_operands": [
                                                        "reliability_index_frac",
                                                        "var_mean",
                                                    ],
                                                },
                                                "temperature": {
                                                    "temperature_operation": "scaled",
                                                    "temperature_operands": [
                                                        "reliability_index_frac",
                                                        sc,
                                                    ],
                                                },
                                                "mask_p": mask,
                                                "q_initialisation": q_init,
                                                "offset": offset,
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
for t in [0.6]:
    # for t in np.linspace(0.1, 1, 10):
    # for lr in np.linspace(0.1, 1, 10):
    for lr in [0.2]:
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
for r in [0.8, 0.9, 0.99, 1.0]:
    for g in [0.9, 0.99, 0.999, 0.9999]:
        for lr in np.linspace(0.1, 1, 10):
            for t in [0.6]:
                for q_init in [0.01]:
                    ducb_agents.append(
                        {
                            "name": f"ducb_{r}_{g}_{lr}_{t}_{q_init}",
                            "agent": "td_ducb",
                            "rho": r,
                            "gamma": g,
                            "learning_rate": lr,
                            "temperature": t,
                            "q_initialisation": q_init,
                            "scalar_log_spec": [
                                "mean_mean",
                                "mean_var",
                                "var_mean",
                                "var_var",
                            ],
                        }
                    )

# agents = constant_agents + dd_agents + boltzmann_agents + ducb_agents
# agents = ducb_agents + dd_agents + ddt_agents + ddlr_agents + boltzmann_agents
agents = ducb_agents
# CONFIG_CHANGES = {"agent_ablation": [{"agents": agents}]}
