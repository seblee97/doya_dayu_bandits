import itertools

import numpy as n

NUM_ARMS = 5

# doya-dayu agents
dd_agents = []
for ens in []:
    for m in []:
        for lr in []:
            for q_init in []:
                for s_init in []:
                    for mask in []:
                        for ud in [True, False]:
                            dd_agents.append(
                                {
                                    "name": f"dd_{ens}_{m}_{q_init}_{s_init}_{mask}",
                                    "agent": "td_doya_dayu",
                                    "n_ens": ens,
                                    "adaptation_modules": {
                                        "reliability_index": {
                                            "type": "reliability_index",
                                            "learning_rate": lr,
                                            "num_arms": NUM_ARMS,
                                        },
                                        "reliability_index_frac": {
                                            "type": "reliability_index",
                                            "multiple": m,
                                            "learning_rate": lr,
                                            "num_arms": NUM_ARMS,
                                        },
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
for t in []:
    for lr in []:
        for q_init in []:
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
for r in []:
    for g in []:
        for lr in []:
            for q_init in []:
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
