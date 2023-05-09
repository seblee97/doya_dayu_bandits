from config_manager import config_field, config_template

from dd_bandits import constants


class ConfigTemplate:

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.CHANGE_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.CHANGE_PROBABILITY,
                types=[float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.NUM_EPISODES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.NUM_SEEDS,
                types=[int],
            ),
            config_field.Field(
                name=constants.DEFAULT_LEARNING_RATE,
                types=[float],
                requirements=[lambda x: x > 0.0],
            ),
        ],
        level=[constants.TRAINING],
    )

    _bandit_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_ARMS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.MEAN_LOWER,
                types=[float],
            ),
            config_field.Field(
                name=constants.MEAN_UPPER,
                types=[float],
            ),
            config_field.Field(
                name=constants.SCALE_LOWER,
                types=[float],
                requirements=[lambda x: x > 0.0],
            ),
            config_field.Field(
                name=constants.SCALE_UPPER,
                types=[float],
                requirements=[lambda x: x > 0.0],
            ),
            config_field.Field(name=constants.BERNOULLI, types=[bool]),
        ],
        level=[constants.BANDIT],
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.SEED, types=[int]),
            config_field.Field(
                name=constants.AGENTS,
                types=[list],
                requirements=[
                    lambda x: isinstance(x[0], dict),
                    lambda x: constants.AGENT in x[0] and constants.NAME in x[0],
                    lambda x: x[0][constants.AGENT]
                    in [
                        constants.TD_DOYA_DAYU,
                        constants.TD_BOLTZMANN,
                        constants.TD_DUCB,
                        constants.DUCB,
                        constants.BOLTZMANN,
                    ],
                ],
            ),
        ],
        nested_templates=[
            _training_template,
            _bandit_template,
        ],
    )
