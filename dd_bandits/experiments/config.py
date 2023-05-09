from typing import Dict, List, Union

from config_manager import base_configuration

from dd_bandits import constants
from dd_bandits.experiments.config_template import ConfigTemplate


class Config(base_configuration.BaseConfiguration):
    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        super().__init__(
            configuration=config,
            template=ConfigTemplate.base_config_template,
            changes=changes,
        )
        self._validate_configuration()

    def _validate_configuration(self):
        """Method to check for non-trivial associations
        in the configuration.
        """
        pass
