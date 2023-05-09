import abc
from typing import Any


class BaseAdaptation(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
