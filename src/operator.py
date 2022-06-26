import yaml
from pathlib import Path
from abc import abstractmethod, ABC
from abc import ABC


class AbstractOperator(ABC):
    @staticmethod
    @abstractmethod
    def load_operator_config(path: Path, name: str) -> dict:
        pass

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()


class Operator(AbstractOperator, ABC):

    @staticmethod
    def load_operator_config(path: Path, name: str) -> dict:
        """Read .yaml config files.

        args:
            path: The path to the configuration file.
            name: Part's name of the config dedicated to the operator.

        return:
            The configuration file as a dict or list of dicts.
        """
        with open(path, 'r') as fp:
            config = yaml.safe_load(fp)
        return config[name]

# todo revise that !!
#     @abstractmethod
#     def run(self) -> None:
#         raise NotImplementedError()
