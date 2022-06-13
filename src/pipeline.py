import yaml
from pathlib import Path
from abc import abstractmethod


class Pipeline(object):

    @staticmethod
    def load_pipeline_config(path: Path, name: str) -> dict:
        """Read .yaml config files.

        args:
            path: The path to the configuration file.
            name: Part's name of the config dedicated to the pipeline.

        return:
            The configuration file as a dict or list of dicts.
        """
        with open(path, 'r') as fp:
            config = yaml.safe_load(fp)
        return config[name]

    @abstractmethod
    def run(self) -> None:
        pass
