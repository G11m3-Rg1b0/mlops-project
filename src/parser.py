import argparse
from pathlib import Path


class PathAction(argparse.Action):
    """An action class to handle path arguments in different os."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, Path(values))


class PipeParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('--config', type=str, help="config's path for pipeline", action=PathAction)
        self.add_argument('--input-dir', type=str, help="pipeline's input data directory", action=PathAction)
        self.add_argument('--output-dir', type=str, help="pipeline's output data directory", action=PathAction)
        self.add_argument('--model-dir', type=str, help="model directory", action=PathAction)
