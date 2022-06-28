import tensorflow as tf
from typing import List
from src.data import prepare
from src.data import preprocessing


class DatasetManager:
    """Simple class to handle access to datasets."""

    @staticmethod
    def save_data(dataset: tf.data.Dataset, data_dir: str = ''):
        """Save tf.data.dataset to directory path/.

        args:
            dataset: The dataset to be saved.
            data_dir: Directory to save data into.

        return:
            Nothing.
        """
        tf.data.experimental.save(dataset, data_dir)

    @staticmethod
    def load_data(data_dir: str = '') -> tf.data.Dataset:
        """Load data from directory.

        args:
            data_dir: Directory where data is stored.

        return:
            The category's dataset.
        """
        return tf.data.experimental.load(data_dir)


def check_config(config: dict, needed_keys: List[tuple or str]) -> None:
    """Test for checking that the operator get all the configuration it needs to run properly.

    args:
        config: The loaded configuration.
        needed_keys: A list of the needed arguments for the operator represented as their path into the
            configuration file (max path of length 2).
    return:
        None.
    raises:
        AssertionError if the configuration file doesn't meet operator's requirements.
    """
    assert config is not None, 'Error while loading the configuration'
    for nk in needed_keys:
        if isinstance(nk, tuple):
            assert nk[1] in config[nk[0]].keys(), f'Missing parameter {nk} in configuration file'
            continue
        assert nk in config.keys(), f'Missing parameter {nk} in configuration file'


def check_data_formatter(data_formatter: str) -> None:
    """Test if module prepare has the data_formatter requested for the preparation operation

    args:
        data_formatter: The formatter used in the preparation.
    return:
        None.
    raises:
        AssertionError if the formatter doesn't exist.
    """
    assert hasattr(prepare, data_formatter), f"The formatter '{data_formatter}' does not exist"


def check_transformations(transformations):
    """Test if module preprocessing has the transformations requested for the preprocessing operation

    args:
        transformations: The list of transformations to be used in preprocessing operation.
    return:
        None.
    raises:
        AssertionError if the transformation doesn't exist.
    """
    for transform in transformations:
        assert hasattr(preprocessing, transform['name']), f"The transformation '{transform['name']}' does not exist"
