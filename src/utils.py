import tensorflow as tf
from typing import List
import yaml
import mlflow
import os

from src.data import prepare
from src.data import preprocessing
from src.model.model import CNNModel


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


def get_base_config() -> dict:
    """Load the base config of the project in 'configs/base_cfg.yaml' and return the 'base'.
    """
    base_config_path = os.path.join('configs', 'base_cfg.yaml')
    assert os.path.exists(base_config_path), f"path to base config '{base_config_path}' does not exists"
    with open(base_config_path, 'r') as fp:
        base_config = yaml.safe_load(fp)
    return base_config['base']


def get_info_path() -> str:
    """Get the last run info path.
    """
    base_cfg = get_base_config()
    return base_cfg['mlflow_last_run_info']


def get_params_config() -> dict:
    """Load dvc's parameter file 'params.yaml'.
    """
    assert os.path.exists('params.yaml'), f"'params.yaml' does not exists"
    with open('params.yaml', 'r') as fp_p:
        params_config = yaml.safe_load(fp_p)
    return params_config


def get_path_to_last_model() -> str:
    """Get the saving path of the last model run with mlflow.
    """
    info_path = get_info_path()

    assert os.path.exists(info_path), f"can't gather information from last run, {info_path} does not exist"
    with open(info_path, 'r') as fp:
        last_run = yaml.safe_load(fp)
    return os.path.join(last_run['artifact_uri'], 'model')


def mlflow_keras_load_model() -> CNNModel:
    """Load the last model from the default mlflow model registry.

    return:
        The last mlflow saved model.
    """
    path_to_model = get_path_to_last_model()
    assert os.path.exists(path_to_model), f"the model you looking for does not exit in local directory {path_to_model}"

    return mlflow.keras.load_model(path_to_model)


def save_run_info(run_id: str, exp_id: str, artifact_uri: str) -> None:
    """Update 'mlflow_last_run_info' configuration file with experiment id, training run id and artifact uri for
    upcoming evaluations.
    """
    run_info_path = get_info_path()

    info = {
        'run_id': run_id,
        'experiment_id': exp_id,
        'artifact_uri': artifact_uri
    }
    with open(run_info_path, 'w') as fp:
        yaml.safe_dump(info, fp)


def save_evaluation_results(outputs: list, names: list) -> None:
    """Save metrics of model's evaluation from last run into artifact repository.
    """
    run_info = get_last_run_info()

    with open(f'{run_info["artifact_uri"]}/evaluation.txt', 'w') as fp:
        text = []
        for v, n in zip(outputs, names):
            line = '{}: {:.6f}\n'.format(n, v)
            print(line)
            text.append(line)
        fp.writelines(text)


def get_last_run_info() -> dict:
    """Load last run info.
    """
    run_info_path = get_info_path()
    with open(run_info_path, 'r') as fp:
        run_info = yaml.safe_load(fp)

    return run_info


### Checks ###

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
    """Test if prepare module has the data_formatter requested for the preparation operation

    args:
        data_formatter: The formatter used in the preparation.
    return:
        None.
    raises:
        AssertionError if the formatter doesn't exist.
    """
    assert hasattr(prepare, data_formatter), f"The formatter '{data_formatter}' does not exist"


def check_transformations(transformations):
    """Test if preprocessing module has the transformations requested for the preprocessing operation

    args:
        transformations: The list of transformations to be used in preprocessing operation.
    return:
        None.
    raises:
        AssertionError if the transformation doesn't exist.
    """
    for transform in transformations:
        assert hasattr(preprocessing, transform['name']), f"The transformation '{transform['name']}' does not exist"
