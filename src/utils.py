import tensorflow as tf
import json
from datetime import datetime
import os
from typing import List, Tuple
import yaml
from pathlib import Path


class DataManager:
    """Simple class to handle access to datasets."""

    @staticmethod
    def save_data_category(dataset: tf.data.Dataset, category: str, path: Path = ''):
        """Save tf.data.dataset to directory path/category/.

        args:
            dataset: The dataset to be saved.
            category: The type of dataset (e.g 'valid' or 'train').
            path: Directory to the different category folders.

        return:
            Nothing.
        """
        category_path = os.path.join(path, category)
        tf.data.experimental.save(dataset, category_path)

    @staticmethod
    def load_data_category(category: str, data_dir: Path = '') -> tf.data.Dataset:
        """Load split data from directory.

        args:
            data_dir: Directory where split data is stored.
            category: The type of dataset (e.g 'valid' or 'train').

        return:
            The category's dataset.
        """
        return tf.data.experimental.load(os.path.join(data_dir, category))

# class deprecated
class ConfigManager:

    ## not used
    @staticmethod
    def load_config(path: str) -> dict:
        """Read .yaml config files.

        args:
            path: The path to the configuration file.
        return:
            The configuration file as a dict.
        """
        with open(path, 'r') as fp:
            config = yaml.safe_load(fp)
        return config

    # deprecated
    @staticmethod
    def format_config(l_layers: List[tf.keras.layers.Layer]) -> dict:
        """Format the configuration of a list of layer so it can be reused as an argument to rebuild them.

        args:
            l_layers: A list of layers to get the configuration from.

        return:
            The configuration of the list of layers formatted in a dictionary.
        """
        return {f'cf_{sub_layer.name}': sub_layer.get_config() for sub_layer in l_layers}

    # deprecated
    @staticmethod
    def save_model_config(cfg: dict, save_dir: str) -> str:
        """save the formatted configuration of a model into a json file named '<timestamp>_cfg.json'.

        args:
            cfg: The configuration dict of the model to be saved.
            save_dir: The directory to the configuration storage.

        return:
            The timestamp of the saved configuration.
        """
        os.makedirs(save_dir, exist_ok=True)
        cfg_tmp = datetime.today().strftime('%Y%m%d%H%M%S%f')
        path = os.path.join(save_dir, cfg_tmp + '_cfg.json')
        with open(path, 'w') as fp:
            json.dump(cfg, fp, indent=4)
        return cfg_tmp

    # deprecated
    def update_config(self, config: dict, params: dict) -> None:
        """Update configuration with new value for some parameters.

        args:
            config: The base configuration dict to change parameter values from.
            params: The new parameters values to set in the base configuration.

        return:
            Nothing.
        """
        for k, v in params.items():
            if isinstance(v, dict):
                self.update_config(config[k], params[k])
            else:
                config.update({k: v})
