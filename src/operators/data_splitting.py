import os
import tensorflow as tf
from pathlib import Path
from src.operator import Operator
from src.parser import OpParser

from src.utils import DatasetManager, check_config


class DataSplitting(Operator):
    needed_params = [
        'train_seed',
        'valid_seed'
    ]

    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config = self.load_operator_config(config_path, 'data_splitting')
        check_config(self.config, self.needed_params)

        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        print('splitting data ...')

        train_seed = self.config['train_seed']
        valid_seed = self.config['valid_seed']
        kwargs = {k: v for k, v in self.config.items() if 'seed' not in k and 'image' not in k}

        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            subset="training",
            directory=os.path.join(self.input_dir, "audio-images"),
            seed=train_seed,
            **kwargs
        )

        valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            subset="validation",
            directory=os.path.join(self.input_dir, "audio-images"),
            seed=valid_seed,
            **kwargs
        )

        # save data
        DatasetManager.save_data(train_dataset, data_dir=os.path.join(self.output_dir, 'train'))
        DatasetManager.save_data(valid_dataset, data_dir=os.path.join(self.output_dir, 'valid'))


if __name__ == '__main__':
    op_parser = OpParser()
    args = op_parser.parse_args()

    splitting = DataSplitting(args.config_path, args.input_dir, args.output_dir)
    splitting.run()
