import os
from pathlib import Path
from src.data.preprocessing import TransformFactory
from src.utils import DatasetManager, check_config, check_transformations
from src.operator import Operator
from src.parser import OpParser


class DataTrainPreprocessing(Operator):
    needed_params = [
        'transformations'
    ]

    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config = self.load_operator_config(config_path, 'data_train_preprocessing')
        check_config(self.config, self.needed_params)
        check_transformations(self.config['transformations'])

        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        print('preprocessing data ...')

        train_dataset = DatasetManager.load_data(os.path.join(self.input_dir, 'train'))

        # preprocess train dataset
        for transform in self.config['transformations']:
            name = transform['name']
            kwargs = {k: v for k, v in transform.items() if k != 'name'}

            transformation = TransformFactory.create(name, kwargs)
            train_dataset = transformation.apply_transform(train_dataset)

        # save data
        DatasetManager.save_data(train_dataset, data_dir=os.path.join(self.output_dir, 'train'))


if __name__ == '__main__':
    op_parser = OpParser()
    args = op_parser.parse_args()

    preprocessing = DataTrainPreprocessing(args.config, args.input_dir, args.output_dir)
    preprocessing.run()
