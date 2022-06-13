""""
Script to run the preprocessing pipeline
"""
from pathlib import Path
from src.data.preprocessing import TransformFactory
from src.utils import DataManager
from src.pipeline import Pipeline
from src.parser import PipeParser


class DataTrainPreprocessing(Pipeline):
    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config = self.load_pipeline_config(config_path, 'data_train_preprocessing')

        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        print('preprocessing data ...')

        train_dataset = DataManager.load_data_category('train', self.input_dir)

        # preprocess train dataset
        for cfg in self.config:
            name = cfg['name']
            kwargs = {k: v for k, v in cfg.items() if k != 'name'}

            transformation = TransformFactory.create(name, kwargs)
            train_dataset = transformation.apply_transform(train_dataset)

        # save data
        DataManager.save_data_category(train_dataset, 'train', path=self.output_dir)


if __name__ == '__main__':
    pipe_parser = PipeParser()
    args = pipe_parser.parse_args()

    preprocessing = DataTrainPreprocessing(args.config, args.input_dir, args.output_dir)
    preprocessing.run()
