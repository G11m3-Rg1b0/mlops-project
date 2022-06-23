import mlflow
from pathlib import Path
import yaml
import os

from src.model.model import CNNModel
from src.utils import DatasetManager
from src.pipeline import Pipeline
from src.parser import PipeParser

from src.mlflow_utils import mlflow_wrapper

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelTraining(Pipeline):
    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config_path = config_path
        self.config = self.load_pipeline_config(config_path, 'model_training')

        self.input_dir = input_dir
        self.output_dir = output_dir

    @mlflow_wrapper
    def run(self):
        print('building model ...')

        # Create CNN model
        model = CNNModel(self.config['model_cfg'])
        model.compile(**self.config['compiler'])
        model.build(
            [None,
             self.config['input_shape']['image_height'],
             self.config['input_shape']['image_width'],
             self.config['input_shape']['n_channels']]
        )

        # load data for training
        valid_dataset = DatasetManager.load_data(os.path.join(self.input_dir, 'valid'))
        train_dataset = DatasetManager.load_data(os.path.join(self.input_dir, 'train'))

        print('training model ...')
        model.fit(
            train_dataset,
            epochs=self.config['training_cfg']['epochs'],
            validation_data=valid_dataset,
            batch_size=self.config['training_cfg']['batch_size']
        )


if __name__ == '__main__':
    pipe_parser = PipeParser()
    args = pipe_parser.parse_args()

    training = ModelTraining(args.config, args.input_dir, args.model_dir)
    training.run()
