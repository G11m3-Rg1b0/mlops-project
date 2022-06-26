from pathlib import Path
import os

from src.model.model import CNNModel
from src.utils import DatasetManager, check_config
from src.operator import Operator
from src.parser import OpParser

from src.mlflow_utils import mlflow_wrapper

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelTraining(Operator):
    needed_params = [
        'experiment',
        'model_cfg',
        'compiler',
        ('input_shape', 'image_height'),
        ('input_shape', 'image_width'),
        ('input_shape', 'n_channels'),
        ('training_cfg', 'epochs'),
        ('training_cfg', 'batch_size'),
    ]

    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config_path = config_path
        self.config = self.load_operator_config(config_path, 'model_training')
        check_config(self.config, self.needed_params)

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
    op_parser = OpParser()
    args = op_parser.parse_args()

    training = ModelTraining(args.config, args.input_dir, args.model_dir)
    training.run()
