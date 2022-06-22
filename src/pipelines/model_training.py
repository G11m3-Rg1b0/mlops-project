import mlflow
from pathlib import Path
import yaml
import os

from src.model.model import CNNModel
from src.utils import DatasetManager
from src.pipeline import Pipeline
from src.parser import PipeParser

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelTraining(Pipeline):
    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config_path = config_path
        self.config = self.load_pipeline_config(config_path, 'model_training')

        self.input_dir = input_dir
        self.output_dir = output_dir

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

        # mlflow setup
        # todo:
        #   add initializer for mlflow runs
        #   lop_params to track the version of the data used during the training

        mlflow.keras.autolog()
        mlflow.set_tracking_uri('http://localhost:5000')

        experiment = mlflow.get_experiment_by_name(self.config['experiment'])
        if experiment:
            mlflow_exp_id = experiment.experiment_id
        else:
            mlflow_exp_id = mlflow.create_experiment(name=self.config['experiment'])

        # train model
        print('training model ...')
        with mlflow.start_run(run_name=self.config['run_name'], experiment_id=mlflow_exp_id) as mlflow_run:
            mlflow_run_id = mlflow_run.info.run_id
            print(f'run id: {mlflow_run_id}')
            mlflow_exp_id = mlflow_run.info.experiment_id
            print(f'exp id: {mlflow_exp_id}')

            model.fit(
                train_dataset,
                epochs=self.config['training_cfg']['epochs'],
                validation_data=valid_dataset,
                batch_size=self.config['training_cfg']['batch_size']
            )

            self.save_run_id(mlflow_run_id, mlflow_exp_id)

    def save_run_id(self, run_id: str, exp_id: str) -> None:
        """Update the configuration file with the training run id for upcoming evaluations."""
        with open(self.config_path, 'r') as fp:
            cfg = yaml.safe_load(fp)

        cfg['model_evaluation']['run_id'] = run_id
        cfg['model_evaluation']['exp_id'] = exp_id

        with open(self.config_path, 'w') as fp:
            yaml.safe_dump(cfg, fp)


if __name__ == '__main__':
    pipe_parser = PipeParser()
    args = pipe_parser.parse_args()

    training = ModelTraining(args.config, args.input_dir, args.model_dir)
    training.run()
