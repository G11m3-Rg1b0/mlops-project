import mlflow
import os
from pathlib import Path

from src.model.model import CNNModel
from src.pipeline import Pipeline
from src.parser import PipeParser
from src.utils import DatasetManager


class ModelEvaluation(Pipeline):
    def __init__(self, config_path: Path, input_dir: Path, model_dir: Path):
        self.config = self.load_pipeline_config(config_path, 'model_evaluation')

        self.input_dir = input_dir
        self.model_dir = model_dir

    def run(self) -> None:
        print(f'evaluate run: {self.config["run_id"]}')
        print(f'use data from: {self.input_dir or "default directory"}')

        valid_dataset = DatasetManager.load_data(os.path.join(self.input_dir, 'valid'))
        model = self.load_model()

        outputs = model.evaluate(valid_dataset, verbose=1, batch_size=self.config['batch_size'])

        names = model.metrics_names
        for v, n in zip(outputs, names):
            print('{}: {:.6f}'.format(n, v))

    def load_model(self) -> CNNModel:
        """Load the model from the default mlflow model registry.

        return:
            The mlflow saved model.
        """
        path_to_model = os.path.join(self.model_dir, self.config['exp_id'], self.config['run_id'],
                                     'artifacts', 'model')
        return mlflow.keras.load_model(path_to_model)


if __name__ == '__main__':
    pipe_parser = PipeParser()
    args = pipe_parser.parse_args()

    evaluation = ModelEvaluation(args.config, args.input_dir, args.model_dir)
    evaluation.run()
