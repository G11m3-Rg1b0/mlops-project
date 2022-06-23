import mlflow
import os
from pathlib import Path

from src.model.model import CNNModel
from src.pipeline import Pipeline
from src.parser import PipeParser
from src.utils import DatasetManager
from src.mlflow_utils import load_model

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelEvaluation(Pipeline):
    def __init__(self, config_path: Path, input_dir: Path, model_dir: Path):
        self.config = self.load_pipeline_config(config_path, 'model_evaluation')

        self.input_dir = input_dir
        self.model_dir = model_dir

    def run(self) -> None:
        print(f'use data from: {self.input_dir or "default directory"}')

        valid_dataset = DatasetManager.load_data(os.path.join(self.input_dir, 'valid'))
        model = load_model()

        outputs = model.evaluate(valid_dataset, verbose=1, batch_size=self.config['batch_size'])

        names = model.metrics_names
        for v, n in zip(outputs, names):
            print('{}: {:.6f}'.format(n, v))


if __name__ == '__main__':
    pipe_parser = PipeParser()
    args = pipe_parser.parse_args()

    evaluation = ModelEvaluation(args.config, args.input_dir, args.model_dir)
    evaluation.run()
