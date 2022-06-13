"""
Script to run data preparation.
"""
import os
from src.data.prepare import FileSysManager, PngFormatter
from src.pipeline import Pipeline
from src.parser import PipeParser
from pathlib import Path


class DataPreparation(Pipeline):
    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config = self.load_pipeline_config(config_path, 'data_preparation')

        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        print('preparing data ...')

        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)

            output_path = FileSysManager().build_output_path(file_path, self.output_dir)

            data_handler = PngFormatter()
            spectrogram = data_handler.build_spectrogram(file_path)
            data_handler.save(spectrogram, output_path)


if __name__ == '__main__':
    pipe_parser = PipeParser()
    args = pipe_parser.parse_args()

    preparation = DataPreparation(args.config, args.input_dir, args.output_dir)
    preparation.run()

    print('data preparation finished')
