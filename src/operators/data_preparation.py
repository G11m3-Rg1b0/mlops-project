import os
from src.data import prepare
from src.operator import Operator
from src.parser import OpParser
from pathlib import Path
from src.utils import check_config, check_data_formatter


class DataPreparation(Operator):
    needed_params = [
        'data_formatter'
    ]

    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path):
        self.config = self.load_operator_config(config_path, 'data_preparation')

        check_config(self.config, self.needed_params)
        check_data_formatter(self.config['data_formatter'])

        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        print('preparing data ...')

        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)

            file_manager = prepare.PrepFileSysManager()
            output_path = file_manager.build_output_path(file_path, self.output_dir)

            data_handler = getattr(prepare, self.config['data_formatter'])()
            spectrogram = data_handler.build_spectrogram(file_path)

            data_handler.save(spectrogram, output_path)


if __name__ == '__main__':
    op_parser = OpParser()
    args = op_parser.parse_args()

    preparation = DataPreparation(args.config, args.input_dir, args.output_dir)
    preparation.run()

    print('data preparation finished')
