import unittest
import os
import shutil
import yaml
from pathlib import Path

from src.data.prepare import PrepFileSysManager
from src.pipelines.data_preparation import DataPreparation


class PrepareDataTest(unittest.TestCase):
    """Tests for the preparation pipeline"""

    def test_data_preparation_config_is_loaded_properly(self):
        cfg = DataPreparation(
            config_path=Path('params.yaml'),
            input_dir=Path(''),
            output_dir=Path(''),
        ).config
        self.assertIsNot(cfg, None, 'no config loaded')
        self.assertIn('data_formatter', cfg.keys(), 'missing parameter "data_formatter" in config')

    def test_if_raw_data_is_wave_format(self):
        with open('configs/base_cfg.yaml', 'r') as fp:
            cfg = yaml.safe_load(fp)
        ls = os.listdir(cfg['base']['dir_raw_data'])
        self.assertTrue(any(fn.endswith('wav') for fn in ls))

    def test_class_detection_from_filename(self):
        filename = '45_smthg_3.wav'
        cl = PrepFileSysManager().get_class(filename)
        self.assertEqual(cl, '45', f'wrong detection of class label from filename {filename}, get {cl} instead of 45')

    def test_prepared_data_directory_structure(self):
        input_dir = os.path.join('tmp', 'data_raw')
        output_dir = os.path.join('tmp', 'structure')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        file_name = '3_file_8.txt'
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'w'):
            pass

        test_path = PrepFileSysManager().build_output_path(file_path, output_dir)
        real_output_dir = os.path.join(output_dir, 'audio-images', 'class_3')

        test_dir, test_name = os.path.split(test_path)

        self.assertEqual(real_output_dir, test_dir,
                         f'raw data saved in a wrong directory tree: {test_dir} != {real_output_dir}')
        self.assertEqual('3_file_8', test_name, f'wrong filename: {test_name} != {file_name}')

        shutil.rmtree('tmp')
