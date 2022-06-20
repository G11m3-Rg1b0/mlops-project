import unittest
from pathlib import Path

from src.pipelines.data_train_preprocessing import DataTrainPreprocessing
from src.pipelines.data_valid_preprocessing import DataValidPreprocessing
from src.data.preprocessing import TransformFactory


class PreprocessDataTest(unittest.TestCase):

    def test_data_preprocess_config_is_loaded_properly(self):

        train_cfg = DataTrainPreprocessing(
            config_path=Path('params.yaml'),
            input_dir=Path(''),
            output_dir=Path(''),
        ).config

        valid_cfg = DataValidPreprocessing(
            config_path=Path('params.yaml'),
            input_dir=Path(''),
            output_dir=Path(''),
        ).config

        for cfg in [train_cfg, valid_cfg]:
            self.assertIsNot(cfg, None, 'no config loaded')
            self.assertIsInstance(cfg, list, f'config has bad input format: {type(cfg)} instead of list')
            for transf_cfg in cfg:
                self.assertIn('name', transf_cfg.keys(), 'missing transformation name')

    def test_unknown_transformation(self):
        self.assertRaises(AssertionError, TransformFactory.create, 'unknown_transformation_name', {})
