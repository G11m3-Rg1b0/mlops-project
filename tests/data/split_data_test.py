import unittest
from pathlib import Path

from src.pipelines.data_splitting import DataSplitting


class SplitDataTest(unittest.TestCase):

    def test_data_split_config_is_loaded_properly(self):
        needed_keys = [
            'valid_seed',
            'train_seed',
        ]

        cfg = DataSplitting(
            config_path=Path('params.yaml'),
            input_dir=Path(''),
            output_dir=Path(''),
        ).config

        self.assertIsNot(cfg, None, 'no config loaded')
        for nk in needed_keys:
            self.assertIn(nk, cfg.keys(), f'missing parameter "{nk}" in config')
