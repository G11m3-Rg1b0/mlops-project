import unittest
from pathlib import Path
import os

from src.pipelines.model_training import ModelTraining
from src.model.model import CNNModel

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelTest(unittest.TestCase):

    def test_model_train_config_is_loaded_properly(self):
        needed_keys = [
            'experiment',
            'model_cfg',
            'compiler',
            ('input_shape', 'image_height'),
            ('input_shape', 'image_width'),
            ('input_shape', 'n_channels'),
            ('training_cfg', 'epochs'),
            ('training_cfg', 'batch_size'),
        ]

        cfg = ModelTraining(
            config_path=Path('params.yaml'),
            input_dir=Path(''),
            output_dir=Path(''),
        ).config

        self.assertIsNot(cfg, None, 'no config loaded')
        for nk in needed_keys:
            if isinstance(nk, tuple):
                self.assertIn(nk[1], cfg[nk[0]].keys(), f'missing parameter "{nk}" in config')
                continue
            self.assertIn(nk, cfg.keys(), f'missing parameter "{nk}" in config')

    def test_model_initialization(self):
        mock_cfg = [
            {
                'name':        'Conv2D',
                'filters':     32,
                'kernel_size': 3
            },
            {
                'name': 'BatchNormalization'
            }
        ]

        model = CNNModel(mock_cfg)

        self.assertIsInstance(model, CNNModel, 'error initializing CNNModel')

    def test_wrong_kwarg_to_layer(self):
        mock_cfg = [
            {
                'name': 'Conv2D',
                'wrong_param': 'value'
            },
            {
                'name': 'BatchNormalization'
            }
        ]
        self.assertRaises(TypeError, CNNModel, mock_cfg, 'did not catch the name of the layer')


        # build a model with a custom layer ?
