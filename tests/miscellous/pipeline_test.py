import unittest
import yaml
from pathlib import Path
import shutil
import os

from src.pipeline import Pipeline


class PipelineTest(unittest.TestCase):

    def test_load_config(self):
        dir_test = 'tmp_test'
        path = Path(os.path.join(dir_test, 'load_config'))
        os.makedirs(path)
        cfg_path = Path(os.path.join(path, 'pipeline_cfg.yaml'))

        cfg_global = {
            'pipeline_name': {'parameter': 'value'}
        }
        with open(cfg_path, 'w') as fp:
            yaml.safe_dump(cfg_global, fp)

        cfg_pipe = Pipeline.load_pipeline_config(cfg_path, 'pipeline_name')

        self.assertEqual(cfg_pipe, cfg_global['pipeline_name'], 'loaded configuration for pipeline is not correct')

        shutil.rmtree(dir_test)
