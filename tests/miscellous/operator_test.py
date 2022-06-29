import pytest
import yaml
from pathlib import Path
import shutil
import os

from src.operator import Operator


@pytest.fixture
def path():
    dir_test = 'tmp_test'
    path = os.path.join(dir_test, 'load_config')
    os.makedirs(path)
    yield path
    shutil.rmtree(dir_test)


@pytest.fixture
def config():
    return {
        'pipeline_name': {'parameter': 'value'}
    }


@pytest.fixture
def cfg_path(path):
    return Path(os.path.join(path, 'pipeline_cfg.yaml'))


def test_load_config(cfg_path, config):
    with open(cfg_path, 'w') as fp:
        yaml.safe_dump(config, fp)

    cfg_pipe = Operator.load_operator_config(cfg_path, 'pipeline_name')

    assert cfg_pipe == config['pipeline_name']
