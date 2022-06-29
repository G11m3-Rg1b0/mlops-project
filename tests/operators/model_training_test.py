import pytest
from unittest.mock import Mock, patch, call
from pathlib import Path
import socket

from src.operators import model_training


@pytest.fixture()
def config():
    return {
        'experiment':   'exp',
        'run_name':     'name',

        'model_cfg':    [],
        'compiler':     {'arg': 'value'},
        'input_shape':  {
            'image_height': 1,
            'image_width':  2,
            'n_channels':   3,
        },
        'training_cfg': {
            'epochs':     1,
            'batch_size': 2
        },
    }


@pytest.fixture()
def patch_load_config(config):
    with patch('src.operators.model_training.ModelTraining.load_operator_config', return_value=config):
        yield config


@pytest.fixture()
def patch_load_data():
    with patch('src.operators.model_training.DatasetManager.load_data', side_effect=lambda x: x):
        yield


@pytest.fixture()
def mock_model():
    mock_model_ = Mock(name='CNNModel')
    mock_model_.attach_mock(Mock(name='compile', return_value=mock_model_), 'compile')
    mock_model_.attach_mock(Mock(name='build', return_value=mock_model_), 'build')
    mock_model_.attach_mock(Mock(name='fit', return_value=mock_model_), 'fit')
    return mock_model_


@pytest.fixture()
def patch_model(mock_model):
    with patch('src.operators.model_training.CNNModel', return_value=mock_model):
        yield mock_model


def check_server_up(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    if result == 0:
        # server is up
        pass
    else:
        # server is down
        raise Exception(f"server is not up, check for connection to {host}:{port}")

# todo add mock connection
def test_training_model_run(patch_load_config, patch_load_data, patch_model):
    check_server_up('localhost', 5000)

    train = model_training.ModelTraining(Path(''), Path(''), Path(''))
    train.run()

    calls = [
        call.compile(**patch_load_config['compiler']),
        call.build([None, *patch_load_config['input_shape'].values()]),
        call.fit('.\\train', validation_data='.\\valid', **patch_load_config['training_cfg'])
    ]

    patch_model.assert_has_calls(calls)
