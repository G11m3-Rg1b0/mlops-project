import pytest
import os

from src.model.model import CNNModel

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@pytest.fixture
def layer_name():
    return 'Conv2D'


@pytest.fixture
def filters():
    return 32


@pytest.fixture
def kernel_size():
    return 3, 3


@pytest.fixture
def wright_config(layer_name, filters, kernel_size):
    return [
        {
            'name':        layer_name,
            'filters':     filters,
            'kernel_size': kernel_size
        },
        {
            'name': 'BatchNormalization'
        }
    ]


@pytest.fixture
def wrong_config():
    return [
        {
            'name':        'Conv2D',
            'wrong_param': 'value'
        },
        {
            'name': 'BatchNormalization'
        }
    ]


def test_model_initialization(wright_config, layer_name, filters, kernel_size):
    config_layers = [layer['name'] for layer in wright_config]
    config_layers.sort()

    model = CNNModel(wright_config)
    assert isinstance(model, CNNModel)

    model_layers = [layer.__class__.__name__ for layer in model.layers]
    model_layers.sort()
    assert config_layers == model_layers

    conv2d_model_params = model.get_layer(layer_name.lower()).get_config()
    assert filters == conv2d_model_params['filters'] and kernel_size == conv2d_model_params['kernel_size']


def test_wrong_kwarg_to_layer(wrong_config):
    with pytest.raises(Exception):
        CNNModel(wrong_config)

# todo empty config ?