import pytest
from unittest.mock import Mock, call
import tensorflow as tf
from pathlib import Path

from src.operators.data_splitting import DataSplitting
from src.operators import data_splitting


@pytest.fixture()
def config():
    return {
        'train_seed': 4,
        'valid_seed': 42,
        'arg1':       'val1',
        'arg2':       'val2'
    }


@pytest.fixture()
def input_dir():
    return Path('dummy_dir')


@pytest.fixture()
def output_dir():
    return Path('dummy_dir')


def return_dataset_type(*args, **kwargs):
    return kwargs['subset']




def test_data_splitting_run(config, input_dir, output_dir):
    tf.keras.preprocessing.image_dataset_from_directory = Mock(name='tf_preprocess', side_effect=return_dataset_type)
    DataSplitting.load_operator_config = Mock(name='load_config', return_value=config)
    data_splitting.DatasetManager.save_data = Mock(name='data_save', return_value=None)

    split = DataSplitting(Path(''), input_dir, output_dir)
    split.run()

    tf_calls = [
        call(subset='training', directory='dummy_dir\\audio-images', seed=4, arg1='val1', arg2='val2'),
        call(subset='validation', directory='dummy_dir\\audio-images', seed=42, arg1='val1', arg2='val2')
    ]
    assert tf.keras.preprocessing.image_dataset_from_directory.call_args_list == tf_calls

    save_calls = [
        call('training', data_dir='dummy_dir\\train'),
        call('validation', data_dir='dummy_dir\\valid')
    ]
    assert data_splitting.DatasetManager.save_data.call_args_list == save_calls
