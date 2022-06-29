import shutil
from unittest.mock import Mock, patch

from src.utils import *
from src import utils

import pytest

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@pytest.fixture()
def dir_test():
    dir_test_ = 'tmp_test'
    os.mkdir(dir_test_)
    yield dir_test_
    shutil.rmtree(dir_test_)


class TestDatasetManager:

    @pytest.fixture
    def dataset(self):
        return tf.data.Dataset.from_tensor_slices([1, 2, 3])

    @pytest.fixture
    def path(self, dir_test):
        path = os.path.join(dir_test, 'data_save_load')
        os.makedirs(path)
        return path

    @staticmethod
    def test_save_and_load_data(dataset, path):
        DatasetManager.save_data(dataset, data_dir=path)
        dataset_loaded = DatasetManager.load_data(path)

        assert dataset.element_spec == dataset_loaded._element_spec


@pytest.fixture
def needed_keys():
    return [
        'arg1',
        ('arg3', 'arg4')
    ]


@pytest.fixture
def config():
    return {
        'arg1': '',
        'arg3': {'arg4': ''}
    }


def test_check_config(config, needed_keys):
    check_config(config, needed_keys)
    with pytest.raises(AssertionError):
        check_config(config, ['arg5'])


@pytest.fixture()
def transformations():
    return [
        {'name': 'unk_transform'},
        {'name': 'Rescaling'},
        {'name': 'RandomRotation'}
    ]


def test_check_transformations(transformations):
    with pytest.raises(AssertionError, match="The transformation 'unk_transform' does not exist"):
        check_transformations(transformations)


@pytest.fixture()
def data_formatter():
    return [
        'PngFormatter',
        'ArrFormatter',
        'unk_formatter'
    ]


def test_data_formatter(data_formatter):
    with pytest.raises(AssertionError, match="The formatter 'unk_formatter' does not exist"):
        for formatter in data_formatter:
            check_data_formatter(formatter)


@pytest.fixture(params=[get_base_config, get_params_config])
def config_getter(request):
    return request.param


def test_get_config(config_getter):
    config = config_getter()
    assert config
    assert isinstance(config, dict)


@pytest.fixture()
def info_path(dir_test):
    info_path_ = os.path.join(dir_test, 'info_path.yaml')
    with open(info_path_, 'w'):
        pass
    return info_path_


@pytest.fixture()
def mock_get_base_config(info_path):
    with patch('src.utils.get_base_config', return_value={'mlflow_last_run_info': info_path}):
        yield


@pytest.fixture()
def model_dir():
    return 'model_dir'


@pytest.fixture()
def mock_yaml_safe_load(model_dir):
    with patch('src.utils.yaml.safe_load', return_value={'artifact_uri': model_dir}):
        yield


def test_get_path_to_last_model(mock_get_base_config, mock_yaml_safe_load, model_dir):
    expected_path = os.path.join(model_dir, 'model')

    tested_path = get_path_to_last_model()

    assert expected_path == tested_path


@pytest.fixture()
def module_path(dir_test):
    module_path_ = os.path.join(dir_test, 'path_to_model')
    os.makedirs(module_path_)
    return module_path_


def test_mlflow_keras_load_model(module_path):
    mlflow.keras.load_model = Mock(name='mlflow.keras.load_model')
    utils.get_path_to_last_model = Mock(name='get_path_to_last_model', return_value=module_path)

    mlflow_keras_load_model()

    mlflow.keras.load_model.assert_called_once_with('tmp_test\\path_to_model')


@pytest.fixture()
def run_info():
    return {
        'run_id':        '0',
        'experiment_id': '0',
        'artifact_uri':  'path'
    }


def test_save_run_info(mock_get_base_config, run_info, info_path):
    save_run_info(run_info['run_id'], run_info['experiment_id'], run_info['artifact_uri'])

    with open(info_path, 'r') as fp:
        tested_run_info = yaml.safe_load(fp)

    assert tested_run_info == run_info
