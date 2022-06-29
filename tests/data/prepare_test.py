import pytest
import os
import shutil
import yaml
from pathlib import Path

from src.data.prepare import PrepFileSysManager, PngFormatter, ArrFormatter
from src.operators.data_preparation import DataPreparation


@pytest.fixture
def file_name():
    return '45_audio_3.wav'


@pytest.fixture
def test_dir():
    dir_ = os.path.normpath('tmp')
    os.mkdir(dir_)
    yield dir_
    shutil.rmtree(dir_)


@pytest.fixture
def file_path(test_dir, file_name):
    input_dir = os.path.join(test_dir, 'data_raw')
    os.makedirs(input_dir)
    file_path_ = os.path.join(input_dir, file_name)
    with open(file_path_, 'w'):
        pass
    return file_path_


@pytest.fixture
def output_dir(test_dir):
    output_dir_ = os.path.join(test_dir, 'structure')
    os.makedirs(output_dir_)
    return output_dir_


def test_class_detection_from_filename(file_name):
    num_class = PrepFileSysManager().get_class(file_name)
    assert num_class == '45'


def test_prepared_data_directory_structure(file_path, output_dir):
    test_path = PrepFileSysManager().build_output_path(file_path, output_dir)
    test_dir, test_name = os.path.split(test_path)

    real_output_dir = os.path.join(output_dir, 'audio-images', 'class_45')

    assert real_output_dir == test_dir
    assert '45_audio_3' == test_name


@pytest.fixture(params=[PngFormatter, ArrFormatter])
def data_formatter(request):
    return request.param


def test_data_formatter(data_formatter):
    raise NotImplementedError


#####################
#   data testing    #
#####################

@pytest.fixture
def base_config():
    with open('configs/base_cfg.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    return config['base']


def test_if_raw_data_is_wave_format(base_config):
    ls = os.listdir(base_config['dir_raw_data'])
    assert any(fn.endswith('wav') for fn in ls)
