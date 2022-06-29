import pytest
from unittest.mock import Mock
import os
import shutil
from pathlib import Path

from src.operators.data_preparation import DataPreparation
from src.data import prepare


@pytest.fixture()
def file_list():
    return [
        '0_smthg_0.wav',
        '1_smthg_0.wav',
        '1_smthg_1.wav',
        '2_smthg_0.wav',
        '54_smthg_0.wav',
        '54_smthg_1.wav',
        '99_smthg_0.wav'
    ]


@pytest.fixture
def config():
    return {'data_formatter': 'DataFormatter'}


@pytest.fixture()
def test_dir():
    test_dir_ = 'tmp_test'
    os.mkdir(test_dir_)
    yield test_dir_
    shutil.rmtree(test_dir_)


@pytest.fixture()
def output_dir(test_dir):
    output_dir_ = os.path.join(test_dir, 'output')
    os.mkdir(output_dir_)
    return output_dir_


@pytest.fixture()
def input_dir(test_dir, file_list):
    input_dir_ = os.path.join(test_dir, 'input')
    os.mkdir(input_dir_)
    for file in file_list:
        path = os.path.join(input_dir_, file)
        with open(path, 'w') as fp:
            pass
    return input_dir_


def data_handler_save_file(spec, file_path):
    with open(file_path + '.ext', 'w') as fp:
        pass


def test_data_preparation_run(config, input_dir, output_dir, file_list):
    DataPreparation.load_operator_config = Mock(name='load_config', return_value=config)

    data_handler = Mock(name='data_handler')
    prepare.DataFormatter = Mock(name='DataFormatter', return_value=data_handler)
    data_handler.attach_mock(Mock(name='build_spectrogram', return_value=None), 'build_spectrogram')
    data_handler.attach_mock(Mock(name='save', return_value=None, side_effect=data_handler_save_file), 'save')

    assert os.listdir(input_dir) == file_list
    assert os.listdir(output_dir) == []

    prep = DataPreparation(Path(''), Path(input_dir), Path(output_dir))
    prep.run()

    prepare.DataFormatter.assert_called_once()
    assert data_handler.build_spectrogram.call_count == len(file_list)
    assert data_handler.save.call_count == len(file_list)

    for class_num in ['0', '1', '2', '54', '99']:
        class_dir = os.path.join(output_dir, 'audio-images', f'class_{class_num}')
        assert os.path.exists(class_dir)

        assert len(os.listdir(class_dir)) == [f.startswith(class_num) for f in file_list].count(True)
