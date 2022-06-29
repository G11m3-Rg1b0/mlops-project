import pytest
from pathlib import Path
from unittest.mock import Mock, patch, PropertyMock, call

from src.operators.model_evaluation import ModelEvaluation


@pytest.fixture()
def patch_load_data():
    with patch('src.operators.model_evaluation.DatasetManager.load_data', return_value='some_data'):
        yield


@pytest.fixture()
def mock_metrics():
    return PropertyMock(return_value=['arg1', 'arg2', 'arg3'])


@pytest.fixture()
def mock_model(mock_metrics):
    mock_model_ = Mock(name='CNNModel')
    mock_model_.attach_mock(Mock(name='evaluate', return_value=[0.123, 0.456666, 0.78999]), 'evaluate')
    type(mock_model_).metrics_names = mock_metrics
    return mock_model_


@pytest.fixture()
def patch_load_model(mock_model):
    with patch('src.operators.model_evaluation.mlflow_keras_load_model', return_value=mock_model):
        yield mock_model


@pytest.fixture()
def config():
    return {
        'batch_size': 9
    }


@pytest.fixture()
def patch_load_config(config):
    with patch('src.operators.model_evaluation.ModelEvaluation.load_operator_config', return_value=config):
        yield


def test_model_evaluation_run(patch_load_data, patch_load_config, patch_load_model, mock_metrics):
    evaluate = ModelEvaluation(Path(''), Path(''), Path(''))
    evaluate.run()

    calls = [
        call.evaluate('some_data', verbose=1, batch_size=9)
    ]

    patch_load_model.assert_has_calls(calls)
    mock_metrics.assert_called_once()
