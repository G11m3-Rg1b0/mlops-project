import pytest
from pathlib import Path
from unittest.mock import Mock, call

from src.operators.data_train_preprocessing import DataTrainPreprocessing
from src.operators.data_valid_preprocessing import DataValidPreprocessing
from src.operators import data_train_preprocessing
from src.operators import data_valid_preprocessing

config = {
    'transformations': [
        {
            'name': 'transform_1',
            'arg1': 'x'
        },
        {
            'name': 'transform_2'
        },
        {
            'name': 'transform_3',
            'arg1': 'y',
            'arg2': 'z'
        }
    ]
}


@pytest.fixture(params=[(DataTrainPreprocessing, data_train_preprocessing),
                        (DataValidPreprocessing, data_valid_preprocessing)])
def class_module(request):
    return request.param


# can be improve if one checks that each transformation has been correctly applied
## also we keep track of what has been called but maybe one need to check the process via the inputs
def test_module_data_preprocessing_run(class_module):
    cls, module = class_module

    cls.load_operator_config = Mock(name='load_config', return_value=config)

    module.DatasetManager = Mock(name='DatasetManager')
    module.DatasetManager.attach_mock(Mock(name='load_data', return_value=None), 'load_data')
    module.DatasetManager.attach_mock(Mock(name='save_data', return_value=None), 'save_data')

    module.check_transformations = Mock(name='check_transformation', return_value=True)
    transformation = Mock(name='transformation', return_value=None)
    transformation.attach_mock(Mock(name='apply_transform', return_value=None), 'apply_transform')
    module.TransformFactory.create = Mock(name='create', return_value=transformation)

    cls_ = cls(Path(''), Path(''), Path(''))
    cls_.run()

    assert cls_.load_operator_config.called
    assert module.check_transformations.called

    assert module.DatasetManager.load_data.called

    calls = [
        call('transform_1', {'arg1': 'x'}),
        call('transform_2', {}),
        call('transform_3', {'arg1': 'y', 'arg2': 'z'})
    ]
    assert module.TransformFactory.create.call_args_list == calls
    assert transformation.apply_transform.call_count == len(config['transformations'])

    assert module.DatasetManager.save_data.called
