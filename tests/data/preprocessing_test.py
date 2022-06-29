import pytest
from pathlib import Path
import tensorflow as tf

from src.data import preprocessing
from src.data.preprocessing import TransformFactory

import numpy as np


def test_create_unknown_transformation():
    unk = 'unk_tf'
    with pytest.raises(AssertionError, match=f"Unknown transformation '{unk}'"):
        TransformFactory.create(unk, {})


@pytest.fixture(params=[('RandomFlip', {}), ('RandomRotation', {'factor': 1}), ('Rescaling', {'scale': 1})])
def transforms_params(request):
    return request.param


@pytest.fixture()
def dataset():
    return tf.data.Dataset.from_tensor_slices(([np.random.rand(128, 128, 128)], [0]))


def test_create_keras_transformation(transforms_params):
    name, kwargs = transforms_params
    TransformFactory.create(name, kwargs)


@pytest.fixture()
def transformation(transforms_params):
    name, kwargs = transforms_params
    return TransformFactory.create(name, kwargs)


def test_apply_transform(transformation, dataset):
    transformation.apply_transform(dataset)
