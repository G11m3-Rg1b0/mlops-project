import unittest
import tensorflow as tf
import os
from pathlib import Path
import shutil

from src.utils import DatasetManager

# quiet tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class UtilsTest(unittest.TestCase):

    def test_save_and_load_data(self):
        dir_test = 'tmp_test'
        path = Path(os.path.join(dir_test, 'data_save_load'))
        os.makedirs(path)

        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

        DatasetManager.save_data(dataset, path=path)
        dataset_loaded = DatasetManager.load_data(path)

        self.assertEqual(dataset.element_spec, dataset_loaded._element_spec, 'element saved and reloaded are different')

        shutil.rmtree(dir_test)
