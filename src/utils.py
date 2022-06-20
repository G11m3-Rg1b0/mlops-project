import tensorflow as tf


class DatasetManager:
    """Simple class to handle access to datasets."""

    @staticmethod
    def save_data(dataset: tf.data.Dataset, data_dir: str = ''):
        """Save tf.data.dataset to directory path/.

        args:
            dataset: The dataset to be saved.
            data_dir: Directory to save data into.

        return:
            Nothing.
        """
        tf.data.experimental.save(dataset, data_dir)

    @staticmethod
    def load_data(data_dir: str = '') -> tf.data.Dataset:
        """Load data from directory.

        args:
            data_dir: Directory where data is stored.

        return:
            The category's dataset.
        """
        return tf.data.experimental.load(data_dir)
