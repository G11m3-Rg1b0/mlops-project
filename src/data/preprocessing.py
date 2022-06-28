from abc import ABCMeta, abstractmethod

import tensorflow as tf


class AbstractTransform(metaclass=ABCMeta):
    """Abstract class to apply transformations on dataset."""

    @abstractmethod
    def apply_transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply transformation to dataset."""
        pass


class TransformFactory:

    @staticmethod
    def create(name: str, kwargs: dict) -> AbstractTransform:
        assert name in globals(), f"Unknown transformation '{name}'"
        # not good for debugging
        return globals()[name](kwargs)


########################################
#    official keras transformations    #
########################################

class KerasTransformation(tf.keras.Sequential, AbstractTransform):
    """Keras transformation."""

    def apply_transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x, y: (self.__call__(x, training=True), y))


class RandomFlip(KerasTransformation):
    """Transformation class 'RandomFlip' from tensorflow."""

    def __init__(self, kwargs):
        super(RandomFlip, self).__init__()
        self.add(
            tf.keras.layers.experimental.preprocessing.RandomFlip(**kwargs)
        )


class RandomRotation(KerasTransformation):
    """Transformation class 'RandomRotation' from tensorflow."""

    def __init__(self, kwargs):
        super(RandomRotation, self).__init__()
        self.add(
            tf.keras.layers.experimental.preprocessing.RandomRotation(**kwargs)
        )


class Rescaling(KerasTransformation):
    """Transformation class 'Rescaling' from tensorflow."""

    def __init__(self, kwargs):
        super(Rescaling, self).__init__()
        self.add(
            tf.keras.layers.experimental.preprocessing.Rescaling(**kwargs)
        )
