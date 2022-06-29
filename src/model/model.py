import tensorflow as tf
from typing import List


# todo add custom layers

class CNNModel(tf.keras.Model):

    def __init__(self, config: List[dict]):
        super().__init__()
        self.model_cfg = config

        self.model_layers = []
        self._initiate_model()

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x

    def _initiate_model(self):
        for layer in self.model_cfg:
            name = layer['name']
            params = {k: v for k, v in layer.items() if k != 'name'}
            try:
                self.model_layers.append(getattr(tf.keras.layers, name)(**params))
            except TypeError as e:
                raise Exception(f"error initializing layer '{name}'\n{e}")
