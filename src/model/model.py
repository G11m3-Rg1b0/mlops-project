import tensorflow as tf
from typing import List


class CNNModel(tf.keras.Model):

    def __init__(self, config: List[dict]):
        super(CNNModel, self).__init__()
        self.model_cfg = config

        self.model_layers = []
        for layer in self.model_cfg:
            name = layer['name']
            params = {k: v for k, v in layer.items() if k != 'name'}
            self.model_layers.append(getattr(tf.keras.layers, name)(**params))

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x
