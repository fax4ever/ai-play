import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics

def make_model1(input_shape):
    input = layers.Input(input_shape)
    x = layers.Dense(32, activation="relu")(input)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(1)(x)

    return models.Model(inputs=input, outputs=x, name="regression_fc1")