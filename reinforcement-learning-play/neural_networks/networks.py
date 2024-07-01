import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics

def make_model1(n_features):
    input = layers.Input([n_features])
    x = layers.Dense(32, activation="relu")(input)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(1)(x)

    return models.Model(inputs=input, outputs=x, name="regression_fc1")

def make_model2():
    return models.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

def make_model3():
    return models.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])