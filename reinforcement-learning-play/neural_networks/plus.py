import sklearn.datasets
from sklearn.model_selection import train_test_split
from networks import make_model1
from tensorflow import keras
from keras import callbacks, metrics
import tensorflow as tf
import datetime

dataset1 = sklearn.datasets.fetch_california_housing()
n_samples = dataset1.data.shape[0]
n_features = dataset1.data.shape[1]
print(dataset1.DESCR)
print(f"N samples: {n_samples}")
print(f"N features: {n_features}")

train_x1, test_x1, train_y1, test_y1 = train_test_split(
    dataset1.data,
    dataset1.target,
    test_size=0.2,
    shuffle=True,
    random_state=1918357
)

model1 = make_model1(n_features)
model1.summary()

model1.compile(
    optimizer="adam", # Use an optimizer class and decrement the learning rate (learning step)
    loss="mean_squared_error"
)

calls = callbacks.TensorBoard(log_dir="logs/run1")

model1.fit(
    train_x1,
    train_y1,
    batch_size=128,
    epochs=20,
    callbacks=calls
)

pred_y1 = model1.predict(test_x1)
mse = metrics.mean_squared_error(test_y1, pred_y1[:,0])
print('Mean squared error', mse)