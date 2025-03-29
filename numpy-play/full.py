import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Generate synthetic data
np.random.seed(42)
x = np.linspace(-1, 1, 100)  # 100 points between -1 and 1
y = 3 * x + 2 + np.random.randn(100) * 0.1  # y = 3x + 2 + noise

X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

# Define a simple linear regression model
model = keras.Sequential([
    layers.Dense(1, input_shape=(1,), activation=None)  # Single neuron for y = mx + b
])

# Compile the model
model.compile(optimizer='sgd', loss='mse')  # Stochastic Gradient Descent & Mean Squared Error

# Train the model
model.fit(X, Y, epochs=200, verbose=1)

# Evaluate the model
m, b = model.get_weights()
print(f"Learned parameters: m = {m[0][0]:.4f}, b = {b[0]:.4f}")

# Make predictions
X_test = np.array([[-1], [0], [1]])  # Test inputs
predictions = model.predict(X_test)
print("Predictions:", predictions)
