import numpy as np
import keras
from keras import layers

class LinearRegression:
    def __init__(self, x : np.ndarray, y : np.ndarray):
        # Define a simple linear regression model
        self.model = keras.Sequential([
            # Single neuron for y = mx + b
            layers.Dense(1, input_shape=(1,), activation=None)  
        ])

        # Compile the model
        # Stochastic Gradient Descent & Mean Squared Error
        self.model.compile(optimizer='sgd', loss='mse')  

        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

        # Train the model
        self.model.fit(X, Y, epochs=200, verbose=1)

        # Evaluate the model
        m, b = self.model.get_weights()
        print(f"Learned parameters: m = {m[0][0]:.4f}, b = {b[0]:.4f}")

    def prediction(self, x : np.ndarray) -> np.ndarray:
        X = x.reshape(-1, 1)
        Y = self.model.predict(X)
        return Y.reshape(-1)
