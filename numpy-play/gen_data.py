import numpy as np

class SimpleDataset:
    def generate1(self) -> tuple[np.ndarray, np.ndarray]:
        # Generates 100 integer numbers in the interval [10 - 200]
        x_train = np.linspace(10, 200, 100)
        x_shape = x_train.shape
        # Generates 100 random numbers [-1,1] 
        r_train = np.random.randn(*x_shape)
        # Take the original series, divide it by 7 adding some perturbation
        rr_train = x_train / 7 + r_train
        # Truncate to integer
        y_train = np.maximum(1.0, rr_train)
        x_norm = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
        return (x_norm, y_train)
    
    def generate2(self) -> tuple[np.ndarray, np.ndarray]:
        # Generate synthetic data
        np.random.seed(42)
        x = np.linspace(-1, 1, 100)  # 100 points between -1 and 1
        y = 3 * x + 2 + np.random.randn(100) * 0.1  # y = 3x + 2 + noise
        return (x, y)