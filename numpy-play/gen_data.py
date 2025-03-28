import numpy as np

class SimpleDataset:
    def __init__(self):
        (self.x, self.y) = generate()

def generate() -> tuple[np.ndarray, np.ndarray]:
    # Generates 100 integer numbers in the interval [10 - 200]
    x_train = np.linspace(10, 200, 100, dtype=int)
    x_shape = x_train.shape

    # Generates 100 random numbers [-1,1] 
    r_train = np.random.randn(*x_shape)

    # Take the original series, divide it by 7 adding some perturbation
    rr_train = x_train / 7 + r_train

    # Truncate to integer
    y_train = np.maximum(1.0, rr_train).astype(int)
    return (x_train, y_train)