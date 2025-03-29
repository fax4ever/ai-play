import numpy as np
from gen_data import SimpleDataset
from linear_regression import LinearRegression

dataset = SimpleDataset()

# Generate synthetic data
np.random.seed(42)
x = np.linspace(-1, 1, 100)  # 100 points between -1 and 1
y = 3 * x + 2 + np.random.randn(100) * 0.1  # y = 3x + 2 + noise

lr = LinearRegression(x, y)