#!/usr/bin/env python

import matplotlib.pyplot as plt
from gen_data import SimpleDataset
from linear_regression import LinearRegression
import numpy as np

def execute(x : np.ndarray, y : np.ndarray):
    plt.scatter(x, y)
    plt.show()
    lr = LinearRegression(x, y)
    plt.scatter(x, lr.prediction(x))
    plt.show()

def main():
    dataset = SimpleDataset()
    (x1, y1) = dataset.generate1()
    (x2, y2) = dataset.generate2()
    execute(x1, y1)
    execute(x2, y2)

if __name__ == "__main__":
    main()