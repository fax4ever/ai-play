#!/usr/bin/env python

import matplotlib.pyplot as plt
from gen_data import SimpleDataset
from linear_regression import LinearRegression
import numpy as np

def main():
    dataset = SimpleDataset()
    plt.scatter(dataset.x, dataset.y)
    #plt.show()

    lr = LinearRegression(dataset.x, dataset.y)
    plt.scatter(dataset.x, lr.prediction(dataset.x))
    #plt.show()
    pass

if __name__ == "__main__":
    main()