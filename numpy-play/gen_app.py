#!/usr/bin/env python

import matplotlib.pyplot as plt
from gen_data import SimpleDataset
from linear_regression import LinearRegression

def main():
    dataset = SimpleDataset()
    plt.scatter(dataset.x_train, dataset.y_train)
    plt.show()

    li = LinearRegression()
    pass

if __name__ == "__main__":
    main()