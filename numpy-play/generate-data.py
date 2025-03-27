#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Generates 100 integer numbers in the interval [10 - 200]
x_train = np.linspace(10, 200, 100, dtype=int)
print(x_train)

x_shape = x_train.shape

# Generates 100 random numbers [-1,1] 
r_train = np.random.randn(*x_shape)
print(r_train)

# Take the original series, divide it by 7 adding some perturbation
rr_train = x_train / 7 + r_train
print(rr_train)

# Truncate to integer
y_train = np.maximum(1.0, rr_train).astype(int)
print(y_train)

plt.scatter(x_train, y_train)
plt.show()