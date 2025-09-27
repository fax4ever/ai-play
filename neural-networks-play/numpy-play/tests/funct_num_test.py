import numpy as np
from numpy_play.funct_num import func_num

def test_func_num():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Y = func_num(X)
    assert Y is not None