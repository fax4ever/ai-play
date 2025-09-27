import numpy as np

def func_num(X: np.ndarray) -> np.ndarray:
    # Check that X has shape (n, 2)
    if len(X.shape) != 2:
        raise ValueError(f"Expected X to be 2D array, but got {len(X.shape)}D array with shape {X.shape}")
    if X.shape[1] != 2:
        raise ValueError(f"Expected X to have 2 columns, but got {X.shape[1]} columns with shape {X.shape}")
    
    # Extract x1 and x2 from the input array
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Implement f(x) = sin(x1) * cos(x2) + sin(0.5*x1) * cos(0.5*x2)
    result = np.sin(x1) * np.cos(x2) + np.sin(0.5 * x1) * np.cos(0.5 * x2)
    
    return result