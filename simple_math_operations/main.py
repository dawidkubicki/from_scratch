import numpy as np

def matmul_forward(X: np.array, W: np.array) -> np.array:
    assert(X.shape[0] == W.shape[1])

    return np.dot(X, W)
