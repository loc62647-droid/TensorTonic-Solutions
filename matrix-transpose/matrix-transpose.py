import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    m, n = len(A), len(A[0])
    B = np.zeros((n, m))
    for x in range(m):
        for y in range(n):
            B[y][x] = A[x][y]
    return B