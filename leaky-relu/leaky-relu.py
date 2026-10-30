import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    # Write code here
    ans = []
    for i in range(len(x)):
        if x[i] >= 0:
            ans.append(x[i])
        else:
            ans.append(x[i] * alpha)
    return np.array(ans)