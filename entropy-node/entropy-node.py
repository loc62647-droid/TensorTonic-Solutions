import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    classes, counts = np.unique(y, return_counts = True)
    H = 0
    for i in range(len(classes)):
        if len(counts) == 0:
            H += 0
        else:
            H += -(counts[i] / len(y) * np.log2(counts[i] / len(y)))
    return float(H)