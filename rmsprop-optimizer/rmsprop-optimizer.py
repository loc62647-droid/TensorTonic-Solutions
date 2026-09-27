import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    s = np.array(s)
    w = np.array(w)
    g = np.array(g)
    new_w = [0] * len(w)
    new_s = [0] * len(s)
    for i in range(len(s)):
        new_s[i] = beta * s[i] + (1 - beta) * g[i] ** 2
        new_w[i] = w[i] - lr / (np.sqrt(new_s[i] + eps)) * g[i]
    return new_w, new_s
        