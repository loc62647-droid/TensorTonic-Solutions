import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    #First moment
    m_new = [0] * len(m)
    v_new = [0] * len(v)
    param_new = [0] * len(param)
    for i in range(len(param)):
        m_new[i] = beta1 * m[i] + (1 - beta1) * grad[i]
        v_new[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
        m_bias = m_new[i] / (1 - beta1 ** (t))
        v_bias= v_new[i] / (1 - beta2 ** (t))
        param_new[i] = param[i] - lr * m_bias / (np.sqrt(v_bias) + eps)
    return param_new, m_new, v_new