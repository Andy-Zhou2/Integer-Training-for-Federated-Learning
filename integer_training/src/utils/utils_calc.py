import numpy as np
import logging


def truncate_divide(a, b):
    """
    Truncate division (towards zero) of a by b, which agrees with C++.
    np.divide gives fp with 3/2.
    // operator in Python is floor division, so -5 // 2 = -3 while in C++ -5 / 2 = -2.
    note that we don't use np.int_(a/b) because this involves floating point division
    which cause non-deterministic behavior for parallel processing.
    Hence, we use a // b if a * b > 0 else (a + (-a % b)) // b.
    """
    product = a * b
    expected_sign = np.sign(a) * np.sign(b)
    if np.any(expected_sign != np.sign(product)):
        logging.warning(f"Sign mismatch in truncate_divide: a={a}, b={b}, a*b={product}")
    return np.where(product > 0, a // b, (a + (-a % b)) // b)
