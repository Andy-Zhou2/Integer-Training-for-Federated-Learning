import numpy as np


def truncate_divide(a, b):
    """
    Truncate division (towards zero) of a by b, which agrees with C++.
    np.divide gives fp with 3/2.
    // operator in Python is floor division, so -5 // 2 = -3 while in C++ -5 / 2 = -2.
    note that we don't use np.int_(a/b) because this involves floating point division
    which cause non-deterministic behavior for parallel processing.
    Hence, we use a // b if a * b > 0 else (a + (-a % b)) // b.
    """
    return np.where(a * b > 0, a // b, (a + (-a % b)) // b)
