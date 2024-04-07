import numpy as np


def truncate_divide(a, b):
    """
    Truncate division of a by b, which agrees with C++.
    np.divide gives fp with 3/2.
    // operator in Python is floor division, so -5 // 2 = -3 while in C++ -5 / 2 = -2.
    """
    return np.int_(a / b)
