from .pkt_mat import PktMat
from ..utils.utils_calc import truncate_divide
from typing import Tuple
import numpy as np


def pocket_tanh(mat_in: PktMat, k: int, num_items: int) -> Tuple[PktMat, PktMat]:
    mat_out: PktMat = PktMat(mat_in.row, mat_in.col)
    mat_actv_grad_inv: PktMat = PktMat(mat_in.row, mat_in.col)
    yMax = 2 ** (k - 1) - 1
    yMin = -yMax
    joints = [-127, -74, -31, 32, 75, 128]
    divisor = (1 << k) * num_items
    slopesInv = [yMax, 8, 2, 1, 2, 8, yMax]

    x = truncate_divide(mat_in.mat, divisor)
    conditions = [
        x < joints[0],
        (x >= joints[0]) & (x < joints[1]),
        (x >= joints[1]) & (x < joints[2]),
        (x >= joints[2]) & (x < joints[3]),
        (x >= joints[3]) & (x < joints[4]),
        (x >= joints[4]) & (x < joints[5]),
        x >= joints[5]
    ]
    calculations = [
        np.full(mat_in.mat.shape, yMin, dtype=int),
        truncate_divide(x, 4) - 88,
        x - 32,
        2 * x,
        x + 32,
        truncate_divide(x, 4) + 88,
        np.full(mat_in.mat.shape, yMax, dtype=int)
    ]
    mat_out.mat = np.select(conditions, calculations)
    mat_actv_grad_inv.mat = np.select(conditions, slopesInv)

    return mat_out, mat_actv_grad_inv


def activate(mat_in: PktMat, actv, k: int, num_items: int) -> Tuple[PktMat, PktMat]:
    """Returns the activated matrix and the inverse of the gradient of the activation function."""
    if actv == 'pocket_tanh':
        return pocket_tanh(mat_in, k, num_items)
