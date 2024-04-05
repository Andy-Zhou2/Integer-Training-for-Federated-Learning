import numpy as np
from pktnn_mat import PktMat
from pktnn_consts import *
from calc_util import truncate_divide

def pocket_tanh(mat_out: PktMat, mat_in: PktMat, mat_actv_grad_inv: PktMat, k: int, num_items: int):
    assert mat_out.dims_equal(mat_in)
    if not mat_actv_grad_inv.dims_equal(mat_in):
        mat_actv_grad_inv.init_zeros(mat_in.row, mat_in.col)
    yMax = PKT_MAX
    yMin = PKT_MIN
    joints = [-127, -74, -31, 32, 75, 128]
    divisor = (1 << k) * num_items
    slopesInv = [yMax, 8, 2, 1, 2, 8, yMax]

    for r in range(mat_out.row):
        for c in range(mat_out.col):
            x = truncate_divide(mat_in[r, c], divisor)
            if x < joints[0]:
                mat_out[r, c] = yMin
                mat_actv_grad_inv[r, c] = slopesInv[0]
            elif x < joints[1]:
                y = truncate_divide(x, 4) - 88
                mat_out[r, c] = y
                mat_actv_grad_inv[r, c] = slopesInv[1]
            elif x < joints[2]:
                y = x - 32
                mat_out[r, c] = y
                mat_actv_grad_inv[r, c] = slopesInv[2]
            elif x < joints[3]:
                y = 2 * x
                mat_out[r, c] = y
                mat_actv_grad_inv[r, c] = slopesInv[3]
            elif x < joints[4]:
                y = x + 32
                mat_out[r, c] = y
                mat_actv_grad_inv[r, c] = slopesInv[4]
            elif x < joints[5]:
                y = truncate_divide(x, 4) + 88
                mat_out[r, c] = y
                mat_actv_grad_inv[r, c] = slopesInv[5]
            else:
                mat_out[r, c] = yMax
                mat_actv_grad_inv[r, c] = slopesInv[6]
    return mat_out, mat_actv_grad_inv

def activate(mat_out: PktMat, mat_in: PktMat, mat_actv_grad_inv: PktMat, actv, k: int, num_items: int):
    """Set mat_out and mat_actv_grad_inv according to chosen activation"""
    if not mat_out.dims_equal(mat_in):
        mat_out.init_zeros(mat_in.row, mat_in.col)

    if actv == 'pocket_tanh':
        return pocket_tanh(mat_out, mat_in, mat_actv_grad_inv, k, num_items)
