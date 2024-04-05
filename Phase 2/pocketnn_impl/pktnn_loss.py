from pktnn_mat import PktMat


def batch_l2_loss_delta(y_mat: PktMat, y_hat_mat: PktMat):
    # y : target
    # y_hat : prediction
    assert y_mat.dims_equal(y_hat_mat)
    loss_delta_mat = PktMat.fill(y_mat.row, y_mat.col, 0)

    # accum_loss_delta = 0

    for r in range(y_mat.row):
        for c in range(y_mat.col):
            diff = y_hat_mat[r, c] - y_mat[r, c]
            loss_delta_mat[r, c] = diff
            # accum_loss_delta += diff
    return loss_delta_mat