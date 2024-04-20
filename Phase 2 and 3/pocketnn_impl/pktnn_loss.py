from pktnn_mat import PktMat


def batch_l2_loss_delta(y_mat: PktMat, y_hat_mat: PktMat):
    # y : target
    # y_hat : prediction
    assert y_mat.dims_equal(y_hat_mat)
    loss_delta_mat = PktMat(y_mat.row, y_mat.col, y_hat_mat.mat - y_mat.mat)

    loss_delta_mat.mat = y_hat_mat.mat - y_mat.mat
    return loss_delta_mat
