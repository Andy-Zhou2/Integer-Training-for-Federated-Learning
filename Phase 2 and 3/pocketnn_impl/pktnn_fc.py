from pktnn_layer import PktLayer
from pktnn_mat import PktMat, mat_elem_div_mat, mat_mul_const, mat_elem_mul_mat, transpose_of, \
    mat_div_const, mat_add_mat, deep_copy, mat_mul_mat
from pktnn_actv import activate
from pktnn_consts import *
import numpy as np


class PktFc(PktLayer):
    def __init__(self, in_dim, out_dim, use_dfa=True, activation='pocket_tanh', use_bn=False):
        super().__init__()
        self.layer_type = PktLayer.LayerType.POCKET_FC
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.weight = PktMat(in_dim, out_dim)
        self.bias = PktMat(1, out_dim)

        self.actv_grad_inv = PktMat()

        self.use_bn = use_bn

        self.use_dfa = use_dfa
        self.dfa_weight = PktMat()

        # self.name = "fc_noname"
        self.activation = activation

        self.output = PktMat()
        self.rowss = self.in_dim
        self.colss = self.out_dim

        self.input = PktMat()

    def __repr__(self):
        return f"FC Layer: {self.in_dim} -> {self.out_dim}"

    def batch_normalization(self):
        raise NotImplementedError

    def set_next_layer(self, layer):
        self.next_layer = layer
        layer.prev_layer = self
        return self

    def get_output_for_fc(self):
        return self.output

    def forward(self, x):
        """Note that input, output and actv_grad_inv are set and used in the backwards process"""
        self.input = x

        inter = mat_mul_mat(x, self.weight)
        inter.self_add_mat(self.bias)
        self.output, self.actv_grad_inv = activate(inter, self.activation, K_BIT, self.in_dim)

        if self.next_layer is not None:
            self.next_layer.forward(self.output)

    def set_random_dfa_weight(self, r, c):
        self.dfa_weight.init_zeros(r, c)
        weight_range = np.int_(np.floor(np.sqrt((12 * SHRT_MAX) / (self.in_dim + self.out_dim))))
        self.dfa_weight.set_random(False, -weight_range, weight_range)

    def compute_deltas(self, final_layer_delta_mat):
        # Handling WITHOUT batch normalization
        if self.next_layer is None:
            deltas = mat_elem_div_mat(final_layer_delta_mat, self.actv_grad_inv)
        else:
            assert self.use_dfa
            if not ((self.dfa_weight.row == final_layer_delta_mat.col) and
                    (self.dfa_weight.col == self.weight.col)):  # if uninitialized, set random
                self.set_random_dfa_weight(final_layer_delta_mat.col, self.weight.col)
            deltas = mat_mul_mat(final_layer_delta_mat, self.dfa_weight)
            deltas.self_elem_div_mat(self.actv_grad_inv)

        return deltas

    def backward(self, final_layer_delta_mat, lr_inv):
        deltas = self.compute_deltas(final_layer_delta_mat)

        batch_size = deltas.row

        if self.prev_layer is None:
            prev_output_transpose = transpose_of(self.input)
        else:
            prev_output_transpose = transpose_of(self.prev_layer.get_output_for_fc())

        # Update weights
        weight_update = mat_mul_mat(prev_output_transpose, deltas)
        weight_update.self_div_const(-lr_inv)
        self.weight.self_add_mat(weight_update)

        # Update bias
        assert not self.use_bn
        all_one_mat = PktMat.fill(row=1, col=batch_size, value=1)
        bias_update = mat_mul_mat(all_one_mat, deltas)
        bias_update.self_div_const(-lr_inv)
        self.bias.self_add_mat(bias_update)

        self.weight.clamp_mat(SHRT_MIN + 1, SHRT_MAX)
        self.bias.clamp_mat(SHRT_MIN + 1, SHRT_MAX)

        if self.prev_layer is not None:
            self.prev_layer.backward(final_layer_delta_mat, lr_inv)

        return self
