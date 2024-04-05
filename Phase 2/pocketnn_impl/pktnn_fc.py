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
        self.input = x
        # if isinstance(x, PktLayer):
        #     assert self.next_layer is not x
        #     return self.forward(x.get_output_for_fc())

        inter = mat_mul_mat(x, self.weight)

        inter.self_add_mat(self.bias)
        activate(self.output, inter, self.actv_grad_inv, self.activation, K_BIT, self.in_dim)  # sets self.output

        if self.next_layer is not None:
            self.next_layer.forward(self.output)


    def set_random_dfa_weight(self, r, c):
        self.dfa_weight.init_zeros(r, c)
        weight_range = np.int_(np.floor(np.sqrt((12 * SHRT_MAX) / (self.in_dim + self.out_dim))))
        self.dfa_weight.set_random(False, -weight_range, weight_range)


    def compute_deltas(self, final_layer_delta_mat):
        # Handling WITHOUT batch normalization
        if self.next_layer is None:
            # Assuming last_layer_delta_mat is lossDelta for the last layer
            deltas = mat_elem_div_mat(final_layer_delta_mat, self.actv_grad_inv)
        else:
            assert self.use_dfa
            if not ((self.dfa_weight.row == final_layer_delta_mat.col) and
                    (self.dfa_weight.col == self.weight.col)):  # if uninitialized, set random
                self.set_random_dfa_weight(final_layer_delta_mat.col, self.weight.col)
            deltas = mat_mul_mat(final_layer_delta_mat, self.dfa_weight)
            deltas.self_elem_div_mat(self.actv_grad_inv)

        print(f'final_layer_delta_mat: {final_layer_delta_mat.sum()}, self.actv_grad_inv: {self.actv_grad_inv.sum()}, self.dfa_weight: {self.dfa_weight.sum()}, deltas: {deltas.sum()}')
        # print('final_layer_delta_mat:')
        # final_layer_delta_mat.print()
        # print('self.actv_grad_inv:')
        # self.actv_grad_inv.print()
        # print('self.dfa_weight:')
        # self.dfa_weight.print()
        # print('deltas:')
        # deltas.print()

        return deltas

    def backward(self, final_layer_delta_mat, lr_inv):
        deltas = self.compute_deltas(final_layer_delta_mat)
        print('deltas:', deltas.sum())
        # deltas.print()

        batch_size = deltas.row

        if self.prev_layer is None:
            prev_output_transpose = transpose_of(self.input)
        else:
            prev_output_transpose = transpose_of(self.prev_layer.get_output_for_fc())
        print('prev_output_transpose:', prev_output_transpose.sum())
        # prev_output_transpose.print()

        # Update weights
        weight_update = mat_mul_mat(prev_output_transpose, deltas)
        print('weight_update:', weight_update.sum())
        weight_update.self_div_const(-lr_inv)
        print('weight_update2:', weight_update.sum())
        self.weight.self_add_mat(weight_update)
        print('weight:', self.weight.sum())

        # Update bias
        assert not self.use_bn
        all_one_mat = PktMat.fill(row=1, col=batch_size, value=1)
        bias_update = mat_mul_mat(all_one_mat, deltas)
        print('bias_update:', bias_update.sum())
        bias_update.self_div_const(-lr_inv)
        print('bias_update2:', bias_update.sum())
        self.bias.self_add_mat(bias_update)
        print('bias:', self.bias.sum())

        self.weight.clamp_mat(SHRT_MIN, SHRT_MAX)
        self.bias.clamp_mat(SHRT_MIN, SHRT_MAX)

        if self.prev_layer is not None:
            self.prev_layer.backward(final_layer_delta_mat, lr_inv)

        return self