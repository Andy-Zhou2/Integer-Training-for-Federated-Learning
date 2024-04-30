from .pkt_layer import PktLayer
from .pkt_mat import PktMat, mat_elem_div_mat, mat_mul_const, mat_elem_mul_mat, transpose_of, \
    mat_div_const, mat_add_mat, deep_copy, mat_mul_mat
from .pkt_actv import activate
import numpy as np
import logging


class PktFc(PktLayer):
    def __init__(self, in_dim: int, out_dim: int, use_dfa: bool = True, activation: str = 'pocket_tanh',
                 use_bn: bool = False, weight_max_absolute: int = 32767, name: str = "fc_noname",
                 activation_k_bit: int = 8):
        super().__init__()
        self.layer_type = PktLayer.LayerType.POCKET_FC
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.weight = PktMat(in_dim, out_dim)
        self.bias = PktMat(1, out_dim)
        self.name = name
        self.weight_max_absolute = weight_max_absolute
        self.activation_k_bit = activation_k_bit

        self.use_bn = use_bn
        self.use_dfa = use_dfa
        self.dfa_weight = PktMat()

        self.actv_grad_inv = PktMat()
        self.output = PktMat()
        self.input = PktMat()
        self.activation = activation


    def __repr__(self):
        return f"FC Layer: {self.in_dim} -> {self.out_dim} with clip {self.weight_max_absolute}"

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

        def save_mat(mat, name):
            import os
            # layer_count = 1
            # while os.path.exists(f'./activations/layer_{layer_count}_{name}.npy'):
            #     layer_count += 1
            # np.save(f'./activations/layer_{layer_count}_{name}.npy', mat.mat)

        inter = mat_mul_mat(x, self.weight)
        # print(f'inter: {np.percentile(inter.mat, np.arange(0, 101, 50))}')
        save_mat(inter, 'inter')
        inter.self_add_mat(self.bias)
        # print(f'inter2: {np.percentile(inter.mat, np.arange(0, 101, 50))}')
        print(f'inter2: {max(abs(inter.mat.max()), abs(inter.mat.min()))}')
        save_mat(inter, 'inter2')
        self.output, self.actv_grad_inv = activate(inter, self.activation, self.activation_k_bit, self.in_dim)
        # print(f'self.output: {np.percentile(self.output.mat, np.arange(0, 101, 50))}')
        save_mat(inter, 'output')

        if self.next_layer is not None:
            self.next_layer.forward(self.output)

    def set_random_dfa_weight(self, r, c):
        self.dfa_weight.init_zeros(r, c)
        weight_range = np.int64(np.floor(np.sqrt((12 * self.weight_max_absolute) / (self.in_dim + self.out_dim))))
        if weight_range == 0:
            weight_range = 1
            logging.warning(f"Weight range is 0 for layer {self.name}. Setting to 1.")
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

        self.weight.clamp_mat(-self.weight_max_absolute, self.weight_max_absolute)
        self.bias.clamp_mat(-self.weight_max_absolute, self.weight_max_absolute)

        if self.prev_layer is not None:
            self.prev_layer.backward(final_layer_delta_mat, lr_inv)

        return self
