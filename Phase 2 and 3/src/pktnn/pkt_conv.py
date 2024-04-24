from .pkt_layer import PktLayer
from .pkt_mat import PktMat, mat_elem_div_mat, mat_mul_const, mat_elem_mul_mat, transpose_of, \
    mat_div_const, mat_add_mat, deep_copy, mat_mul_mat
from .pkt_actv import activate
import numpy as np
import logging
import torch
import torch.nn.functional as F
from typing import Union


def conv(A: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convolution operation by converting conv to matmul.
    Suppose A is of shape (bs, N, N) and K is of shape (K, K).
    The output should have shape (bs, T, T) where T = N - K + 1.
    K is converted to matrix M of shape (T^2, N^2) and A is converted to matrix V of shape (N^2, bs).
    """
    ...


class PktConv(PktLayer):
    """This modules uses numpy array instead of PktMat. """

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int,
                 use_dfa: bool = True, weight_max_absolute: int = 32767, name: str = "fc_noname",
                 ):
        super().__init__()
        self.layer_type = PktLayer.LayerType.POCKET_FC
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.weight = np.zeros((out_channel, in_channel, kernel_size, kernel_size))
        self.bias = np.zeros(out_channel)
        self.name = name
        self.weight_max_absolute = weight_max_absolute

        self.use_dfa = use_dfa
        self.dfa_weight = None

        self.input: Union[None, PktMat] = None
        self.input_4d_shape: Union[None, np.ndarray] = None
        self.output: Union[None, PktMat] = None
        self.output_4d_shape: Union[None, np.ndarray] = None


    def __repr__(self):
        return (f"Conv Layer: {self.out_channel, self.in_channel, self.kernel_size} "
                f"with clip {self.weight_max_absolute}")

    def set_next_layer(self, layer: PktLayer):
        self.next_layer = layer
        layer.prev_layer = self
        return self

    def get_output_for_fc(self):
        return self.output

    def forward(self, x: PktMat):
        """Note that input, output and actv_grad_inv are set and used in the backwards process"""
        self.input = x

        x = x.mat  # (bs, in_channel * n * n)
        image_side_length = int(x.shape[1] ** 0.5)
        assert image_side_length ** 2 == x.shape[1]
        x = x.reshape(x.shape[0], self.in_channel, image_side_length, image_side_length)
        self.input_4d_shape = x

        output = F.conv2d(torch.from_numpy(x), torch.from_numpy(self.weight), bias=torch.from_numpy(self.bias))
        # output of shape (bs, out_channel, t, t) where t = n - kernel_size + 1
        assert output.dtype == torch.int64
        output = output.numpy()
        self.output_4d_shape = output
        output = output.reshape(output.shape[0], -1)  # (bs, out_channel * t * t)
        self.output = PktMat(output.shape[0], output.shape[1], output)


        if self.next_layer is not None:
            self.next_layer.forward(self.output)

    def set_random_dfa_weight(self, r, c):
        self.dfa_weight.init_zeros(r, c)
        weight_range = np.int_(np.floor(np.sqrt((12 * self.weight_max_absolute) / (self.in_dim + self.out_dim))))
        if weight_range == 0:
            weight_range = 1
            logging.warning(f"Weight range is 0 for layer {self.name}. Setting to 1.")
        self.dfa_weight.set_random(False, -weight_range, weight_range)

    def compute_deltas(self, final_layer_delta_mat):
        if self.next_layer is None:
            raise NotImplementedError("Unexpected circumstance: Conv layer should not be the last layer.")

        assert self.use_dfa
        if self.dfa_weight is None:  # if uninitialized, set random
            self.set_random_dfa_weight(final_layer_delta_mat.col, self.output.shape[1])
        deltas = mat_mul_mat(final_layer_delta_mat, self.dfa_weight)

        return deltas

    def backward(self, final_layer_delta_mat, lr_inv):
        deltas = self.compute_deltas(final_layer_delta_mat)
        deltas = deltas.mat.reshape(self.output_4d_shape.shape)

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
