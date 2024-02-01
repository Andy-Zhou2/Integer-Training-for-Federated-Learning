from pktnn_layer import PktLayer
from pktnn_mat import PktMat
from pktnn_actv import activate
from pktnn_consts import *


class PktFc(PktLayer):
    def __init__(self, in_dim, out_dim, use_dfa=True, activation='pocket_tanh', use_bn=False):
        super().__init__()
        self.layer_type = PktLayer.LayerType.POCKET_FC
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.weight = PktMat(in_dim, out_dim)
        self.bias = PktMat(1, out_dim)
        self.inter = PktMat()
        self.deltas = PktMat()
        self.deltas_transpose = PktMat()
        self.d_actv_transpose = PktMat()
        self.actv_grad_inv = PktMat()
        self.weight_update = PktMat(in_dim, out_dim)
        self.bias_update = PktMat(1, out_dim)

        self.use_bn = use_bn
        self.mean = PktMat()
        self.variance = PktMat()
        self.stdev_with_eps = PktMat()
        self.standardized = PktMat()
        self.gamma = PktMat()
        self.beta = PktMat()
        self.batch_normalized = PktMat()
        self.d_gamma = PktMat()
        self.d_beta = PktMat()
        self.d_bn = PktMat()
        self.gamma_update = PktMat()
        self.beta_update = PktMat()

        self.use_dfa = use_dfa
        self.dfa_weight = PktMat()

        self.name = "fc_noname"
        self.activation = activation

        self.output = PktMat()
        self.rowss = self.in_dim
        self.colss = self.out_dim

        self.input = PktMat()

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
        if isinstance(x, PktLayer):
            assert self.next_layer is not x
            return self.forward(x.get_output_for_fc())

        print('forward: ', x.shape, self.weight.shape)
        self.inter.mat_mul_mat(x, self.weight)

        if self.use_bn:
            self.batch_normalization()
            activate(self.output, self.batch_normalized, self.actv_grad_inv, self.activation, K_BIT, self.in_dim)
        else:
            self.inter.self_add_mat(self.bias)
            activate(self.output, self.inter, self.actv_grad_inv, self.activation, K_BIT, self.in_dim)

        if self.next_layer is not None:
            self.next_layer.forward(self)
        return self

    def set_random_dfa_weight(self, in_dim, out_dim):
        raise NotImplementedError

    def compute_deltas(self, last_layer_delta_mat, lr_inv):
        if self.use_bn:
            # Step 1: Calculate d_bn (self.d_bn) if it's the last layer or not
            if self.next_layer is None:
                # Assuming last_layer_delta_mat is lossDelta for the last layer
                self.d_bn.mat_elem_div_mat(last_layer_delta_mat, self.actv_grad_inv)
            else:
                # Calculate deltas for a layer with batch normalization
                self.d_actv_transpose.mat_mul_mat(self.next_layer.weight, self.next_layer.deltas_transpose)
                self.d_bn.transpose_of(self.d_actv_transpose)
                self.d_bn.self_elem_div_mat(self.actv_grad_inv)

            # Step 2 & 3: Calculate d_gamma (self.d_gamma) and d_beta (self.d_beta)
            num_items = self.d_bn.row  # N
            feature_dims = self.d_bn.col  # Dk
            self.d_gamma.init_zeros(1, feature_dims)
            self.d_beta.init_zeros(1, feature_dims)

            for c in range(feature_dims):
                for r in range(num_items):
                    self.d_gamma[0][c] += self.d_bn[r, c] * self.standardized[r, c]
                    self.d_beta[0][c] += self.d_bn[r, c]

            # Step 4: Calculate mDeltas (self.deltas) with the complex formula
            gamma_stdev = PktMat()
            gamma_stdev.mat_elem_div_mat(self.gamma, self.stdev_with_eps)

            d_gamma_xhat = PktMat()
            d_gamma_xhat.mat_elem_mul_mat(self.d_gamma, self.standardized)
            d_gamma_xhat.self_mul_const(-1)

            d_bn_times_n = PktMat()
            d_bn_times_n.mat_mul_const(self.d_bn, num_items)

            one_column_vec = PktMat()
            one_column_vec.reset_all(num_items, 1, 1)

            d_beta_matrix = PktMat()
            d_beta_matrix.mat_mul_mat(one_column_vec, self.d_beta)
            d_beta_matrix.self_mul_const(-1)

            self.deltas.init_zeros(num_items, feature_dims, 0)
            self.deltas.mat_add_mat(d_gamma_xhat, d_bn_times_n)
            self.deltas.self_add_mat(d_beta_matrix)
            self.deltas.mat_elem_mul_self(gamma_stdev)
            self.deltas.self_div_const(num_items)
        else:
            # Handling without batch normalization
            if self.next_layer is None:
                # Assuming last_layer_delta_mat is lossDelta for the last layer
                self.deltas.mat_elem_div_mat(last_layer_delta_mat, self.actv_grad_inv)
            else:
                if self.use_dfa:
                    if not ((self.dfa_weight.row == last_layer_delta_mat.col) and
                            (self.dfa_weight.col == self.weight.col)):
                        self.set_random_dfa_weight(last_layer_delta_mat.col, self.weight.col)
                    self.deltas.mat_mul_mat(last_layer_delta_mat, self.dfa_weight)
                    self.deltas.self_elem_div_mat(self.actv_grad_inv)
                else:
                    # Vanilla backpropagation for hidden or first layer
                    self.d_actv_transpose.mat_mul_mat(self.next_layer.weight, self.next_layer.deltas_transpose)
                    self.deltas.transpose_of(self.d_actv_transpose)
                    self.deltas.self_elem_div_mat(self.actv_grad_inv)


        self.deltas_transpose.transpose_of(self.deltas)
        return self

    def backward(self, last_layer_delta_mat, lr_inv):
        self.compute_deltas(last_layer_delta_mat, lr_inv)

        batch_size = self.deltas.row

        prev_output_transpose = PktMat()

        if self.prev_layer is None:
            prev_output_transpose.transpose_of(self.input)
        else:
            prev_output_transpose.transpose_of(self.prev_layer.get_output_for_fc())

        self.weight_update.mat_mul_mat(prev_output_transpose, self.deltas)
        self.weight_update.self_div_const(-lr_inv)
        self.weight.self_add_mat(self.weight_update)

        if self.use_bn:
            self.gamma_update.mat_div_const(self.d_gamma, -lr_inv)
            self.gamma.self_add_mat(self.gamma_update)

            self.beta_update.mat_div_const(self.d_beta, -lr_inv)
            self.beta.self_add_mat(self.beta_update)
        else:
            all_one_mat = PktMat()
            all_one_mat.reset_all(1, batch_size, 1)
            self.bias_update.mat_mul_mat(all_one_mat, self.deltas)
            self.bias_update.self_div_const(-lr_inv)
            self.bias.self_add_mat(self.bias_update)

        self.weight.clamp_mat(-32768, 32767)
        self.bias.clamp_mat(-32768, 32767)

        if self.prev_layer is not None:
            self.prev_layer.backward(last_layer_delta_mat, lr_inv)

        return self