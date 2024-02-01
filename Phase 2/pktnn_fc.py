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
