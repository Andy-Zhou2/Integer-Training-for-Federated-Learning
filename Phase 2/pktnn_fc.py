from pktnn_layer import PktLayer
from pktnn_mat import PktMat

class PktFc(PktLayer):
    def __init__(self, in_dim, out_dim, use_dfa=True, activation='pocket_tanh', use_bn=False):
        super().__init__()
        self.layer_type = PktLayer.LayerType.POCKET_FC
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.weight = PktMat(in_dim, out_dim)
        self.bias = PktMat(1, out_dim)
        self.inter = None
        self.deltas = None
        self.deltas_transpose = None
        self.d_actv_transpose = None
        self.actv_grad_inv = None
        self.weight_update = PktMat(in_dim, out_dim)
        self.bias_update = PktMat(1, out_dim)

        self.use_bn = use_bn
        self.mean = None
        self.variance = None
        self.stdev_with_eps = None
        self.standardized = None
        self.gamma = None
        self.beta = None
        self.batch_normalized = None
        self.d_gamma = None
        self.d_beta = None
        self.d_bn = None
        self.gamma_update = None
        self.beta_update = None

        self.use_dfa = use_dfa
        self.dfa_weight = None

        self.name = "fc_noname"
        self.activation = activation

        self.output = None
        self.rowss = self.in_dim
        self.colss = self.out_dim


        self.input = None



    def set_next_layer(self, layer):
        self.next_layer = layer
        layer.prev_layer = self
        return self

