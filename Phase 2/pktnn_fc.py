from pktnn_layer import PktLayer
from pktnn_mat import PktMat

class PktFc(PktLayer):
    def __init__(self, in_dim, out_dim, use_dfa=True, activation='pocket_tanh', use_bn=False):
        super().__init__()
        self.layer_type = PktLayer.LayerType.POCKET_FC
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = PktMat(in_dim, out_dim)
        self.bias = PktMat(1, out_dim)
        self.weight_update = PktMat(in_dim, out_dim)
        self.bias_update = PktMat(1, out_dim)

        self.prev_layer = None
        self.next_layer = None
        self.input = None

        self.use_dfa = use_dfa
        self.activation = activation
        self.use_bn = use_bn

    def set_next_layer(self, layer):
        self.next_layer = layer
        layer.prev_layer = self
        return self

