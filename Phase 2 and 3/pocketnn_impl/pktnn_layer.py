from pktnn_mat import PktMat


class PktLayer:
    class LayerType:
        POCKET_FC = 1
        POCKET_CONV = 2

    def __init__(self):
        self.layer_type = None
        self.prev_layer = None
        self.next_layer = None
        self.dummy_3d = PktMat()

    def get_layer_type(self):
        return self.layer_type

    def get_prev_layer(self):
        return self.prev_layer

    def get_next_layer(self):
        return self.next_layer

    def set_prev_layer(self, layer):
        self.prev_layer = layer
        return self

    def set_next_layer(self, layer):
        self.next_layer = layer
        return self

    def forward(self, x):
        raise NotImplementedError

    def backward(self, last_deltas_mat, lr_inv):
        raise NotImplementedError

    def get_output_for_fc(self):
        raise NotImplementedError

    def get_output_for_conv(self):
        raise NotImplementedError