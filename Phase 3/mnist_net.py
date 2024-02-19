import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat
from pktnn_consts import UNSIGNED_4BIT_MAX
from pktnn_loss import batch_l2_loss_delta

class MNISTNet:
    def __init__(self):
        num_classes = 10
        mnist_rows = 28
        mnist_cols = 28

        dim_input = mnist_rows * mnist_cols

        fc_dim1 = 100
        fc_dim2 = 50
        activation = 'pocket_tanh'

        fc1 = PktFc(dim_input, fc_dim1, use_dfa=True, activation=activation)
        fc2 = PktFc(fc_dim1, fc_dim2, use_dfa=True, activation=activation)
        fc_last = PktFc(fc_dim2, num_classes, use_dfa=True, activation=activation)

        fc1.set_next_layer(fc2)
        fc2.set_next_layer(fc_last)

        self.fc1 = fc1
        self.fc2 = fc2
        self.fc_last = fc_last
