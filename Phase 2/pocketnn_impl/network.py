from abc import abstractmethod

import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat
from state import save_state, load_state
from typing import List


class PktNet:
    @abstractmethod
    def forward(self, x: PktMat) -> PktMat:
        """Given input x, return the output of the network."""
        pass

    @abstractmethod
    def backward(self, loss_delta_mat: PktMat, lr_inv: np.int_):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def get_fc_list(self) -> List[PktFc]:
        pass


class MNISTNet(PktNet):
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

    def forward(self, x: PktMat) -> PktMat:
        self.fc1.forward(x)
        return self.fc_last.output

    def backward(self, loss_delta_mat: PktMat, lr_inv: np.int_):
        self.fc_last.backward(loss_delta_mat, lr_inv)

    def save(self, filename: str):
        save_state([self.fc1, self.fc2, self.fc_last], filename)

    def load(self, filename: str):
        load_state([self.fc1, self.fc2, self.fc_last], filename)

    def get_fc_list(self) -> List[PktFc]:
        return [self.fc1, self.fc2, self.fc_last]


class FashionMNISTNet(PktNet):
    def __init__(self):
        num_classes = 10
        mnist_rows = 28
        mnist_cols = 28

        dim_input = mnist_rows * mnist_cols

        fc_dim1 = 200
        fc_dim2 = 100
        fc_dim3 = 50
        activation = 'pocket_tanh'

        fc1 = PktFc(dim_input, fc_dim1, use_dfa=True, activation=activation)
        fc2 = PktFc(fc_dim1, fc_dim2, use_dfa=True, activation=activation)
        fc3 = PktFc(fc_dim2, fc_dim3, use_dfa=True, activation=activation)
        fc_last = PktFc(fc_dim3, num_classes, use_dfa=True, activation=activation)

        fc1.set_next_layer(fc2)
        fc2.set_next_layer(fc3)
        fc3.set_next_layer(fc_last)

        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3
        self.fc_last = fc_last

    def forward(self, x: PktMat) -> PktMat:
        self.fc1.forward(x)
        return self.fc_last.output

    def backward(self, loss_delta_mat: PktMat, lr_inv: np.int_):
        self.fc_last.backward(loss_delta_mat, lr_inv)

    def save(self, filename: str):
        save_state([self.fc1, self.fc2, self.fc3, self.fc_last], filename)

    def load(self, filename: str):
        load_state([self.fc1, self.fc2, self.fc3, self.fc_last], filename)

    def get_fc_list(self) -> List[PktFc]:
        return [self.fc1, self.fc2, self.fc3, self.fc_last]