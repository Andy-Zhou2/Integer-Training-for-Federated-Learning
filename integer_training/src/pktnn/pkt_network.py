from abc import abstractmethod
import numpy as np
import re

from .pkt_fc import PktFc
from .pkt_mat import PktMat
from .pkt_state import save_state, load_state
from typing import List, Union


class PktNet:
    @abstractmethod
    def forward(self, x: PktMat) -> PktMat:
        """Given input x, return the output of the network."""
        pass

    @abstractmethod
    def backward(self, loss_delta_mat: PktMat, lr_inv: np.int32):
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

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the parameters of the network. Used for FL."""
        fc_list = self.get_fc_list()
        assert len(parameters) == len(fc_list) * 2, \
            "Number of parameters should be twice the number of layers (weight & bias)"
        for i in range(len(fc_list)):
            fc_list[i].weight.mat = parameters[i * 2]
            fc_list[i].bias.mat = parameters[i * 2 + 1]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the parameters of the network. Used for FL."""
        fc_list = self.get_fc_list()
        parameters = []
        for fc in fc_list:
            parameters.append(fc.weight.mat)
            parameters.append(fc.bias.mat)
        return parameters


class LinearNet(PktNet):
    def __init__(self, num_classes: int, dim_input: int, fc_dims: List[int], activation: str,
                 weight_clip_each_layer: Union[None, List[int]] = None):
        """
        Create a custom linear network. The network will have len(fc_dims) + 1 layers.
        :param num_classes:
        :param dim_input:
        :param fc_dims: The dimensions of the hidden layers.
        :param activation:
        :param weight_clip_each_layer: The maximum absolute value of the weights for each layer. If None, the weights
        will be clipped to a default range of [-32767, 32767].
        """
        self.fc_list = []

        all_dims = [dim_input] + fc_dims + [num_classes]
        for i in range(len(all_dims) - 1):
            clip_range = 32767 if weight_clip_each_layer is None else weight_clip_each_layer[i]
            fc = PktFc(all_dims[i], all_dims[i + 1], use_dfa=True, activation=activation,
                       weight_max_absolute=clip_range)
            self.fc_list.append(fc)

        for i in range(len(self.fc_list) - 1):
            self.fc_list[i].set_next_layer(self.fc_list[i + 1])

    def forward(self, x: PktMat) -> PktMat:
        self.fc_list[0].forward(x)
        return self.fc_list[-1].output

    def backward(self, loss_delta_mat: PktMat, lr_inv: np.int32):
        self.fc_list[-1].backward(loss_delta_mat, lr_inv)

    def save(self, filename: str):
        save_state(self.fc_list, filename)

    def load(self, filename: str):
        load_state(self.fc_list, filename)

    def get_fc_list(self) -> List[PktFc]:
        return self.fc_list

    def __repr__(self):
        return f'LinearNet({self.fc_list})'


def get_net(model_name: str) -> PktNet:
    """
    Model name consists of two parts, specifying the model and the clip ranges respectively.
    Model name could be mnist_default, fashion_mnist_default, or [200,100] with no space.
    Clip ranges are optional, and consists of a list of integers with no space.
    """
    pattern = r'^(\[[0-9,]*\]|mnist_default|fashion_mnist_default)( \[[0-9,]*\])?$'
    match = re.match(pattern, model_name)
    if not match:
        raise ValueError(f'Invalid model name: {model_name}')

    architecture, clip_ranges = match.groups()

    if clip_ranges is not None:
        clip_ranges = eval(clip_ranges)  # None if not provided

    if architecture == 'mnist_default':
        dims = [100, 50]
    elif architecture == 'fashion_mnist_default':
        dims = [200, 100, 50]
    else:
        dims = eval(architecture)

    if clip_ranges is not None:
        assert len(dims) == len(clip_ranges) - 1, "Number of clip ranges and layers do not match."

    return LinearNet(num_classes=10, dim_input=28 * 28, fc_dims=dims, activation='pocket_tanh',
                     weight_clip_each_layer=clip_ranges)


if __name__ == '__main__':
    test_strings = [
        "mnist_default",
        "fashion_mnist_default",
        "fashion_mnist_default [10,20,30]",
        "[1,2,3] [5,6,7,8]",
        "[] [10]",
        "[1,2,3]",
    ]

    for test in test_strings:
        print(f"Model name: {test}")
        net = get_net(test)
        print(net)
        print()
