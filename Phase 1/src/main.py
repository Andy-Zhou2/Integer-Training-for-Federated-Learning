import numpy as np
import torch
from torchvision import datasets, transforms
from quantize.quantized_number import QuantizedArray, compute_M0repr_and_n

model_ckpt = torch.load('data/model_ckpt/mnist_cnn.pt', map_location=torch.device('cpu'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_train = datasets.MNIST('./data', train=True,
                               transform=transform, download=True)


def get_q_params_from_range(min_value: float, max_value: float, bit_width: int):
    """
    Calculates the scale and zero_point of the quantized numbers.
    First calculate the scale, and re-adjust the scale to ensure max/min values are representable.
    :param min_value: The minimum value observed
    :param max_value: The maximum value observed
    :param bit_width: Bit-width used
    :return: scale and zero_point of the quantized numbers
    """
    assert min_value <= max_value, 'min_value must be less than max_value'
    if min_value > 0:
        min_value = 0
    if max_value < 0:
        max_value = 0

    scale = (max_value - min_value) / (2 ** bit_width - 1)
    unsigned_zero_point = np.round(-min_value / scale).astype(np.int_)

    if unsigned_zero_point == 0:
        scale = max_value / (2 ** bit_width - 1)
    elif unsigned_zero_point == 2 ** bit_width - 1:
        scale = -min_value / (2 ** bit_width - 1)
    else:
        scale = max(-min_value / unsigned_zero_point, max_value / (2 ** bit_width - 1 - unsigned_zero_point))

    zero_point = unsigned_zero_point - 2 ** (bit_width - 1)

    return scale, zero_point


def get_per_channel_q_params(weight):
    """
    Calculates the per-channel scale and zero_point of the quantized weight (or bias).
    :param weight: The weight (or bias) to be quantized
    :return: (array of scales, array of zero_points)
    """
    num_channels = weight.shape[0]

    scales = np.empty(num_channels, dtype=np.float64)
    zero_points = np.empty(num_channels, dtype=np.int64)
    for ch in range(num_channels):
        scales[ch], zero_points[ch] = get_q_params_from_range(weight[ch].min().item(),
                                                              weight[ch].max().item(),
                                                              bit_width=8)
    return scales, zero_points


data = dataset_train[0][0]
input_bit_width = 8
input_scale, input_zero_point = get_q_params_from_range(min_value=-0.4242129623889923,
                                                        max_value=2.821486711502075,
                                                        bit_width=input_bit_width)
data = QuantizedArray.quantize_per_tensor(data, input_scale, input_zero_point, input_bit_width)

conv1_weight = model_ckpt['conv1.weight']
conv1_weight_bit_width = 8
conv1_weight_scale, conv1_weight_zero_point = get_per_channel_q_params(conv1_weight)
conv1_weight = QuantizedArray.quantize_per_channel(conv1_weight, conv1_weight_scale, conv1_weight_zero_point,
                                                   conv1_weight_bit_width)

conv1_bias = model_ckpt['conv1.bias']
print(f'Maximum bias: {conv1_bias.max()}, Minimum bias: {conv1_bias.min()}')
conv1_bias_bit_width = 32
conv1_bias_scale = input_scale * conv1_weight_scale
conv1_bias_zero_point = np.zeros(conv1_bias.shape, dtype=np.int64)
conv1_bias = QuantizedArray.quantize_per_channel(conv1_bias, conv1_bias_scale, conv1_bias_zero_point,
                                                 conv1_bias_bit_width)
# sanity check
print(f'Scale: {conv1_bias_scale}')
print(f'Maximum value represented: {conv1_bias_scale * (2 ** (conv1_bias_bit_width - 1) - 1)}')

act1_bit_width = 8
act1_scale, act1_zero_point = get_q_params_from_range(min_value=0,  # simulate relu
                                                      max_value=3.0551493167877197,
                                                      bit_width=act1_bit_width)

act1_M0repr, act1_n = compute_M0repr_and_n(input_scale, conv1_weight_scale, act1_scale)

act1 = data.conv2d(conv1_weight, conv1_bias, act1_scale, act1_zero_point, act1_bit_width, act1_M0repr, act1_n)
print(act1.dequantize())

