import numpy as np
import torch
from torchvision import datasets, transforms
from quantize.quantized_number import QuantizedArray, compute_M0repr_and_n
import time
import os

# model_ckpt = torch.load('data/model_ckpt/mnist_cnn.pt', map_location=torch.device('cpu'))
model_ckpt = torch.load(r'../../MNIST/model_ckpt/mnist_2_4.pt', map_location=torch.device('cpu'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_test = datasets.MNIST('./data', train=False,
                              transform=transform, download=True)


def get_q_params_from_range(min_value: float, max_value: float, bit_width: int):
    """
    Calculates the scale and zero_point of the quantized numbers.
    First calculate the scale based on (max-min)/range.
    Then, re-adjust the scale to ensure 0 is exactly representable and max/min values are within representable range.

    :param min_value: The minimum value observed
    :param max_value: The maximum value observed
    :param bit_width: Bit-width used
    :return: scale and zero_point of the quantized numbers
    """
    assert min_value <= max_value, 'min_value must be less than max_value'

    # must cover 0 in the range
    if min_value > 0:
        min_value = 0
    if max_value < 0:
        max_value = 0

    scale = (max_value - min_value) / (2 ** bit_width - 1)
    unsigned_zero_point = np.round(-min_value / scale).astype(np.int64)

    if unsigned_zero_point == 0:
        scale = max_value / (2 ** bit_width - 1)
    elif unsigned_zero_point == 2 ** bit_width - 1:
        scale = -min_value / (2 ** bit_width - 1)
    else:
        scale = max(-min_value / unsigned_zero_point, max_value / (2 ** bit_width - 1 - unsigned_zero_point))

    zero_point = unsigned_zero_point - 2 ** (bit_width - 1)  # convert from unsigned to signed

    return scale, zero_point


def get_per_channel_q_params(weight, bit_width):
    """
    Calculates the per-channel scale and zero_point of the quantized weight (or bias).
    :param bit_width: The bit-width used
    :param weight: The weight (or bias) to be quantized
    :return: (array of scales, array of zero_points)
    """
    num_channels = weight.shape[0]

    scales = np.empty(num_channels, dtype=np.float64)
    zero_points = np.empty(num_channels, dtype=np.int64)
    for ch in range(num_channels):
        scales[ch], zero_points[ch] = get_q_params_from_range(weight[ch].min().item(),
                                                              weight[ch].max().item(),
                                                              bit_width=bit_width)
    return scales, zero_points

def infer(data, bit_width_config):
    input_bit_width = bit_width_config['input']
    input_scale, input_zero_point = get_q_params_from_range(min_value=-0.4242129623889923,
                                                            max_value=2.821486711502075,
                                                            bit_width=input_bit_width)
    data = QuantizedArray.quantize_per_tensor(data, input_scale, input_zero_point, input_bit_width)

    conv1_weight = model_ckpt['conv1.weight']
    conv1_weight_bit_width = bit_width_config['conv1_weight']
    conv1_weight_scale, conv1_weight_zero_point = get_per_channel_q_params(conv1_weight, conv1_weight_bit_width)
    conv1_weight = QuantizedArray.quantize_per_channel(conv1_weight, conv1_weight_scale, conv1_weight_zero_point,
                                                       conv1_weight_bit_width)

    conv1_bias = model_ckpt['conv1.bias']
    conv1_bias_bit_width = bit_width_config['conv1_bias']
    conv1_bias_scale = input_scale * conv1_weight_scale
    conv1_bias_zero_point = np.zeros(conv1_bias.shape, dtype=np.int64)
    conv1_bias = QuantizedArray.quantize_per_channel(conv1_bias, conv1_bias_scale, conv1_bias_zero_point,
                                                     conv1_bias_bit_width)

    act1_bit_width = bit_width_config['act1']
    act1_scale, act1_zero_point = get_q_params_from_range(min_value=0,  # simulate relu
                                                          max_value=3.0551493167877197,
                                                          bit_width=act1_bit_width)
    act1_M0repr, act1_n = compute_M0repr_and_n(input_scale, conv1_weight_scale, act1_scale)

    act1 = data.conv2d(conv1_weight, conv1_bias, act1_scale, act1_zero_point, act1_bit_width, act1_M0repr, act1_n)

    conv2_weight = model_ckpt['conv2.weight']
    conv2_weight_bit_width = bit_width_config['conv2_weight']
    conv2_weight_scale, conv2_weight_zero_point = get_per_channel_q_params(conv2_weight, conv2_weight_bit_width)
    conv2_weight = QuantizedArray.quantize_per_channel(conv2_weight, conv2_weight_scale, conv2_weight_zero_point,
                                                       conv2_weight_bit_width)

    conv2_bias = model_ckpt['conv2.bias']
    conv2_bias_bit_width = bit_width_config['conv2_bias']
    conv2_bias_scale = act1_scale * conv2_weight_scale
    conv2_bias_zero_point = np.zeros(conv2_bias.shape, dtype=np.int64)
    conv2_bias = QuantizedArray.quantize_per_channel(conv2_bias, conv2_bias_scale, conv2_bias_zero_point,
                                                     conv2_bias_bit_width)

    act2_bit_width = bit_width_config['act2']
    act2_scale, act2_zero_point = get_q_params_from_range(min_value=0,  # simulate relu
                                                          max_value=4.931317329406738,
                                                          bit_width=act2_bit_width)
    act2_M0repr, act2_n = compute_M0repr_and_n(act1_scale, conv2_weight_scale, act2_scale)

    act2 = act1.conv2d(conv2_weight, conv2_bias, act2_scale, act2_zero_point, act2_bit_width, act2_M0repr, act2_n)

    pooled_act2 = act2.maxpool2d(2)

    act2_flattened = pooled_act2.flatten()

    fc1_weight = model_ckpt['fc1.weight']
    fc1_weight_bit_width = bit_width_config['fc1_weight']
    fc1_weight_scale, fc1_weight_zero_point = get_per_channel_q_params(fc1_weight, fc1_weight_bit_width)
    fc1_weight = QuantizedArray.quantize_per_channel(fc1_weight, fc1_weight_scale, fc1_weight_zero_point,
                                                       fc1_weight_bit_width)

    fc1_bias = model_ckpt['fc1.bias']
    fc1_bias_bit_width = bit_width_config['fc1_bias']
    fc1_bias_scale = act2_flattened.scale * fc1_weight_scale
    fc1_bias_zero_point = np.zeros(fc1_bias.shape, dtype=np.int64)
    fc1_bias = QuantizedArray.quantize_per_channel(fc1_bias, fc1_bias_scale, fc1_bias_zero_point,
                                                     fc1_bias_bit_width)

    act3_bit_width = bit_width_config['act3']
    act3_scale, act3_zero_point = get_q_params_from_range(min_value=0,  # simulate relu
                                                          max_value=9.289616584777832,
                                                          bit_width=act3_bit_width)
    act3_M0repr, act3_n = compute_M0repr_and_n(act2_flattened.scale, fc1_weight_scale, act3_scale)

    act3 = act2_flattened.linear(fc1_weight, fc1_bias, act3_scale, act3_zero_point, act3_bit_width, act3_M0repr, act3_n)

    fc2_weight = model_ckpt['fc2.weight']
    fc2_weight_bit_width = bit_width_config['fc2_weight']
    fc2_weight_scale, fc2_weight_zero_point = get_per_channel_q_params(fc2_weight, fc2_weight_bit_width)
    fc2_weight = QuantizedArray.quantize_per_channel(fc2_weight, fc2_weight_scale, fc2_weight_zero_point,
                                                       fc2_weight_bit_width)

    fc2_bias = model_ckpt['fc2.bias']
    fc2_bias_bit_width = bit_width_config['fc2_bias']
    fc2_bias_scale = act3.scale * fc2_weight_scale
    fc2_bias_zero_point = np.zeros(fc2_bias.shape, dtype=np.int64)
    fc2_bias = QuantizedArray.quantize_per_channel(fc2_bias, fc2_bias_scale, fc2_bias_zero_point,
                                                     fc2_bias_bit_width)

    act4_bit_width = bit_width_config['act4']
    act4_scale, act4_zero_point = get_q_params_from_range(min_value=-37.50343322753906,  # no relu
                                                          max_value=10.646228790283203,
                                                          bit_width=act4_bit_width)
    act4_M0repr, act4_n = compute_M0repr_and_n(act3.scale, fc2_weight_scale, act4_scale)

    act4 = act3.linear(fc2_weight, fc2_bias, act4_scale, act4_zero_point, act4_bit_width, act4_M0repr, act4_n)
    return act4

if __name__ == '__main__':
    bit_width_config = {
        'input': 8,
        'conv1_weight': 8,
        'conv1_bias': 32,
        'act1': 8,
        'conv2_weight': 8,
        'conv2_bias': 32,
        'act2': 8,
        'fc1_weight': 8,
        'fc1_bias': 32,
        'act3': 8,
        'fc2_weight': 8,
        'fc2_bias': 32,
        'act4': 8,
    }

    for bw in [4,5,6,7]:
        total_test = 0
        correct_test = 0
        for key in bit_width_config:
            if 'bias' not in key:
                bit_width_config[key] = bw
        print('bit_width_config', bit_width_config)

        for data, answer in dataset_test:
            model_result = infer(data, bit_width_config).array.argmax()
            # print(answer, model_result)

            if model_result == answer:
                correct_test += 1
            total_test += 1

            if total_test % 100 == 0:
                print(f'Accuracy: {correct_test / total_test * 100:.2f}%, Correct: {correct_test}, Total: {total_test}')
