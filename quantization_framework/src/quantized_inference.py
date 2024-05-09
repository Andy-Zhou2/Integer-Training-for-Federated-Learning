import numpy as np
import torch
from torchvision import datasets, transforms
from quantize.quantized_number import QuantizedArray, compute_M0repr_and_n
from collect_statistics import get_statistics
import copy
import hydra
from omegaconf import DictConfig

channel1 = 2
channel2 = 4

model_ckpt = torch.load(f'../model_ckpt/mnist_linear.pt', map_location=torch.device('cpu'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_test = datasets.MNIST('../data', train=False,
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


def infer(data, bit_width_config, min_max_config):
    input_bit_width = bit_width_config['input']
    input_scale, input_zero_point = get_q_params_from_range(min_value=min_max_config['input'][0],
                                                            max_value=min_max_config['input'][1],
                                                            bit_width=input_bit_width)
    data = QuantizedArray.quantize_per_tensor(data, input_scale, input_zero_point, input_bit_width)

    act2_flattened = data.flatten()

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
                                                          max_value=min_max_config['fc1'][1],
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
    act4_scale, act4_zero_point = get_q_params_from_range(min_value=0,  # relu
                                                          max_value=min_max_config['fc2'][1],
                                                          bit_width=act4_bit_width)
    act4_M0repr, act4_n = compute_M0repr_and_n(act3.scale, fc2_weight_scale, act4_scale)

    act4 = act3.linear(fc2_weight, fc2_bias, act4_scale, act4_zero_point, act4_bit_width, act4_M0repr, act4_n)

    fc3_weight = model_ckpt['fc3.weight']
    fc3_weight_bit_width = bit_width_config['fc3_weight']
    fc3_weight_scale, fc3_weight_zero_point = get_per_channel_q_params(fc3_weight, fc3_weight_bit_width)
    fc3_weight = QuantizedArray.quantize_per_channel(fc3_weight, fc3_weight_scale, fc3_weight_zero_point,
                                                     fc3_weight_bit_width)

    fc3_bias = model_ckpt['fc3.bias']
    fc3_bias_bit_width = bit_width_config['fc3_bias']
    fc3_bias_scale = act4.scale * fc3_weight_scale
    fc3_bias_zero_point = np.zeros(fc3_bias.shape, dtype=np.int64)
    fc3_bias = QuantizedArray.quantize_per_channel(fc3_bias, fc3_bias_scale, fc3_bias_zero_point,
                                                   fc3_bias_bit_width)

    act5_bit_width = bit_width_config['act5']
    act5_scale, act5_zero_point = get_q_params_from_range(min_value=min_max_config['fc3'][0],  # no relu
                                                          max_value=min_max_config['fc3'][1],
                                                          bit_width=act5_bit_width)
    act5_M0repr, act5_n = compute_M0repr_and_n(act4.scale, fc3_weight_scale, act5_scale)

    act5 = act4.linear(fc3_weight, fc3_bias, act5_scale, act5_zero_point, act5_bit_width, act5_M0repr, act5_n)
    return act5


@hydra.main(config_path='config', config_name='bw', version_base='1.2')
def main(cfg: DictConfig):
    print('Configuration:', cfg)

    bit_width_config_copy = cfg
    min_max_config = get_statistics()
    print('bit_width_config_copy', bit_width_config_copy)
    print('min_max_config', min_max_config)

    # for bw in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32]:
    #     for key in bit_width_config:
    #         if 'bias' not in key:
    #             bit_width_config[key] = bw

    # for layer in ['conv1_weight', 'conv2_weight', 'fc1_weight', 'fc2_weight']:
    #     for bw in [4, 2, 1]:
    #         bit_width_config = copy.deepcopy(bit_width_config_copy)
    #         bit_width_config[layer] = bw

    # print('bit_width_config', bit_width_config)
    bit_width_config = bit_width_config_copy

    correct_test_top1 = 0
    correct_test_top2 = 0
    correct_test_top3 = 0
    total_test = 0

    for i, (data, answer) in enumerate(dataset_test):
        output = infer(data, bit_width_config, min_max_config).array
        sorted_indices = np.argsort(output)[::-1]  # Sort indices of output in descending order of confidence

        # Get top-1, top-2, and top-3 predictions
        top1_prediction = sorted_indices[0]
        top2_predictions = sorted_indices[:2]
        top3_predictions = sorted_indices[:3]

        # Increment correct counters as appropriate
        if top1_prediction == answer:
            correct_test_top1 += 1
        if answer in top2_predictions:
            correct_test_top2 += 1
        if answer in top3_predictions:
            correct_test_top3 += 1

        # Optional: Print predictions for the first 10 instances or if the model is incorrect
        if top1_prediction != answer or i < 10:
            print(f'Prediction at index {i}, answer: {answer}, prediction: {top1_prediction}, output: {output}')

        total_test += 1

        # Print accuracies every 100 samples
        if total_test % 100 == 0:
            accuracy_top1 = correct_test_top1 / total_test * 100
            accuracy_top2 = correct_test_top2 / total_test * 100
            accuracy_top3 = correct_test_top3 / total_test * 100
            print(
                f'Accuracy Top-1: {accuracy_top1:.2f}%, Top-2: {accuracy_top2:.2f}%, Top-3: {accuracy_top3:.2f}%')
            print(
                f'Correct Top-1: {correct_test_top1}, Top-2: {correct_test_top2}, Top-3: {correct_test_top3}, '
                f'Total: {total_test}')
    accuracy_top1 = correct_test_top1 / total_test * 100
    accuracy_top2 = correct_test_top2 / total_test * 100
    accuracy_top3 = correct_test_top3 / total_test * 100

    with open('mnist_accuracy.txt', 'a') as f:
        f.write(f'{bit_width_config}\t{accuracy_top1}\t{accuracy_top2}\t{accuracy_top3}\n')

if __name__ == '__main__':
    main()