import numpy as np
from typing import Union


def quantize_and_clip(array, scale, zero_point, bit_width):
    array = np.array(array, dtype=np.float64) / scale + zero_point
    array = np.round(array)
    array = np.clip(array, -2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1)
    array = array.astype(np.int64)

    return array


def compute_M0repr_and_n(input_scale, weights_scale, output_scale):
    """
    Computes M0_repr and n for the quantization of the output of a quantized layer.

    Define M = input_scale * weights_scale / output_scale.
    Then normalize M = 2^(-n) * M0, where 0.5 <= M0 < 1.
    Let M0_repr = M0 * 2^31.
    :return: M0_repr, n
    """
    M = input_scale * weights_scale / output_scale
    n = -np.ceil(np.log2(M)).astype(np.int_)
    for i in range(len(n)):
        if np.allclose(2. ** (-n[i]), M[i]):  # M is a power of 2
            n[i] -= 1
    M0 = M / (2. ** (-n))  # M0 in [0.5, 1)
    M0_repr = np.round(M0 * 2 ** 31).astype(np.int32)
    return M0_repr, n


class QuantizedArray:
    def __init__(self, array, scale, zero_point, bit_width, per_channel=False):
        # check if array is consistent with bit_width
        assert 1 <= bit_width <= 32, 'bit_width must be between 1 and 32'
        assert np.all(array <= 2 ** (bit_width - 1) - 1) and np.all(array >= -2 ** (bit_width - 1)), \
            'array is not consistent with bit_width'
        assert np.all(zero_point <= 2 ** (bit_width - 1) - 1) and np.all(zero_point >= -2 ** (bit_width - 1)), \
            'zero_point is not consistent with bit_width'
        assert np.all(scale > 0), 'scale must be positive'

        self.array = np.array(array, dtype=np.int64)
        self.scale = np.array(scale, dtype=np.float64)
        self.zero_point = np.array(zero_point, dtype=np.int64)
        self.bit_width = np.array(bit_width, dtype=np.int32)
        self.per_channel = per_channel

    def __repr__(self):
        return f'QuantizedArray({self.array}, scale={self.scale}, zero_point={self.zero_point}, bit_width={self.bit_width})'

    @property
    def shape(self):
        return self.array.shape

    @classmethod
    def quantize_per_tensor(cls, array: Union[list, np.ndarray], scale: Union[float, int], zero_point: int,
                            bit_width: int):
        assert 0 < bit_width <= 32, 'bit_width must be between 1 and 32'

        array = quantize_and_clip(array, scale, zero_point, bit_width)

        return cls(array, scale, zero_point, bit_width)

    @classmethod
    def quantize_per_channel(cls, array: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, bit_width: int):
        assert 0 < bit_width <= 32, 'bit_width must be between 1 and 32'
        assert array.shape[0] == scale.shape[0] == zero_point.shape[0], \
            'array, scale, and zero_point must have the same number of channels'
        result = np.empty(array.shape, dtype=np.int64)
        for ch in range(len(scale)):
            result[ch] = quantize_and_clip(array[ch], scale[ch], zero_point[ch], bit_width)

        return cls(result, scale, zero_point, bit_width, per_channel=True)

    def dequantize(self):  # returns float64
        if not self.per_channel:  # per tensor
            return (self.array - self.zero_point) * self.scale
        else:  # per channel
            broadcast_shape = [self.scale.size] + [1] * (self.array.ndim - 1)
            return (self.array - self.zero_point.reshape(broadcast_shape)) * self.scale.reshape(broadcast_shape)

    def add(self, other, scale, zero_point, bit_width):
        assert isinstance(other, QuantizedArray), 'Can only add QuantizedArrays'
        assert self.shape == other.shape, 'Arrays must be of equal shape'

        r = self.dequantize() + other.dequantize()
        return QuantizedArray.quantize_per_tensor(r, scale, zero_point, bit_width)

    def matmul(self, other: 'QuantizedArray', scale, zero_point, result_bit_width, intermediate_bit_width=32):
        assert isinstance(other, QuantizedArray), 'Can only add QuantizedArrays'
        assert self.array.ndim == 2 and other.array.ndim == 2, 'Arrays must be 2D'
        assert self.shape[1] == other.shape[0], 'Arrays must be of appropriate shape'
        assert not self.per_channel and not other.per_channel, 'Per-channel quantization not supported'

        result = np.empty([self.shape[0], other.shape[1]], dtype=np.int64)

        m = self.scale * other.scale / scale

        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                q = np.zeros([1], dtype=np.int64)
                for k in range(self.shape[1]):
                    q += (self.array[i, k] - self.zero_point) * (other.array[k, j] - other.zero_point)
                    q = np.clip(q, -2 ** (intermediate_bit_width - 1), 2 ** (intermediate_bit_width - 1) - 1)
                q = np.round(q * m + zero_point)
                result[i, j] = q
        return QuantizedArray(result, scale, zero_point, result_bit_width)

    def conv2d(self, weight: 'QuantizedArray', bias: 'QuantizedArray', scale, zero_point, bit_width,
               M0_repr: np.ndarray[np.int32], n: np.ndarray[np.int_]):
        # input: (C_in, H, W)
        # weight: (C_out, C_in, kH, kW)
        # bias: (C_out)
        # scale, zero_point, bit_width: those of the output
        # bias should have scale of S1 * S2 and zero_point 0
        # Normalize M := S1 * S2 / S3 = 2^(-n) * M0, where 0.5 <= M0 < 1
        # M0_repr: M0 * 2^31
        assert not self.per_channel, 'Per-channel quantization not supported for input'
        assert weight.per_channel, 'Per-channel quantization required for weight'
        assert bias.per_channel, 'Per-channel quantization required for bias'
        assert np.all(bias.zero_point == 0), 'Bias must be zero-pointed at 0'
        assert np.allclose(bias.scale, self.scale * weight.scale), \
            'Bias scale must be equal to input scale * weight scale'

        C_in, H, W = self.shape
        C_out, _, kH, kW = weight.shape

        # output
        outH = H - (kH - 1)
        outW = W - (kW - 1)
        output = np.empty((C_out, outH, outW), dtype=np.int64)

        for c_out in range(C_out):
            for h in range(outH):
                for w in range(outW):
                    # Convolution
                    q = np.zeros([1], dtype=np.int64)  # but values clipped within 32 bit
                    for c_in in range(C_in):
                        for kh in range(kH):
                            for kw in range(kW):
                                a1 = np.clip(weight.array[c_out, c_in, kh, kw] - weight.zero_point[c_out], -2 ** 31,
                                             2 ** 31 - 1)
                                a2 = np.clip(self.array[c_in, h + kh, w + kw] - self.zero_point, -2 ** 31, 2 ** 31 - 1)
                                delta = a1 * a2  # 2 ** 32 squared would overflow int64, so clip each term

                                # clip delta, prevent overflow
                                delta = np.clip(delta, np.iinfo(np.int32).min - q, np.iinfo(np.int32).max - q)
                                q += delta

                    # Add bias
                    delta = bias.array[c_out]
                    delta = np.clip(delta, np.iinfo(np.int32).min - q, np.iinfo(np.int32).max - q)
                    q += delta

                    # multiply by M
                    q = np.multiply(q, M0_repr[c_out], dtype=np.int64)
                    # right shift and round by n bits
                    bits_to_shift = n[c_out] + 31

                    q += np.bitwise_and(q, np.left_shift(1, bits_to_shift - 1, dtype=np.int64))  # to round
                    # works for negative and positive numbers
                    # for negative numbers, -1.5 -> -1, -2.5 -> -2

                    q = np.right_shift(q, bits_to_shift)
                    q += zero_point
                    q = np.clip(q, -2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1)

                    output[c_out, h, w] = q

        # quantize
        output = QuantizedArray(output, scale, zero_point, bit_width)

        return output

    def linear(self, weight: 'QuantizedArray', bias: 'QuantizedArray', scale, zero_point, bit_width,
               M0_repr: np.ndarray[np.int32], n: np.ndarray[np.int_]):
        # input: (C_in)
        # weight: (C_out, C_in)
        # bias: (C_out)
        # scale, zero_point, bit_width: those of the output
        # bias should have scale of S1 * S2 and zero_point 0
        # Normalize M := S1 * S2 / S3 = 2^(-n) * M0, where 0.5 <= M0 < 1
        # M0_repr: M0 * 2^31
        assert not self.per_channel, 'Per-channel quantization not supported for input'
        assert weight.per_channel, 'Per-channel quantization required for weight'
        assert bias.per_channel, 'Per-channel quantization required for bias'
        assert np.all(bias.zero_point == 0), 'Bias must be zero-pointed at 0'
        assert np.allclose(bias.scale, self.scale * weight.scale), \
            'Bias scale must be equal to input scale * weight scale'


        C_out, C_in = weight.shape
        assert self.shape[0] == C_in, 'Input and weight must have compatible shapes'
        assert bias.shape[0] == C_out, 'Bias and weight must have compatible shapes'

        output = np.empty((C_out,), dtype=np.int64)
        for c_out in range(C_out):
            # Linear
            q = np.zeros([1], dtype=np.int64)
            for c_in in range(C_in):
                a1 = np.clip(weight.array[c_out, c_in] - weight.zero_point[c_out], -2 ** 31, 2 ** 31 - 1)
                a2 = np.clip(self.array[c_in] - self.zero_point, -2 ** 31, 2 ** 31 - 1)
                delta = a1 * a2
                delta = np.clip(delta, np.iinfo(np.int32).min - q, np.iinfo(np.int32).max - q)
                q += delta

            # Add bias
            delta = bias.array[c_out]
            delta = np.clip(delta, np.iinfo(np.int32).min - q, np.iinfo(np.int32).max - q)
            q += delta
            q = np.multiply(q, M0_repr[c_out], dtype=np.int64)
            bits_to_shift = n[c_out] + 31
            q += np.bitwise_and(q, np.left_shift(1, bits_to_shift - 1, dtype=np.int64))  # to round
            q = np.right_shift(q, bits_to_shift)
            q += zero_point
            q = np.clip(q, -2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1)

            output[c_out] = q
        output = QuantizedArray(output, scale, zero_point, bit_width)
        return output

    def maxpool2d(self, kernel_size):
        assert not self.per_channel, 'Per-channel quantization not supported for input'

        stride = kernel_size

        C, H, W = self.shape
        outH = H // stride
        outW = W // stride
        output = np.empty((C, outH, outW), dtype=np.int64)
        for c in range(C):
            for h in range(outH):
                for w in range(outW):
                    output[c, h, w] = np.max(self.array[c, h * stride:h * stride + kernel_size,
                                             w * stride:w * stride + kernel_size])
        return QuantizedArray(output, self.scale, self.zero_point, self.bit_width)

    def flatten(self):
        assert not self.per_channel, 'Per-channel quantization not supported for input'

        return QuantizedArray(self.array.flatten(), self.scale, self.zero_point, self.bit_width)