import numpy as np


class QuantizedArray:
    def __init__(self, array, scale, zero_point, bit_width):
        self.array = np.array(array, dtype=np.int64)
        self.scale = np.array(scale, dtype=np.float64)
        self.zero_point = np.array(zero_point, dtype=np.int64)
        self.bit_width = np.array(bit_width, dtype=np.int32)

    def __repr__(self):
        return f'QuantizedArray({self.array}, scale={self.scale}, zero_point={self.zero_point}, bit_width={self.bit_width})'

    @property
    def shape(self):
        return self.array.shape

    @classmethod
    def quantize(cls, array, scale, zero_point, bit_width):
        assert isinstance(bit_width, int), 'bit_width must be an integer'
        assert 0 < bit_width <= 64, 'bit_width must be between 1 and 64'

        array = np.array(array, dtype=np.float64) / scale + zero_point
        array = np.round(array)
        array = np.clip(array, -2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1)
        array = array.astype(np.int64)

        return cls(array, scale, zero_point, bit_width)

    def dequantize(self):  # returns float64
        return (self.array - self.zero_point) * self.scale

    def add(self, other, scale, zero_point, bit_width):
        assert isinstance(other, QuantizedArray), 'Can only add QuantizedArrays'
        assert self.shape == other.shape, 'Arrays must be of equal shape'

        r = self.dequantize() + other.dequantize()
        return QuantizedArray.quantize(r, scale, zero_point, bit_width)

    def matmul(self, other, scale, zero_point, result_bit_width, intermediate_bit_width=32):
        assert isinstance(other, QuantizedArray), 'Can only add QuantizedArrays'
        assert self.array.ndim == 2 and other.array.ndim == 2, 'Arrays must be 2D'
        assert self.shape[1] == other.shape[0], 'Arrays must be of appropriate shape'

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

