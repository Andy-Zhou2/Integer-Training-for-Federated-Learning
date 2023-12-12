import numpy as np


class QuantizedArray:
    def __init__(self, array, /, scale, zero_point, bit_width):
        assert isinstance(bit_width, int), 'bit_width must be an integer'
        assert 0 < bit_width <= 64, 'bit_width must be between 1 and 64'
        self.bit_width = bit_width

        self.scale = np.array(scale, dtype=np.float64)
        self.zero_point = np.array(zero_point, dtype=np.int64)

        array = np.array(array, dtype=np.float64) / scale + zero_point
        array = np.round(array)
        array = np.clip(array, -2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1)
        array = array.astype(np.int64)

        self.array = array

    def __repr__(self):
        return f'QuantizedArray({self.array}, scale={self.scale}, zero_point={self.zero_point}, bit_width={self.bit_width})'

    def dequantize(self):  # returns float64
        return (self.array - self.zero_point) * self.scale

    def add(self, other, scale, zero_point, bit_width):
        assert isinstance(other, QuantizedArray), 'Can only add QuantizedArrays'
        assert self.bit_width == other.bit_width, 'Bit widths must be equal'
        assert self.array.size == other.array.size, 'Arrays must be of equal size'

        r = self.dequantize() + other.dequantize()
        return QuantizedArray(r, scale, zero_point, bit_width)

