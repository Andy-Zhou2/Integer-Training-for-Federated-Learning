import unittest
import numpy as np
from quantize.quantized_number import QuantizedArray


class TestQuantizedArrayCreation(unittest.TestCase):

    def test_creation_per_tensor(self):
        qa = QuantizedArray.quantize_per_tensor([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=64)
        self.assertTrue(np.array_equal(qa.array, np.array([1, 2, 3]) * 2 ** 10))

    def test_creation_per_channel(self):
        qa = QuantizedArray.quantize_per_channel(np.array([1., 2.]), scale=np.array([1e-2, 1e-1]),
                                                 zero_point=np.array([0, 1]), bit_width=64)
        self.assertTrue(np.array_equal(qa.array, np.array([100, 21])))

    def test_bit_width_acc_loss(self):
        qa = QuantizedArray.quantize_per_tensor([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=10)
        self.assertTrue(np.array_equal(qa.array, np.array([511, 511, 511])))

        qa = QuantizedArray.quantize_per_tensor([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=11)
        self.assertTrue(np.array_equal(qa.array, np.full([3], 1023)))

        qa = QuantizedArray.quantize_per_tensor([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=12)
        self.assertTrue(np.array_equal(qa.array, np.array([1024, 2047, 2047])))

    def test_bit_width(self):
        # Test if the bit_width is correctly set
        qa = QuantizedArray.quantize_per_tensor([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=8)
        self.assertEqual(qa.bit_width, 8)

    def test_dequantize_per_tensor(self):
        arr = [1, 2, 3]
        qa = QuantizedArray.quantize_per_tensor(arr, scale=2 ** -10, zero_point=0, bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

        arr = [1, 5, 9, 5]
        qa = QuantizedArray.quantize_per_tensor(arr, scale=2 ** -10, zero_point=0, bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

        arr = [-2, -5, -7, 10]
        qa = QuantizedArray.quantize_per_tensor(arr, scale=2 ** -10, zero_point=2 ** 30, bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

    def test_dequantize_per_channel(self):
        arr = np.array([1., 2.])
        qa = QuantizedArray.quantize_per_channel(arr, scale=np.array([1e-2, 1e-1]),
                                                 zero_point=np.array([0, 1]), bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

    def test_add(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([3, 4, 5])
        a = QuantizedArray.quantize_per_tensor(arr1, scale=2 ** -10, zero_point=2 ** 30, bit_width=64)
        b = QuantizedArray.quantize_per_tensor(arr2, scale=2 ** -10, zero_point=2 ** 30, bit_width=64)
        c = a.add(b, scale=2 ** -10, zero_point=2 ** 30, bit_width=64)
        self.assertTrue(np.allclose(c.dequantize(), arr1 + arr2))


class TestQuantizedArrayMatMul(unittest.TestCase):

    def test_matmul_basic(self):
        # Test basic matrix multiplication
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        scale = 2 ** -10
        zero_point = 0
        bit_width = 64

        qa1 = QuantizedArray.quantize_per_tensor(arr1, scale, zero_point, bit_width)
        qa2 = QuantizedArray.quantize_per_tensor(arr2, scale, zero_point, bit_width)

        result = qa1.matmul(qa2, scale, zero_point, bit_width)
        expected = np.matmul(arr1, arr2)

        self.assertTrue(np.allclose(result.dequantize(), expected))

    def test_matmul_identity(self):
        # Test multiplication with an identity matrix
        arr = np.array([[1, 2], [3, 4]])
        identity = np.array([[1, 0], [0, 1]])
        scale = 2 ** -10
        zero_point = 0
        bit_width = 64

        qa = QuantizedArray.quantize_per_tensor(arr, scale, zero_point, bit_width)
        qa_identity = QuantizedArray.quantize_per_tensor(identity, scale, zero_point, bit_width)

        result = qa.matmul(qa_identity, scale, zero_point, bit_width)

        self.assertTrue(np.allclose(result.dequantize(), arr))

    def test_matmul_zero_matrix(self):
        # Test multiplication with a zero matrix
        arr = np.array([[1, 2], [3, 4]])
        zero_matrix = np.array([[0, 0], [0, 0]])
        scale = 2 ** -10
        zero_point = 0
        bit_width = 64

        qa = QuantizedArray.quantize_per_tensor(arr, scale, zero_point, bit_width)
        qa_zero = QuantizedArray.quantize_per_tensor(zero_matrix, scale, zero_point, bit_width)

        result = qa.matmul(qa_zero, scale, zero_point, bit_width)

        self.assertTrue(np.allclose(result.dequantize(), zero_matrix))

    def test_matmul_different_bit_width(self):
        # Test matrix multiplication with different bit widths
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        scale = 2 ** -10
        zero_point = 0
        bit_width_1 = 64
        bit_width_2 = 32

        qa1 = QuantizedArray.quantize_per_tensor(arr1, scale, zero_point, bit_width_1)
        qa2 = QuantizedArray.quantize_per_tensor(arr2, scale, zero_point, bit_width_2)

        result = qa1.matmul(qa2, scale, zero_point, bit_width_1, intermediate_bit_width=bit_width_2)
        expected = np.matmul(arr1, arr2)

        self.assertTrue(np.allclose(result.dequantize(), expected))


if __name__ == '__main__':
    unittest.main()
