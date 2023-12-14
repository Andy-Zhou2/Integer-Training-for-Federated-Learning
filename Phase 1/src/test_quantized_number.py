import unittest
import numpy as np
from quantize.quantized_number import QuantizedArray
import torch


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
        qa = QuantizedArray.quantize_per_channel(arr,
                                                 scale=np.array([1e-2, 1e-1]),
                                                 zero_point=np.array([0, 1]), bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

        arr = np.array([[[[0, 1], [1, 0]]], [[[0, 1], [1, 0]]]])
        qa = QuantizedArray.quantize_per_channel(arr,
                                                 scale=np.array([0.5, 0.25]),
                                                 zero_point=np.array([50, -21]), bit_width=32)
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


class TestQuantizedArrayConv(unittest.TestCase):
    def simple_test_template(self, input_scale, input_zero_point, weights_scale, weights_zero_point, output_scale,
                             output_zero_point, bit_width=32):
        # input size: 1x3x3
        input_q = QuantizedArray.quantize_per_tensor(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]), input_scale,
                                                     input_zero_point, bit_width)
        # weights size: 2x1x2x2
        weights_q = QuantizedArray.quantize_per_channel(np.array([[[[0, 1], [1, 0]]], [[[0, 1], [1, 0]]]]),
                                                        weights_scale,
                                                        weights_zero_point, bit_width)
        # bias size: 1

        bias_q = QuantizedArray.quantize_per_channel(np.array([1.5, 2]), input_scale * weights_scale, np.array([0, 0]),
                                                     bit_width)

        M = input_scale * weights_scale / output_scale
        n = -np.ceil(np.log2(M)).astype(np.int_)
        for i in range(len(n)):
            if np.allclose(2. ** (-n[i]), M[i]):  # M is a power of 2
                n[i] -= 1
        M0 = M / (2. ** (-n))  # M0 in [0.5, 1)
        M0_repr = np.round(M0 * 2 ** 31).astype(np.int32)

        output_q = input_q.conv2d(weights_q, bias_q, output_scale, output_zero_point, bit_width, M0_repr=M0_repr, n=n)
        self.assertTrue(np.allclose(output_q.dequantize(), np.array([[[7.5, 9.5], [13.5, 15.5]], [[8, 10], [14, 16]]])))
        input_tensor = torch.tensor(input_q.dequantize())
        weights_tensor = torch.tensor(weights_q.dequantize())
        bias_tensor = torch.tensor(bias_q.dequantize())
        output_tensor = torch.nn.functional.conv2d(input_tensor, weights_tensor, bias_tensor)
        self.assertTrue(np.allclose(output_q.dequantize(), output_tensor))

    def test_simple_zero_point_is_zero(self):
        self.simple_test_template(input_scale=0.5, input_zero_point=0, weights_scale=np.array([0.5, 0.25]),
                                  weights_zero_point=np.array([0, 0]), output_scale=0.5, output_zero_point=0)

    def test_simple_M_greater_than_1(self):
        self.simple_test_template(input_scale=0.5, input_zero_point=0, weights_scale=np.array([0.5, 0.25]),
                                  weights_zero_point=np.array([0, 0]), output_scale=0.1, output_zero_point=0)

    def test_simple_zero_point_is_nonzero(self):
        self.simple_test_template(input_scale=0.5, input_zero_point=40, weights_scale=np.array([0.5, 0.25]),
                                  weights_zero_point=np.array([50, 75]), output_scale=0.1, output_zero_point=30)

    def test_simple_non_power_of_2_weights(self):
        self.simple_test_template(input_scale=0.5, input_zero_point=40, weights_scale=np.array([1e-3, 2e-5]),
                                  weights_zero_point=np.array([50, 75]), output_scale=0.0004, output_zero_point=30)

    def random_test(self):
        # Parameters for the test
        input_size = np.random.randint(8, 15)  # Random size for the input array
        kernel_size = np.random.randint(2, 4)  # Random kernel size
        C_in = np.random.randint(1, 3)  # Number of input channels
        C_out = np.random.randint(1, 5)  # Number of output channels

        bit_width = 32

        # Generate random input, weights, and bias
        input_array = np.random.randint(0, 255, (C_in, input_size, input_size), dtype=np.int64)
        weights = np.random.randint(-127, 128, (C_out, C_in, kernel_size, kernel_size), dtype=np.int64)
        bias = np.random.randint(-128, 128, C_out, dtype=np.int64)

        # Scales and zero points
        input_scale = np.random.uniform(0.01, 1)
        weight_scale = np.random.uniform(0.01, 1, C_out)
        bias_scale = input_scale * weight_scale
        output_scale = np.random.uniform(0.01, 1)

        input_zero_point = 0
        weight_zero_point = np.zeros(C_out, dtype=np.int64)
        bias_zero_point = np.zeros(C_out, dtype=np.int64)
        output_zero_point = 0

        # Calculate M0 and n for the conv2d function
        M = input_scale * weight_scale / output_scale
        n = -np.ceil(np.log2(M)).astype(np.int_)
        for i in range(len(n)):
            if np.allclose(2. ** (-n[i]), M[i]):  # M is a power of 2
                n[i] -= 1
        M0 = M / (2. ** (-n))  # M0 in [0.5, 1)
        M0_repr = np.round(M0 * 2 ** 31).astype(np.int32)

        # Quantize the input, weights, and bias
        input_q = QuantizedArray.quantize_per_tensor(input_array, input_scale, input_zero_point, bit_width)
        weights_q = QuantizedArray.quantize_per_channel(weights, weight_scale, weight_zero_point, bit_width)
        bias_q = QuantizedArray.quantize_per_channel(bias, bias_scale, bias_zero_point, bit_width)

        # print(input_q.dequantize())
        # print(weights_q.dequantize())
        # print(bias_q.dequantize())

        # Perform the convolution
        output_q = input_q.conv2d(weights_q, bias_q, output_scale, output_zero_point, bit_width, M0_repr, n)

        # Check the shape of the output
        expected_output_shape = (C_out, input_size - (kernel_size - 1), input_size - (kernel_size - 1))
        assert output_q.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {output_q.shape}"

        # Compare the output with PyTorch's convolution
        input_tensor = torch.tensor(input_q.dequantize())
        weights_tensor = torch.tensor(weights_q.dequantize())
        bias_tensor = torch.tensor(bias_q.dequantize())
        output_tensor = torch.nn.functional.conv2d(input_tensor, weights_tensor, bias_tensor)

        # print(f'output_tensor: {output_tensor}')
        # print(f'our output: {output_q.dequantize()}')

        # inconsistent_coords = np.argwhere(np.isclose(output_q.dequantize(), output_tensor, rtol=0.01) == False)
        # print(f'Inconsistent coordinates: {inconsistent_coords}')
        # print(f'Inconsistent values: {output_q.dequantize()[inconsistent_coords[:, 0], inconsistent_coords[:, 1], inconsistent_coords[:, 2]]}')
        # print(f'Inconsistent values: {output_tensor[inconsistent_coords[:, 0], inconsistent_coords[:, 1], inconsistent_coords[:, 2]]}')

        self.assertTrue(np.allclose(output_q.dequantize(), output_tensor, rtol=0.01, atol=0.5))

    def test_sample(self):
        np.random.seed(1024)
        for _ in range(10):
            self.random_test()

if __name__ == '__main__':
    unittest.main()
