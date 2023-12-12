import unittest
import numpy as np
from quantize.quantized_number import QuantizedArray


class TestQuantizedArray(unittest.TestCase):

    def test_creation(self):
        # Test basic creation of a QuantizedArray
        qa = QuantizedArray([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=64)
        self.assertTrue(np.array_equal(qa.array, np.array([1, 2, 3])* 2**10))

    def test_bit_width_acc_loss(self):
        qa = QuantizedArray([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=10)
        self.assertTrue(np.array_equal(qa.array, np.array([511, 511, 511])))

        qa = QuantizedArray([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=11)
        self.assertTrue(np.array_equal(qa.array, np.full([3], 1023)))

        qa = QuantizedArray([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=12)
        self.assertTrue(np.array_equal(qa.array, np.array([1024, 2047, 2047])))


    def test_bit_width(self):
        # Test if the bit_width is correctly set
        qa = QuantizedArray([1, 2, 3], scale=2 ** -10, zero_point=0, bit_width=8)
        self.assertEqual(qa.bit_width, 8)

    def test_dequantize(self):
        arr = [1, 2, 3]
        qa = QuantizedArray(arr, scale=2 ** -10, zero_point=0, bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

        arr = [1, 5, 9, 5]
        qa = QuantizedArray(arr, scale=2 ** -10, zero_point=0, bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

        arr = [-2, -5, -7, 10]
        qa = QuantizedArray(arr, scale=2 ** -10, zero_point=2**30, bit_width=64)
        self.assertTrue(np.allclose(qa.dequantize(), arr))

    def test_add(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([3, 4, 5])
        a = QuantizedArray(arr1, scale=2 ** -10, zero_point=2**30, bit_width=64)
        b = QuantizedArray(arr2, scale=2 ** -10, zero_point=2**30, bit_width=64)
        c = a.add(b, scale=2 ** -10, zero_point=2**30, bit_width=64)
        self.assertTrue(np.allclose(c.dequantize(), arr1+arr2))




if __name__ == '__main__':
    unittest.main()
