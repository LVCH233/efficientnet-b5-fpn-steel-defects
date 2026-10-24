import importlib.util
import unittest
import numpy as np


class RLETests(unittest.TestCase):
    def api(self):
        if importlib.util.find_spec('vdcp.rle') is None:
            self.fail('RLE implementation is missing')
        from vdcp.rle import decode_rle, encode_rle
        return decode_rle, encode_rle

    def test_column_major_one_based_known_mask(self):
        decode, encode = self.api()
        expected = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(decode('1 2 6 1', (2, 3)), expected)
        self.assertEqual(encode(expected), '1 2 6 1')

    def test_empty_and_all_pixels(self):
        decode, encode = self.api()
        np.testing.assert_array_equal(decode('', (2, 3)), np.zeros((2, 3)))
        self.assertEqual(encode(np.ones((2, 3), dtype=np.uint8)), '1 6')
        self.assertEqual(encode(np.zeros((2, 3), dtype=np.uint8)), '')

    def test_invalid_runs_rejected(self):
        decode, _ = self.api()
        for bad in ['0 1', '6 2', '1 -1', '1', '1 3 2 2', 'a b']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decode(bad, (2, 3))


if __name__ == '__main__':
    unittest.main()
