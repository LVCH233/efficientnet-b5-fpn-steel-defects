"""Kaggle's one-based, column-major run-length encoding."""
import numpy as np


def decode_rle(text, shape=(256, 1600)):
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError('Mask dimensions must be positive')
    flat = np.zeros(height * width, dtype=np.uint8)
    if text is None or str(text).strip() == '' or str(text).lower() == 'nan':
        return flat.reshape(shape, order='F')
    try:
        runs = np.asarray([int(x) for x in str(text).split()], dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError('RLE must contain integer start/length pairs') from error
    if runs.size % 2:
        raise ValueError('RLE requires start/length pairs')
    starts, lengths = runs[::2] - 1, runs[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(lengths <= 0) or np.any(ends > flat.size):
        raise ValueError('RLE run is outside the image')
    if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError('RLE runs must be ordered and non-overlapping')
    changes = np.zeros(flat.size + 1, dtype=np.int32)
    np.add.at(changes, starts, 1)
    np.add.at(changes, ends, -1)
    return np.cumsum(changes[:-1]).astype(np.uint8).reshape(shape, order='F')


def encode_rle(mask):
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError('RLE encoding requires a 2D mask')
    pixels = (array.ravel(order='F') > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], pixels, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = np.empty(starts.size * 2, dtype=np.int64)
    runs[::2], runs[1::2] = starts + 1, ends - starts
    return ' '.join(map(str, runs))
