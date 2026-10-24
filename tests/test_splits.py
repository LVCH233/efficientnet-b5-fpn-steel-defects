import importlib.util
import unittest


class SplitTests(unittest.TestCase):
    def test_duplicate_images_stay_together_and_every_image_is_assigned(self):
        if importlib.util.find_spec('vdcp.splits') is None:
            self.fail('Grouped split implementation is missing')
        from vdcp.splits import create_splits
        records = [dict(image_id=f'{i}.jpg', sha256=str(i//2),
            **{f'c{c+1}': int((i//2) % (c+2) == 0) for c in range(4)}) for i in range(200)]
        first = create_splits(records)
        second = create_splits(records)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {row['image_id'] for row in records})
        self.assertEqual(set(first.values()), {'train', 'val', 'test'})
        for i in range(0, 200, 2):
            self.assertEqual(first[f'{i}.jpg'], first[f'{i+1}.jpg'])


if __name__ == '__main__':
    unittest.main()
