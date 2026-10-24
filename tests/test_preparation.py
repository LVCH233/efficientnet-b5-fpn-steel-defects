import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


class PreparationTests(unittest.TestCase):
    def api(self):
        if importlib.util.find_spec('vdcp.preparation') is None:
            self.fail('Safe dataset extraction is missing')
        import vdcp.preparation as preparation
        return preparation

    def test_archive_traversal_is_rejected(self):
        api = self.api()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['../escape.jpg', '..\\escape.jpg', 'C:/escape.jpg', '/escape.jpg']:
                with ZipFile(root / 'bad.zip', 'w') as archive:
                    archive.writestr(name, b'test')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    api.safe_extract(root / 'bad.zip', root / 'images')
            self.assertFalse((root / 'escape.jpg').exists())

    def test_existing_files_verified_and_not_silently_replaced(self):
        api = self.api()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with ZipFile(root / 'good.zip', 'w') as archive:
                archive.writestr('nested/image.jpg', b'original')
            api.safe_extract(root / 'good.zip', root / 'images')
            api.safe_extract(root / 'good.zip', root / 'images')
            (root / 'images/nested/image.jpg').write_bytes(b'different')
            with self.assertRaises(ValueError):
                api.safe_extract(root / 'good.zip', root / 'images')


if __name__ == '__main__':
    unittest.main()
