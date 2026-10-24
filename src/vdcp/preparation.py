"""Read-only source archives, checked extraction, reproducible manifests."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import zlib


def safe_extract(archive_path, target):
    root = Path(target).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        members = []
        for info in archive.infolist():
            name = info.filename.replace('\\', '/')
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name:
                raise ValueError(f'Unsafe archive member: {info.filename}')
            output = root.joinpath(*relative.parts).resolve()
            if not output.is_relative_to(root):
                raise ValueError(f'Archive member escapes dataset: {info.filename}')
            members.append((info, output))
        for info, output in members:
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if output.exists():
                crc = 0
                with output.open('rb') as existing:
                    while block := existing.read(1024 * 1024):
                        crc = zlib.crc32(block, crc)
                if output.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError(f'Existing dataset file differs: {output}')
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open('wb') as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
