"""Crash-safe checkpoint publication and unattended single-writer cloud supervision.

No project data or snapshots are deleted. Torch is imported only for checkpoint I/O;
telemetry and the controller otherwise use the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _warn(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except OSError:
        # Telemetry must stay nonfatal even when its log storage also fails.
        pass


def _fsync_directory(path):
    if os.name == 'posix':
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        # Only this call's unpublished temporary file can be removed.
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, payload, retries=3, retry_delay=0.1, strict=False):
    """Retry replace failures; telemetry failure returns False, strict data raises."""
    try:
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError) as error:
        if strict:
            raise
        _warn(json.dumps({'event': 'telemetry_write_warning', 'path': str(path), 'error': repr(error), 'at': utc_now()}))
        return False
    for attempt in range(max(1, retries)):
        try:
            _atomic_write(path, lambda handle: handle.write(encoded))
            return True
        except OSError as error:
            if attempt + 1 < max(1, retries):
                time.sleep(retry_delay * (2 ** attempt))
                continue
            if strict:
                raise
            _warn(json.dumps({'event': 'telemetry_write_warning', 'path': str(path), 'error': repr(error), 'at': utc_now()}))
    return False


def atomic_checkpoint(path, payload, save_fn=None):
    """Durably replace a genuine checkpoint, preserving the prior file on failure.

    save_fn has torch.save's (payload, binary_file) signature. Failures are strict:
    callers must retry or restart, never continue from an unsaved update.
    """
    if save_fn is None:
        import torch
        save_fn = torch.save
    _atomic_write(path, lambda handle: save_fn(payload, handle))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def append_history(path, record):
    """Atomic epoch upsert; supports the project's epochs.jsonl and JSON lists."""
    path = Path(path)
    if path.exists():
        text = path.read_text(encoding='utf-8')
        rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
    else:
        rows = []
    epoch = int(record['epoch'])
    rows = [row for row in rows if int(row['epoch']) != epoch] + [record]
    rows.sort(key=lambda row: int(row['epoch']))
    if path.suffix == '.jsonl':
        encoded = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows).encode('utf-8')
        _atomic_write(path, lambda handle: handle.write(encoded))
    else:
        atomic_json(path, rows, strict=True)
    return rows


class ProcessLock:
    """Kernel lock released on process death; the lock file itself is retained."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        try:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0:
                    self.handle.write(b' '); self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self.handle.close(); self.handle = None
            raise RuntimeError('locked: another process owns ' + str(self.path)) from error
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close(); self.handle = None


def pick_snapshot_folder(folders, reserve_bytes, size_estimate):
    failures = []
    for candidate in folders:
        folder = Path(candidate)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(folder).free
            if free >= int(reserve_bytes) + int(size_estimate):
                return folder
            failures.append(str(folder) + ': free=' + str(free))
        except OSError as error:
            failures.append(str(folder) + ': ' + repr(error))
    raise OSError('insufficient snapshot space: ' + '; '.join(failures))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source, target):
    def writer(handle):
        with Path(source).open('rb') as original:
            shutil.copyfileobj(original, handle, 4 * 1024 * 1024)
    _atomic_write(target, writer)


def _load_checkpoint(path, load_fn=None):
    if load_fn is None:
        import torch
        return torch.load(path, map_location='cpu', weights_only=False)
    return load_fn(path)


def _publish(output, checkpoint, record, snapshot_folders, reserve_bytes):
    output = Path(output)
    epoch = int(checkpoint['epoch'])
    if bool(checkpoint.get('pending_validation', False)):
        raise ValueError('cannot publish a pending-validation checkpoint')
    if int(record['epoch']) != epoch or int(record.get('validation', {}).get('images', 0)) != 1885:
        raise ValueError('finalized epoch requires its full 1885-image validation')
    source = output / 'last.pt'
    digest = sha256_file(source)
    index_path = output / 'snapshot_index.json'
    index = read_json(index_path, {})
    existing = index.get(str(epoch))
    folders = list(snapshot_folders or [output / 'checkpoints'])
    # Existing artifacts win over free-space selection, making restart idempotent.
    candidates = [Path(existing['path'])] if existing else []
    candidates += [Path(folder) / ('epoch_%04d.pt' % epoch) for folder in folders]
    snapshot = next((candidate for candidate in candidates if candidate.is_file()), None)
    if snapshot is None:
        folder = pick_snapshot_folder(folders, reserve_bytes, source.stat().st_size)
        snapshot = folder / ('epoch_%04d.pt' % epoch)
        _copy_atomic(source, snapshot)
    if sha256_file(snapshot) != digest:
        raise ValueError('immutable epoch snapshot differs from checkpoint: ' + str(snapshot))
    index[str(epoch)] = {'path': str(snapshot.resolve()), 'sha256': digest, 'bytes': snapshot.stat().st_size}
    atomic_json(index_path, index, strict=True)
    if checkpoint.get('_cloud_publication', {}).get('is_best', False):
        _copy_atomic(source, output / 'best_region_f2.pt')
    append_history(output / 'epochs.jsonl', record)
    atomic_json(output / 'last_header.json', {
        'epoch': epoch, 'last_validated_epoch': epoch, 'pending_validation': False,
        'full_validation': True, 'validation_images': 1885,
        'checkpoint_sha256': digest, 'snapshot': str(snapshot.resolve()), 'at': utc_now(),
    }, strict=True)
    return snapshot


def commit_epoch(output, checkpoint, record, *, is_best=False, save_fn=None, snapshot_folders=None, reserve_bytes=512 * 1024 * 1024):
    """Commit finalized last.pt first; publication can be repaired after any crash.

    Caller must hold writer.lock throughout training and publication. Immutable
    snapshots contain exactly the committed bytes; existing epochs never change.
    """
    if int(record.get('validation', {}).get('images', 0)) != 1885:
        raise ValueError('full 1885-image validation is required')
    if int(record['epoch']) != int(checkpoint['epoch']):
        raise ValueError('epoch mismatch')
    payload = dict(checkpoint)
    payload.update(last_epoch_record=record, pending_validation=False,
                   _cloud_publication={'is_best': bool(is_best)})
    atomic_checkpoint(Path(output) / 'last.pt', payload, save_fn=save_fn)
    return _publish(output, payload, record, snapshot_folders, reserve_bytes)


def recover_publication(output, *, load_fn=None, save_fn=None, snapshot_folders=None, reserve_bytes=512 * 1024 * 1024):
    """Finish finalized publication, or leave pending validation to the trainer."""
    path = Path(output) / 'last.pt'
    if not path.exists():
        return None
    checkpoint = _load_checkpoint(path, load_fn)
    if checkpoint.get('pending_validation', False):
        return None
    record = checkpoint.get('last_epoch_record')
    if record is None:
        return None
    _publish(output, checkpoint, record, snapshot_folders, reserve_bytes)
    return record


def controller_decision(output, target_epoch=200):
    output = Path(output)
    if (output / 'STOP').exists():
        return 'stopped'
    status = read_json(output / 'status.json', {})
    header = read_json(output / 'last_header.json', {})
    if (status.get('status') in ('milestone_complete', 'complete', 'epoch_complete')
            and int(status.get('last_validated_epoch', 0)) >= target_epoch
            and int(header.get('epoch', 0)) >= target_epoch
            and header.get('pending_validation') is False
            and header.get('full_validation') is True
            and int(header.get('validation_images', 0)) == 1885
            and header.get('checkpoint_sha256')):
        expected = header.get('checkpoint_sha256')
        if expected and sha256_file(output / 'last.pt') != expected:
            raise ValueError('last checkpoint digest disagrees with finalized header')
        return 'complete'
    return 'resume'


def _event(output, event, **fields):
    payload = {'event': event, 'at': utc_now(), 'pid': os.getpid(), **fields}
    try:
        with (Path(output) / 'controller_events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        _warn('controller_event_write_warning: ' + repr(error))
    atomic_json(Path(output) / 'controller_status.json', {'status': event, **payload})


def _backoff(output, seconds, poll_seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (Path(output) / 'STOP').exists():
            return
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def _progress_files(output):
    signatures = []
    for name in ('status.json', 'trainer_stdout.log', 'trainer_stderr.log'):
        try:
            stat = (Path(output) / name).stat()
            signatures.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            signatures.append((name, None, None))
    return tuple(signatures)


def supervise(root, output, entry, *, checkpoint=None, backoff_initial=5., backoff_max=300., poll_seconds=2., stall_seconds=1200., terminate_grace_seconds=30.):
    """Stay alive across worker exits; every launch resumes, never starts fresh.

    The caller detaches this controller with start_new_session=True and redirects
    stdin/out/err to persistent files. Worker STOP is honored at saved boundaries.
    The watchdog measures inactivity, never total training or validation time.
    """
    root, output, entry = Path(root).resolve(), Path(output).resolve(), Path(entry).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint) if checkpoint else output / 'last.pt'
    failures = 0
    with ProcessLock(output / 'controller.lock'):
        while True:
            try:
                if (output / 'STOP').exists():
                    _event(output, 'stopped', target_epoch=200)
                    return 0
                policy = read_json(output / 'policy.json', {})
                target = int(policy.get('target_epoch', 200))
                if target != 200:
                    raise ValueError('cloud target_epoch must remain 200')
                decision = controller_decision(output, target)
                if decision != 'resume':
                    _event(output, decision, target_epoch=target)
                    return 0
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise FileNotFoundError('resume checkpoint missing or empty: ' + str(checkpoint))
                # The trainer strictly loads/validates checkpoint state on every
                # resume, including migrated checkpoints without a cloud header.
                command = [sys.executable, str(entry), '--root', str(root), '--output', str(output), '--resume']
                with (output / 'trainer_stdout.log').open('ab', buffering=0) as stdout, (output / 'trainer_stderr.log').open('ab', buffering=0) as stderr:
                    process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                    last_progress = time.monotonic()
                    signature = _progress_files(output)
                    stalled = False
                    _event(output, 'running', child_pid=process.pid, attempt=failures + 1, target_epoch=target, stall_seconds=stall_seconds)
                    while process.poll() is None:
                        time.sleep(poll_seconds)
                        current = _progress_files(output)
                        now = time.monotonic()
                        if current != signature:
                            signature = current
                            last_progress = now
                            _event(output, 'progress_heartbeat', child_pid=process.pid, attempt=failures + 1, status='running')
                        if process.poll() is None and now - last_progress >= stall_seconds:
                            stalled = True
                            _event(output, 'worker_stalled', child_pid=process.pid, idle_seconds=now - last_progress, stall_seconds=stall_seconds)
                            process.terminate()
                            try:
                                process.wait(timeout=terminate_grace_seconds)
                            except subprocess.TimeoutExpired:
                                _event(output, 'worker_killed', child_pid=process.pid, terminate_grace_seconds=terminate_grace_seconds)
                                process.kill()
                                process.wait()
                    returncode = process.returncode
                _event(output, 'process_exit', child_pid=process.pid, returncode=returncode, attempt=failures + 1, cause='stalled' if stalled else 'exit')
                decision = controller_decision(output, target)
                if decision != 'resume':
                    _event(output, decision, returncode=returncode, target_epoch=target)
                    return 0
                failures += 1
                delay = min(backoff_max, backoff_initial * (2 ** min(failures - 1, 20)))
                _event(output, 'restart_backoff', returncode=returncode, failures=failures, seconds=delay)
            except (OSError, ValueError, TypeError, KeyError) as error:
                failures += 1
                delay = min(backoff_max, backoff_initial * (2 ** min(failures - 1, 20)))
                _event(output, 'recovery_error', failures=failures, seconds=delay, error=repr(error))
            _backoff(output, delay, poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--entry', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--backoff-initial', type=float, default=5.)
    parser.add_argument('--backoff-max', type=float, default=300.)
    parser.add_argument('--poll-seconds', type=float, default=2.)
    parser.add_argument('--stall-seconds', type=float, default=1200., help='Restart only after this many seconds without status/log progress; no total runtime limit.')
    parser.add_argument('--terminate-grace-seconds', type=float, default=30.)
    args = parser.parse_args(argv)
    if min(args.backoff_initial, args.backoff_max, args.poll_seconds, args.stall_seconds, args.terminate_grace_seconds) <= 0:
        parser.error('backoff and poll durations must be positive')
    return supervise(args.root, args.output, args.entry, checkpoint=args.checkpoint,
                     backoff_initial=args.backoff_initial, backoff_max=args.backoff_max, poll_seconds=args.poll_seconds,
                     stall_seconds=args.stall_seconds, terminate_grace_seconds=args.terminate_grace_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
