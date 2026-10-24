import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch


def runtime():
    path = Path(__file__).resolve().parents[1] / 'scripts/cloud_b5_runtime.py'
    assert path.exists(), 'cloud reliability module is not implemented'
    spec = importlib.util.spec_from_file_location('cloud_runtime_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_json_failure_is_nonfatal_to_updates(tmp_path, monkeypatch, capsys):
    m = runtime()
    path = tmp_path / 'status.json'
    path.write_text('{"old": true}')
    def deny(*args):
        raise PermissionError('reader holds telemetry')
    monkeypatch.setattr(m.os, 'replace', deny)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    for _ in range(2):
        opt.zero_grad(); p.square().sum().backward(); opt.step()
        assert m.atomic_json(path, {'new': True}, retries=2, retry_delay=0) is False
    assert p.item() == pytest.approx(.64)
    assert json.loads(path.read_text()) == {'old': True}
    assert 'telemetry_write_warning' in capsys.readouterr().err


def test_checkpoint_failed_replace_preserves_real_checkpoint(tmp_path, monkeypatch):
    m = runtime(); path = tmp_path / 'last.pt'
    m.atomic_checkpoint(path, {'epoch': 72, 'weight': torch.tensor([3.0])})
    def deny(*args):
        raise OSError('disk unavailable')
    monkeypatch.setattr(m.os, 'replace', deny)
    with pytest.raises(OSError):
        m.atomic_checkpoint(path, {'epoch': 73, 'weight': torch.tensor([9.0])})
    assert torch.load(path, weights_only=False)['weight'].item() == 3


def test_history_replaces_same_epoch_without_duplicate(tmp_path):
    m = runtime(); path = tmp_path / 'history.json'
    m.append_history(path, {'epoch': 74, 'loss': 1})
    m.append_history(path, {'epoch': 73, 'loss': 2})
    m.append_history(path, {'epoch': 74, 'loss': .5})
    assert json.loads(path.read_text()) == [{'epoch': 73, 'loss': 2}, {'epoch': 74, 'loss': .5}]


def test_process_lock_excludes_other_writer_and_releases(tmp_path):
    m = runtime(); path = tmp_path / 'writer.lock'
    with m.ProcessLock(path):
        with pytest.raises(RuntimeError, match='locked'):
            with m.ProcessLock(path):
                pass
    with m.ProcessLock(path):
        pass


def test_snapshot_folder_uses_other_disk_without_deletion(tmp_path, monkeypatch):
    m = runtime(); first = tmp_path / 'system'; second = tmp_path / 'data'
    first.mkdir(); second.mkdir(); sentinel = first / 'keep'; sentinel.write_text('immutable')
    from collections import namedtuple
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: Usage(100, 95, 5) if Path(p) == first else Usage(100, 10, 90))
    assert m.pick_snapshot_folder([first, second], reserve_bytes=10, size_estimate=20) == second
    assert sentinel.read_text() == 'immutable'
    with pytest.raises(OSError, match='space'):
        m.pick_snapshot_folder([first], reserve_bytes=10, size_estimate=20)


def test_finalized_transaction_recovery_repairs_history_best_and_snapshot(tmp_path, monkeypatch):
    m = runtime(); output = tmp_path / 'out'; output.mkdir(); snapshots = tmp_path / 'snapshots'
    payload = {'epoch': 73, 'pending_validation': False, 'weight': torch.tensor([4.0]), 'best': {'dice': .8}}
    record = {'epoch': 73, 'validation': {'dice': .8, 'images': 1885}}
    original = m.append_history
    monkeypatch.setattr(m, 'append_history', lambda *a, **k: (_ for _ in ()).throw(OSError('history unavailable')))
    with pytest.raises(OSError):
        m.commit_epoch(output, payload, record, is_best=True, snapshot_folders=[snapshots], reserve_bytes=0)
    assert torch.load(output / 'last.pt', weights_only=False)['last_epoch_record'] == record
    monkeypatch.setattr(m, 'append_history', original)
    assert m.recover_publication(output, snapshot_folders=[snapshots], reserve_bytes=0) == record
    assert torch.load(output / 'best_region_f2.pt', weights_only=False)['epoch'] == 73
    assert [json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()] == [record]
    index = json.loads((output / 'snapshot_index.json').read_text())
    snapshot = Path(index['73']['path'])
    assert snapshot.is_file()
    assert torch.load(snapshot, weights_only=False)['weight'].item() == 4
    m.recover_publication(output, snapshot_folders=[snapshots], reserve_bytes=0)
    assert len([json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()]) == 1


def test_pending_validation_is_never_published(tmp_path):
    m = runtime(); m.atomic_checkpoint(tmp_path / 'last.pt', {'epoch': 74, 'pending_validation': True})
    assert m.recover_publication(tmp_path) is None
    assert not (tmp_path / 'history.json').exists()


def test_controller_stop_has_priority_over_complete(tmp_path):
    m = runtime()
    m.atomic_checkpoint(tmp_path / 'last.pt', {'epoch': 200, 'pending_validation': False})
    m.atomic_json(tmp_path / 'status.json', {'status': 'milestone_complete', 'last_validated_epoch': 200}, strict=True)
    m.atomic_json(tmp_path / 'last_header.json', {'epoch': 200, 'pending_validation': False, 'full_validation': True, 'validation_images': 1885, 'checkpoint_sha256': m.sha256_file(tmp_path / 'last.pt')}, strict=True)
    assert m.controller_decision(tmp_path, 200) == 'complete'
    (tmp_path / 'STOP').write_text('manual')
    assert m.controller_decision(tmp_path, 200) == 'stopped'
    (tmp_path / 'STOP').unlink()
    m.atomic_json(tmp_path / 'last_header.json', {'epoch': 200, 'pending_validation': True, 'full_validation': False}, strict=True)
    assert m.controller_decision(tmp_path, 200) == 'resume'


def test_controller_restarts_failed_process_then_completes(tmp_path):
    m = runtime(); output = tmp_path / 'output'; output.mkdir()
    m.atomic_checkpoint(output / 'last.pt', {'epoch': 199, 'pending_validation': False})
    entry = tmp_path / 'fake_trainer.py'
    entry.write_text('''import argparse, json, pathlib, sys, hashlib
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');p.add_argument('--resume',action='store_true');a=p.parse_args()
o=pathlib.Path(a.output); counter=o/'attempts'; n=int(counter.read_text())+1 if counter.exists() else 1;counter.write_text(str(n))
if n==1: sys.exit(17)
(o/'status.json').write_text(json.dumps({'status':'milestone_complete','last_validated_epoch':200}))
(o/'last_header.json').write_text(json.dumps({'epoch':200,'pending_validation':False,'full_validation':True,'validation_images':1885,'checkpoint_sha256':hashlib.sha256((o/'last.pt').read_bytes()).hexdigest()}))
''')
    result = subprocess.run([sys.executable, str(Path(m.__file__)), '--root', str(tmp_path), '--output', str(output), '--entry', str(entry), '--backoff-initial', '0.01', '--backoff-max', '0.02', '--poll-seconds', '0.01'], timeout=20, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / 'attempts').read_text() == '2'
    status = json.loads((output / 'controller_status.json').read_text())
    assert status['status'] == 'complete'
    events = [json.loads(s) for s in (output / 'controller_events.jsonl').read_text().splitlines()]
    assert any(x.get('returncode') == 17 and x['event'] == 'process_exit' for x in events)
    assert any(x['event'] == 'restart_backoff' for x in events)



def test_telemetry_serialization_failure_is_nonfatal(tmp_path, capsys):
    m = runtime()
    assert m.atomic_json(tmp_path / 'status.json', {'value': float('nan')}) is False
    assert 'telemetry_write_warning' in capsys.readouterr().err
    with pytest.raises(ValueError):
        m.atomic_json(tmp_path / 'status.json', {'value': float('nan')}, strict=True)


def test_complete_requires_checkpoint_digest_and_full_validation_header(tmp_path):
    m = runtime()
    m.atomic_checkpoint(tmp_path / 'last.pt', {'epoch': 200, 'pending_validation': False})
    m.atomic_json(tmp_path / 'status.json', {'status': 'milestone_complete', 'last_validated_epoch': 200}, strict=True)
    m.atomic_json(tmp_path / 'last_header.json', {'epoch': 200, 'pending_validation': False, 'full_validation': True}, strict=True)
    assert m.controller_decision(tmp_path) == 'resume'
    header = {'epoch': 200, 'pending_validation': False, 'full_validation': True, 'validation_images': 1885, 'checkpoint_sha256': m.sha256_file(tmp_path / 'last.pt')}
    m.atomic_json(tmp_path / 'last_header.json', header, strict=True)
    assert m.controller_decision(tmp_path) == 'complete'
    (tmp_path / 'last.pt').write_bytes(b'corruption')
    with pytest.raises(ValueError, match='digest'):
        m.controller_decision(tmp_path)


def test_immutable_snapshot_mismatch_is_visible_and_never_overwritten(tmp_path):
    m = runtime(); snapshots = tmp_path / 'snapshots'; snapshots.mkdir()
    existing = snapshots / 'epoch_0073.pt'; existing.write_bytes(b'old immutable snapshot')
    with pytest.raises(ValueError, match='immutable'):
        m.commit_epoch(tmp_path, {'epoch': 73}, {'epoch': 73, 'validation': {'images': 1885}}, snapshot_folders=[snapshots], reserve_bytes=0)
    assert existing.read_bytes() == b'old immutable snapshot'


def test_controller_missing_checkpoint_waits_for_manual_stop(tmp_path):
    m = runtime(); output = tmp_path / 'output'; output.mkdir()
    process = subprocess.Popen([sys.executable, str(Path(m.__file__)), '--root', str(tmp_path), '--output', str(output), '--entry', str(tmp_path / 'missing.py'), '--backoff-initial', '.01', '--backoff-max', '.02', '--poll-seconds', '.01'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not (output / 'controller_status.json').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        status = json.loads((output / 'controller_status.json').read_text())
        assert status['status'] == 'recovery_error'
        assert 'checkpoint missing' in status['error']
        assert process.poll() is None
        (output / 'STOP').write_text('manual')
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        assert json.loads((output / 'controller_status.json').read_text())['status'] == 'stopped'
    finally:
        if process.poll() is None:
            process.kill(); process.communicate()


def test_telemetry_stderr_outage_cannot_abort_updates(tmp_path, monkeypatch):
    m = runtime()
    class BrokenStream:
        def write(self, *args):
            raise OSError('log storage also unavailable')
        def flush(self):
            pass
    def deny(*args):
        raise OSError('telemetry storage unavailable')
    monkeypatch.setattr(m.os, 'replace', deny)
    monkeypatch.setattr(m.sys, 'stderr', BrokenStream())
    assert m.atomic_json(tmp_path / 'status.json', {'epoch': 73}, retries=1) is False


def test_manual_stop_precedes_invalid_controller_policy(tmp_path):
    m = runtime()
    (tmp_path / 'policy.json').write_text('invalid-json')
    (tmp_path / 'STOP').write_text('manual')
    result = subprocess.run([sys.executable, str(Path(m.__file__)), '--root', str(tmp_path), '--output', str(tmp_path), '--entry', str(tmp_path / 'missing.py'), '--backoff-initial', '.01', '--backoff-max', '.02', '--poll-seconds', '.01'], timeout=2, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / 'controller_status.json').read_text())['status'] == 'stopped'


def test_watchdog_restarts_stalled_worker_and_resumes(tmp_path):
    m = runtime(); output = tmp_path / 'output'; output.mkdir()
    m.atomic_checkpoint(output / 'last.pt', {'epoch': 199})
    entry = tmp_path / 'stall_worker.py'
    entry.write_text('''import argparse,json,pathlib,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');p.add_argument('--resume',action='store_true');a=p.parse_args()
o=pathlib.Path(a.output); c=o/'attempts'; n=int(c.read_text())+1 if c.exists() else 1;c.write_text(str(n))
if n==1: time.sleep(20)
else:
 (o/'status.json').write_text(json.dumps({'status':'milestone_complete','last_validated_epoch':200}))
 (o/'last_header.json').write_text(json.dumps({'epoch':200,'pending_validation':False,'full_validation':True,'validation_images':1885,'checkpoint_sha256':hashlib.sha256((o/'last.pt').read_bytes()).hexdigest()}))
''')
    result = subprocess.run([sys.executable, str(Path(m.__file__)), '--root', str(tmp_path), '--output', str(output), '--entry', str(entry), '--backoff-initial', '.01', '--backoff-max', '.02', '--poll-seconds', '.03', '--stall-seconds', '2', '--terminate-grace-seconds', '.1'], timeout=60, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / 'attempts').read_text() == '2'
    events = [json.loads(line) for line in (output / 'controller_events.jsonl').read_text().splitlines()]
    assert any(x['event'] == 'worker_stalled' and x['idle_seconds'] >= 2 for x in events)
    assert any(x['event'] == 'process_exit' and x.get('cause') == 'stalled' for x in events)
    assert events[-1]['event'] == 'complete'


def test_watchdog_does_not_limit_total_runtime_when_progress_continues(tmp_path):
    m = runtime(); output = tmp_path / 'output'; output.mkdir()
    m.atomic_checkpoint(output / 'last.pt', {'epoch': 199})
    entry = tmp_path / 'progress_worker.py'
    entry.write_text('''import argparse,json,pathlib,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');p.add_argument('--resume',action='store_true');a=p.parse_args()
o=pathlib.Path(a.output); (o/'attempts').write_text('1')
for i in range(7):
 (o/'status.json').write_text(json.dumps({'status':'validation','validation_done':i*8}));print('progress',i,flush=True);time.sleep(.4)
(o/'status.json').write_text(json.dumps({'status':'milestone_complete','last_validated_epoch':200}))
(o/'last_header.json').write_text(json.dumps({'epoch':200,'pending_validation':False,'full_validation':True,'validation_images':1885,'checkpoint_sha256':hashlib.sha256((o/'last.pt').read_bytes()).hexdigest()}))
''')
    start = time.monotonic()
    result = subprocess.run([sys.executable, str(Path(m.__file__)), '--root', str(tmp_path), '--output', str(output), '--entry', str(entry), '--backoff-initial', '.01', '--backoff-max', '.02', '--poll-seconds', '.03', '--stall-seconds', '2', '--terminate-grace-seconds', '.1'], timeout=60, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - start > .4
    events = [json.loads(line) for line in (output / 'controller_events.jsonl').read_text().splitlines()]
    assert not any(x['event'] == 'worker_stalled' for x in events)
    assert sum(x['event'] == 'progress_heartbeat' for x in events) >= 3
