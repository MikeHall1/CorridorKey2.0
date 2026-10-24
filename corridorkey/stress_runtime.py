"""Owned-worker stress exercises; observations are engineering evidence, not acceptance."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .control import CancelledError, ProgressEvent
from .session import JobRequest

TERMINAL = {'result', 'failed', 'cancelled', 'rejected', 'lost'}
SCENARIOS = {'long_clip', 'batch50', 'repeated', 'concurrent_isolated', 'cancel',
             'interruption_recovery', 'memory_growth', 'encoder_failure', 'disk_full'}


def production_prerequisites(weights=None, asset_root=None):
    """No model construction, network or asset deserialization."""
    from .runtime_paths import RuntimePaths
    gaps = []
    try:
        RuntimePaths.from_package(Path(__file__).resolve().parents[1], weights, asset_root=asset_root).validate()
    except (OSError, ValueError) as exc:
        gaps.append(str(exc)[:2048])
    try:
        import importlib
        torch = importlib.import_module('torch')
        if not torch.cuda.is_available():
            gaps.append('CUDA is unavailable; production engine inference is not verified')
    except ImportError:
        gaps.append('PyTorch is unavailable')
    return gaps


class _Unavailable(RuntimeError):
    pass


def _inventory(root, check, cap):
    """Bounded no-link scan, counting apparent payload bytes, never disk reservation."""
    result = {}
    pending = [root]
    count = 0
    total = 0
    while pending:
        directory = pending.pop()
        check()
        try:
            directory_info = directory.lstat()
        except FileNotFoundError:
            if directory == root:
                raise
            # A cooperating worker may remove its owned job scratch subtree.
            continue
        if (not stat.S_ISDIR(directory_info.st_mode)
                or getattr(directory_info, 'st_file_attributes', 0) & 0x400):
            raise RuntimeError('links/reparse or non-directory owned artifacts are refused')
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            if directory == root:
                raise
            # Only an actually absent descendant is benign. A dangling link or
            # another replacement must remain an inventory failure.
            try:
                directory.lstat()
            except FileNotFoundError:
                continue
            raise
        with entries:
            for entry in entries:
                count += 1
                if count > 100000:
                    raise RuntimeError('owned artifact entry limit exceeded')
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # Keep the observed-entry charge; do not invent payload
                    # bytes for an entry that vanished before metadata capture.
                    continue
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise RuntimeError('links/reparse entries in owned artifacts are refused')
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                    if total > cap:
                        raise RuntimeError('cooperative owned artifact byte ceiling exceeded')
                    result[str(Path(entry.path).relative_to(root))] = info.st_size
                else:
                    raise RuntimeError('nonregular owned artifact refused')
    return result, total


def execute_scenario(case, jobs, weights, root, *, check=None, progress=None,
                     worker_factory=None, observer_factory=None, asset_root=None, limits=None):
    """Run one normalized case. Root must be a fresh directory or an empty directory.

    Factories are Python-only test seams; no executable selection comes from case JSON.
    Native calls and filesystem I/O have cooperative, rather than hard, deadlines.
    """
    from .process_memory import ProcessTreeMemory
    from .worker_client import WorkerCleanupError, WorkerClient
    start = time.monotonic()
    clients = []
    admitted = False
    root = Path(root)
    row = {'status': 'failed', 'success': False, 'reason': None, 'error': None,
           'acceptance_verdict': None, 'raw_records': [], 'resources': [], 'coverage': {},
           'partial_paths': [], 'cleanup': {'verified': True, 'owners': []},
           'scope': {'engine': 'injected Python fixture' if worker_factory else 'production shared engine',
                     'artifact_limit': 'cooperatively observed apparent owned-file bytes; not a disk reservation',
                     'memory': 'current summed process-tree RSS; shared pages may be counted repeatedly'}}
    limits = limits or {}
    deadline = start + 86400
    cap = limits.get('max_artifact_bytes', 1024 ** 3)

    def poll():
        if check:
            check()
        if time.monotonic() > deadline:
            raise TimeoutError('stress scenario deadline exceeded')

    def scan():
        files, size = _inventory(root, poll, cap)
        row['artifact_bytes'] = size
        return files

    def new_client():
        poll()
        kwargs = {'weights': weights, 'asset_root': asset_root, 'capacity': 1}
        if worker_factory is not None:
            kwargs.update(root=root, scenario=case['scenario'])
        try:
            client = (worker_factory or WorkerClient)(**kwargs)
        except WorkerCleanupError as exc:
            row['cleanup'].update(verified=False, startup_error=exc.cleanup_error[:2048])
            raise
        clients.append(client)
        return client

    def clone(job, label, *, encoder_fault=False):
        cfg, writer = job.configurations()
        cfg.temp_dir = root / label / 'temp'
        if encoder_fault:
            cfg.output_format = 'video'
            writer.codec = 'corridorkey_deliberately_invalid_encoder'
        return JobRequest.create(job.input_path, root / label / 'output', cfg, writer,
                                 initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
                                 profile_stages=job.profile_stages)

    def digest(path):
        expected = path.stat().st_size
        value = hashlib.sha256()
        with path.open('rb') as stream:
            remaining = expected
            while remaining:
                poll()
                data = stream.read(min(1024 * 1024, remaining))
                if not data:
                    raise RuntimeError('artifact truncated while hashing')
                value.update(data)
                remaining -= len(data)
            if stream.read(1):
                raise RuntimeError('artifact grew while hashing')
        return value.hexdigest()

    def readable():
        from .video_io import cv2
        files = scan()
        found = []
        for relative in files:
            if relative.lower().endswith('.exr'):
                poll()
                path = root / relative
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is not None and image.size:
                    found.append(relative)
        return found

    record_bytes = 0

    def admit_evidence(value):
        nonlocal record_bytes
        size = len(json.dumps(value, allow_nan=False).encode())
        if record_bytes + size > min(limits.get('max_evidence_bytes', 4 * 1024 * 1024), 4 * 1024 * 1024):
            raise RuntimeError('stress raw record byte bound exceeded')
        record_bytes += size

    def record(event):
        admit_evidence(event)
        row['raw_records'].append(event)

    def wait(client, request_id, *, cancel=False):
        triggered = False
        while True:
            poll()
            if cancel and not triggered:
                found = readable()
                indices = {int(match.group(1)) for name in found
                           if (match := re.search(r'_(\d{6,})\.exr$', name))}
                if len(indices) >= case.get('cancel_after_readable_frames', 1):
                    row['fault_observation'] = {'readable_frames': found[:64], 'readable_frame_indices': sorted(indices)[:64],
                                                'at_seconds': time.monotonic() - start}
                    client.cancel(request_id, 'stress cancellation after readable EXR')
                    triggered = True
            scan()
            try:
                event = client.next_event(timeout=.05)
            except TimeoutError:
                continue
            if event.get('request_id') == request_id and event['kind'] in TERMINAL:
                record(event)
                return event, triggered

    def run(client, job, *, cancel=False):
        request_id = client.submit(job)
        return wait(client, request_id, cancel=cancel)

    def interruption_recovery():
        cfg, _ = jobs[0].configurations()
        if cfg.output_format != 'exr' or cfg.auto_vram:
            raise _Unavailable('Recovery requires EXR and explicit scheduling (auto_vram=false)')
        from .diagnostics import json_value

        def manifest(job, name):
            cfg, writer = job.configurations()
            value = {'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': job.input_path, 'output_dir': job.output_dir, 'settings': json_value(asdict(cfg)),
                 'writer': asdict(writer), 'hint_first_frame': job.initial_hint_path,
                 'hint_video': job.hint_video_path, 'profile_stages': job.profile_stages}]}
            if weights is not None:
                value['weights'] = str(weights)
            path = root / name
            with path.open('x') as stream:
                json.dump(value, stream, allow_nan=False)
            return path

        interrupted = clone(jobs[0], 'interrupted')
        batch = manifest(interrupted, 'interrupted-batch.json')
        workspace = root / 'recovery-workspace'
        policy = case.get('recovery', {})
        payload = {'batch': str(batch.resolve()), 'workspace': str(workspace.resolve()),
                   'max_cache_bytes': policy.get('max_cache_bytes', min(cap, 256 * 1024 ** 2)), 'resume': False,
                   **{name: policy.get(name, False) for name in
                      ('durable_foundations', 'durable_rvm', 'durable_hann', 'durable_stage2', 'durable_writer_prefix',
                       'reuse_stage2_finishing_edits', 'reuse_upstream_finishing_edits')}}
        client = new_client()
        request_id = client.send('recover', payload)
        partial = {}
        indices = set()
        while not partial:
            poll()
            found = [relative for relative in readable() if relative.startswith('interrupted/output/')]
            indices = {int(match.group(1)) for name in found
                       if (match := re.search(r'_(\d{6,})\.exr$', name))}
            if len(indices) >= case.get('cancel_after_readable_frames', 1):
                partial = {relative: digest(root / relative) for relative in found}
                break
            try:
                event = client.next_event(timeout=.05)
            except TimeoutError:
                continue
            if event.get('request_id') == request_id and event['kind'] in TERMINAL:
                record(event)
                raise _Unavailable('Recovery completed before a readable partial could be interrupted')
        client._terminate_owned()
        client.close()
        row['deliberate_interruption'] = {'pid': client.process.pid, 'returncode': client.process.poll(),
                                         'partial_sha256': partial, 'readable_frame_indices': sorted(indices),
                                         'required_readable_frames': case.get('cancel_after_readable_frames', 1)}
        # Original partial export is never reused as the resumed output directory.
        payload.update(resume=True)
        resumed = new_client()
        event, _ = wait(resumed, resumed.send('recover', payload))
        if event['kind'] != 'result':
            raise RuntimeError('Explicit recovery resume failed')
        from .local_json import read_object
        report_path = Path(event['payload']['record']['report_path'])
        if not report_path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('Recovery report escaped owned root')
        report, _ = read_object(report_path, max_bytes=16 * 1024 * 1024, regular_file=True)
        completed = report['jobs'][0]
        resumed_output = Path(completed['output_dir'])
        if not resumed_output.resolve().is_relative_to(root.resolve()) or resumed_output == Path(interrupted.output_dir):
            raise RuntimeError('Resume did not preserve the original partial output directory')
        row['recovery_report'] = {'report_path': str(report_path), 'mode': completed['mode'],
                                  'output_dir': str(resumed_output)}
        oracle = new_client()
        event, _ = run(oracle, clone(jobs[0], 'oracle'))
        if event['kind'] != 'result':
            raise RuntimeError('Uninterrupted oracle failed')
        files = scan()
        def hashes(prefix):
            values = {}
            for relative in files:
                if relative.startswith(prefix) and relative.lower().endswith('.exr'):
                    poll()
                    values[relative[len(prefix):]] = digest(root / relative)
            return values
        oracle_hashes, resumed_hashes = hashes('oracle/output/'), hashes(resumed_output.resolve().relative_to(root.resolve()).as_posix() + '/')
        if not oracle_hashes or oracle_hashes != resumed_hashes:
            raise RuntimeError('Resumed export differs from uninterrupted oracle')
        if any(digest(root / name) != expected_digest for name, expected_digest in partial.items()):
            raise RuntimeError('Original partial export changed')
        row['coverage'].update(oracle_exact_files=len(oracle_hashes), original_partial_preserved=True,
                               explicit_resume=True, implicit_retry=False)
        row.update(status='measured', success=True)
        return row

    def concurrent_isolated():
        first, second = new_client(), new_client()
        a, b = clone(jobs[0], 'concurrent-a'), clone(jobs[-1], 'concurrent-b')
        ids = [first.submit(a), second.submit(b)]
        began = time.monotonic()
        events = [{}, {}]
        while not all(events):
            poll()
            scan()
            for i, owner in enumerate((first, second)):
                if events[i]:
                    continue
                try:
                    event = owner.next_event(timeout=.02)
                except TimeoutError:
                    continue
                if event.get('request_id') == ids[i] and event['kind'] in TERMINAL:
                    record(event)
                    events[i] = event
        intervals = [event.get('payload', {}).get('execution_interval') for event in events]
        valid = all(isinstance(item, dict) for item in intervals)
        if valid:
            valid = all(item.get('pid') == client.process.pid and
                        type(item.get('started_monotonic_ns')) is int and
                        type(item.get('finished_monotonic_ns')) is int and
                        item['finished_monotonic_ns'] >= item['started_monotonic_ns']
                        for item, client in zip(intervals, (first, second), strict=True))
        overlap = None
        if valid and intervals[0].get('clock_implementation') == intervals[1].get('clock_implementation'):
            overlap = min(item['finished_monotonic_ns'] for item in intervals) - max(item['started_monotonic_ns'] for item in intervals)
        row['coverage']['two_live_workers'] = True
        row['coverage']['overlap_verified'] = overlap is not None and overlap > 0
        row['coverage']['engine_run_overlap_seconds'] = overlap / 1e9 if overlap is not None else None
        row['coverage']['overlap_scope'] = 'engine.run includes model load/preflight/render; not GPU kernel overlap'
        row['coverage']['observation_seconds'] = time.monotonic() - began
        if any(event['kind'] != 'result' for event in events):
            raise RuntimeError('concurrent worker failed')
        if overlap is None or overlap <= 0:
            raise _Unavailable('Two isolated workers completed without verified positive engine.run interval overlap')
        row.update(status='measured', success=True)
        return row

    try:
        durations = (case.get('timeout_seconds', 1800), limits.get('deadline_seconds', 7200))
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= maximum
               for value, maximum in zip(durations, (14400, 86400), strict=True)):
            raise ValueError('deadlines must be finite and within scenario/suite bounds')
        deadline = start + min(durations)
        poll()
        scenario = case.get('scenario')
        if scenario not in SCENARIOS:
            raise ValueError('unknown stress scenario')
        if type(cap) is not int or not 1 <= cap <= 2 ** 63 - 1:
            raise ValueError('invalid artifact byte ceiling')
        if not jobs or len(jobs) > 64:
            raise _Unavailable('one to 64 local immutable job templates are required')
        if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            raise ValueError('stress root must be fresh and empty, without a symlink')
        for job in jobs:
            source = Path(job.input_path).resolve()
            if root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
                raise ValueError('artifact root overlaps source')
            cfg, _ = job.configurations()
            protected_paths = [job.output_dir, cfg.temp_dir, job.initial_hint_path, job.hint_video_path, cfg.corrections_path]
            for protected in protected_paths:
                if protected is not None and root.resolve().is_relative_to(Path(protected).resolve()):
                    raise ValueError('artifact root is inside an original output, temp or dependency path')
        for protected in ([asset_root] if asset_root else []):
            if root.resolve().is_relative_to(Path(protected).resolve()):
                raise ValueError('artifact root is inside selected assets')
        root.mkdir(parents=False, exist_ok=True)
        admitted = True
        if worker_factory is None:
            gaps = production_prerequisites(weights, asset_root)
            if gaps:
                raise _Unavailable('; '.join(gaps))
        if scenario == 'disk_full':
            volume = case.get('fault_volume')
            if not volume or not root.resolve().is_relative_to(Path(volume).resolve()):
                raise _Unavailable('Disk-full requires artifact root within the explicitly prepared disposable volume')
            statvfs = vars(os).get('statvfs')
            if not callable(statvfs) or cast(Callable[[str], Any], statvfs)(volume).f_bavail != 0:
                raise _Unavailable('Prepared volume is not already full; this tool never fills a volume')
            row['fault_observation'] = {'volume': str(volume), 'available_blocks_before': 0,
                                        'method': 'existing full operator-prepared filesystem, no filler'}
        if scenario == 'batch50':
            distinct = len({str(Path(job.input_path).resolve()) for job in jobs[:50]})
            row['coverage']['distinct_selected_source_paths'] = distinct
            if len(jobs) < 50 or distinct < 50:
                raise _Unavailable('batch50 requires 50 distinct selected source paths; repetition is a separate scenario')
            identities = case.get('source_identities', [])
            if len(identities) < 50 or len({item.get('sha256') for item in identities[:50]}) < 50:
                raise _Unavailable('batch50 requires 50 distinct source-content identities; copied clips are repetitions')
            row['coverage']['distinct_selected_source_contents'] = 50
        if scenario == 'long_clip':
            from .video_io import probe_frame_dir_meta, probe_video_meta
            cfg, _ = jobs[0].configurations()
            source = Path(jobs[0].input_path)
            if source.is_dir():
                meta = probe_frame_dir_meta(str(source), num_frames=cfg.num_frames, start_frame=cfg.start_frame,
                                           fps=cfg.frame_dir_fps, shape_quantum=cfg.shape_quantum,
                                           exr_linear=cfg.frame_dir_linear, model_scale=cfg.model_scale,
                                           alpha_mode=cfg.frame_dir_alpha, source_colour_profile=cfg.source_colour_profile)
            else:
                meta = probe_video_meta(str(source), num_frames=cfg.num_frames, shape_quantum=cfg.shape_quantum,
                                        native_resolution=True, start_mode=cfg.start_mode, seed=cfg.seed,
                                        model_scale=cfg.model_scale, linear_brightness=cfg.linear_brightness,
                                        linear_contrast=cfg.linear_contrast, linear_contrast_pivot=cfg.linear_contrast_pivot,
                                        start_frame=cfg.start_frame)
            seconds = meta['num_frames'] / meta['native_fps']
            row['coverage']['nominal_selected_seconds'] = seconds
            if seconds < 600:
                raise _Unavailable('long_clip requires at least 600 nominal selected seconds')
        if scenario == 'encoder_failure':
            _, writer = jobs[0].configurations()
            if shutil.which(writer.ffmpeg_bin) is None:
                raise _Unavailable('FFmpeg unavailable; missing encoder is not a failed encoder test')
        if scenario == 'cancel' and jobs[0].configurations()[0].output_format != 'exr':
            raise _Unavailable('cancellation trigger requires actual readable EXR output')
        if scenario == 'concurrent_isolated':
            return concurrent_isolated()
        if scenario == 'interruption_recovery':
            return interruption_recovery()
        client = new_client()
        if scenario == 'memory_growth':
            observer = (observer_factory or ProcessTreeMemory)(root_pid=client.process.pid)
            warmup = case.get('warmup_jobs', 1)
            repeats = case.get('repetitions', 10)
            if not 1 <= warmup <= 10 or not 10 <= repeats <= 128:
                raise ValueError('memory growth requires 1..10 warmups and 10..128 measured jobs')
        else:
            observer = None
            warmup = 0
            repeats = 50 if scenario == 'batch50' else case.get('repetitions', 3) if scenario == 'repeated' else 1
        if type(repeats) is not int or not 1 <= repeats <= 128:
            raise ValueError('repetitions must be 1..128')
        for index in range(warmup + repeats):
            event, triggered = run(client, clone(jobs[index % len(jobs)], f'job-{index:04d}', encoder_fault=scenario == 'encoder_failure'),
                                   cancel=scenario == 'cancel')
            if scenario == 'encoder_failure':
                payload = json.dumps(event.get('payload', {})).lower()
                if event['kind'] != 'failed' or 'ffmpeg writer failed with exit code' not in payload:
                    raise _Unavailable('No exact failed-FFmpeg-child evidence; generic failure is insufficient')
            elif scenario == 'disk_full':
                payload = json.dumps(event.get('payload', {})).lower()
                if event['kind'] != 'failed' or not any(text in payload for text in ('enospc', 'no space left on device', 'errno 28')):
                    raise _Unavailable('No exact ENOSPC evidence from prepared full volume')
            elif scenario == 'cancel':
                if not triggered or event['kind'] != 'cancelled':
                    raise _Unavailable('Job completed or failed before readable-output cancellation was confirmed')
            elif event['kind'] != 'result':
                raise RuntimeError('stress job did not complete successfully')
            if observer is not None:
                sample = observer.sample(check=poll)
                sample['warmup'] = index < warmup
                admit_evidence(sample)
                row['resources'].append(sample)
            scan()
            if progress:
                progress(ProgressEvent('stress_jobs', index + 1, warmup + repeats))
        row['coverage']['completed_jobs'] = warmup + repeats
        if observer is not None:
            samples = row['resources']
            if any(s.get('status') != 'available' or type(s.get('current_rss_bytes')) is not int for s in samples):
                raise _Unavailable('Current process-tree RSS incomplete; memory growth is not verified')
            baseline = samples[warmup - 1]['current_rss_bytes']
            values = [s['current_rss_bytes'] for s in samples[warmup:]]
            threshold = max(128 * 1024 ** 2, baseline * .1)
            n = len(values)
            slope = sum((i - (n - 1) / 2) * v for i, v in enumerate(values)) / sum((i - (n - 1) / 2) ** 2 for i in range(n))
            row['memory_screen'] = {'warm_rss_bytes': baseline, 'threshold_bytes': threshold,
                                    'max_growth_bytes': max(values) - baseline, 'slope_bytes_per_job': slope,
                                    'within_threshold': max(values) - baseline <= threshold,
                                    'positive_growth_trend': slope > 0, 'review_required': slope > 0,
                                    'acceptance_verdict': None}
            if max(values) - baseline > threshold:
                raise RuntimeError('Current RSS growth exceeds the declared T5 screen threshold')
            if slope > 0:
                raise _Unavailable('Positive current-RSS growth trend requires explanation; memory stability is not verified')
        row.update(status='measured', success=True)
    except _Unavailable as exc:
        row.update(status='not_verified', reason=str(exc)[:2048])
    except (CancelledError, KeyboardInterrupt) as exc:
        row.update(status='cancelled', error=str(exc)[:2048])
    except Exception as exc:
        logging.getLogger(__name__).debug('Stress scenario failed', exc_info=True)
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:2048])
    finally:
        for client in clients:
            owner = {'pid': client.process.pid, 'generation': getattr(client, 'generation', None)}
            try:
                clean = client.close()
                owner.update(clean_stop=clean, reaped=client.process.poll() is not None,
                             error=getattr(client, 'cleanup_error', None))
                verified = owner['reaped'] and not owner['error'] and (clean or row.get('deliberate_interruption', {}).get('pid') == owner['pid'])
            except Exception as exc:
                logging.getLogger(__name__).debug('Stress cleanup failed', exc_info=True)
                owner['error'] = str(exc)[:512]
                verified = False
            row['cleanup']['owners'].append(owner)
            row['cleanup']['verified'] &= verified
        if not row['cleanup']['verified'] and row['status'] == 'measured':
            row.update(status='failed', success=False, error='owned worker cleanup was not verified')
        if admitted and root.is_dir() and not root.is_symlink():
            try:
                files, size = _inventory(root, lambda: None, cap)
                row['partial_paths'] = list(files)[:4096]
                row['artifact_bytes'] = size
            except Exception as exc:
                logging.getLogger(__name__).debug('Stress inventory failed', exc_info=True)
                row['artifact_inventory_error'] = str(exc)[:512]
        row['duration_seconds'] = time.monotonic() - start
    return row
