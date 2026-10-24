"""Explicit bounded production-stress plans over the existing shared workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .batch import jobs_from_data
from .control import CancellationToken, CancelledError, ProgressEvent
from .diagnostics import json_value, write_json_atomic
from .local_json import read_object, read_regular_bytes
from .runtime_paths import RuntimePaths

SCENARIOS = ('long_clip', 'batch50', 'repeated', 'concurrent_isolated', 'cancel',
             'interruption_recovery', 'memory_growth', 'encoder_failure', 'disk_full')
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_SOURCE_FILES = 100000
MAX_METADATA_BYTES = 128 * 1024 * 1024
MAX_BYTES = (1 << 63) - 1


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower},{upper}]')
    return value


def _duration(value, upper, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= upper:
        raise ValueError(f'{name} must be finite in [1,{upper}] seconds')
    return value


def _path(value, parent):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError('Expected a nonempty local path of at most4096characters')
    return str((parent / value).expanduser().resolve())


def load_stress_manifest(path):
    path = Path(path).expanduser().resolve()
    data, raw = read_object(path, max_bytes=1024 * 1024, max_depth=32, regular_file=True)
    if (set(data) - {'schema_version', 'kind', 'limits', 'cases', 'asset_root'}
            or type(data.get('schema_version')) is not int or data['schema_version'] != 1
            or data.get('kind') != 'production_stress'):
        raise ValueError('Expected schema_version1/kindproduction_stress and known fields')
    supplied = data.get('limits', {})
    if not isinstance(supplied, dict) or set(supplied) - {'deadline_seconds', 'max_artifact_bytes'}:
        raise ValueError('Unknown suite limit field')
    limits = {'deadline_seconds': _duration(supplied.get('deadline_seconds', 7200), 86400, 'deadline_seconds'),
              'max_artifact_bytes': _integer(supplied.get('max_artifact_bytes', 1024**3), 1, MAX_BYTES, 'max_artifact_bytes'),
              'max_scenarios': 32, 'max_jobs': 64, 'max_evidence_bytes': 4 * 1024**2}
    cases = data.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError('A suite requires1..32 explicit cases')
    normalized, seen = [], set()
    allowed = {'id', 'scenario', 'batch_manifest', 'required', 'repetitions', 'warmup_jobs',
               'timeout_seconds', 'cancel_after_readable_frames', 'fault_volume', 'recovery'}
    for case in cases:
        if (not isinstance(case, dict) or set(case) - allowed or
                not isinstance(case.get('id'), str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', case['id']) is None
                or case['id'] in seen or case.get('scenario') not in SCENARIOS
                or type(case.get('required', True)) is not bool):
            raise ValueError('Invalid/duplicate case ID, scenario or required flag')
        seen.add(case['id'])
        scenario = case['scenario']
        memory = scenario == 'memory_growth'
        repeated = scenario in ('repeated', 'memory_growth')
        if not repeated and 'repetitions' in case:
            raise ValueError('repetitions applies only to repeated/memory_growth')
        if not memory and case.get('warmup_jobs', 0) != 0:
            raise ValueError('warmup_jobs applies only to memory_growth')
        if scenario not in ('cancel', 'interruption_recovery') and 'cancel_after_readable_frames' in case:
            raise ValueError('Readable-frame trigger applies only to cancel/interruption')
        if scenario != 'disk_full' and 'fault_volume' in case:
            raise ValueError('fault_volume applies only to disk_full')
        if scenario != 'interruption_recovery' and 'recovery' in case:
            raise ValueError('recovery policy applies only to interruption_recovery')
        recovery = case.get('recovery', {})
        flags = ('durable_foundations', 'durable_rvm', 'durable_hann', 'durable_stage2', 'durable_writer_prefix',
                       'reuse_stage2_finishing_edits', 'reuse_upstream_finishing_edits')
        if not isinstance(recovery, dict) or set(recovery) - {'max_cache_bytes', *flags}:
            raise ValueError('Unknown recovery policy field')
        if any(type(recovery.get(flag, False)) is not bool for flag in flags):
            raise ValueError('Recovery switches must be booleans')
        if recovery.get('reuse_stage2_finishing_edits', False) and not recovery.get('durable_stage2', False):
            raise ValueError('Finishing-edit reuse requires durable_stage2')
        if recovery.get('reuse_upstream_finishing_edits', False) and not any(recovery.get(name, False) for name in
                ('durable_foundations', 'durable_rvm', 'durable_hann')):
            raise ValueError('Upstream finishing-edit reuse requires an explicit upstream durable cache')
        normalized.append({**case, 'required': case.get('required', True),
            'batch_manifest': _path(case.get('batch_manifest'), path.parent),
            'repetitions': _integer(case.get('repetitions', 10 if memory else 3 if repeated else 1),
                                    10 if memory else 1, 128, 'repetitions'),
            'warmup_jobs': _integer(case.get('warmup_jobs', 1 if memory else 0), 1 if memory else 0, 10, 'warmup_jobs'),
            'timeout_seconds': _duration(case.get('timeout_seconds', 1800), 14400, 'timeout_seconds'),
            'cancel_after_readable_frames': _integer(case.get('cancel_after_readable_frames', 1), 1, 64, 'cancel_after_readable_frames'),
            'fault_volume': _path(case['fault_volume'], path.parent) if case.get('fault_volume') is not None else None,
            'recovery': {'max_cache_bytes': _integer(recovery.get('max_cache_bytes', min(256 * 1024**2, limits['max_artifact_bytes'])),
                                0, limits['max_artifact_bytes'], 'max_cache_bytes'),
                         **{flag: recovery.get(flag, False) for flag in flags}}})
    return {'cases': normalized, 'limits': limits,
            'asset_root': _path(data['asset_root'], path.parent) if data.get('asset_root') is not None else None}, raw


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_digest(path, check):
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Stress inputs must be regular local files')
        digest, remaining = hashlib.sha256(), before.st_size
        while remaining:
            check()
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError('Stress input shortened during hashing')
            digest.update(block)
            remaining -= len(block)
        if (stream.read(1) or _identity(before) != _identity(os.fstat(stream.fileno()))
                or _identity(before) != _identity(path.stat())):
            raise ValueError('Stress input grew or changed during hashing')
    return {'sha256': digest.hexdigest(), 'bytes': before.st_size, 'identity': list(_identity(before))}


def _source_digest(path, check):
    path = Path(path).resolve()
    if path.is_file():
        return {'path': str(path), 'files': 1, **_file_digest(path, check)}
    if not path.is_dir():
        raise ValueError('Stress source is neither regular file nor frame directory')
    files = []
    for entry in path.iterdir():
        check()
        if len(files) >= MAX_SOURCE_FILES:
            raise ValueError('Stress source directory entry admission exceeded')
        files.append(entry)
    selected = sorted((item for item in files if item.suffix.lower() in {'.png', '.exr', '.jpg', '.jpeg'}), key=lambda p: p.name)
    if not selected:
        raise ValueError('No supported source frames')
    digest, witness, total = hashlib.sha256(), hashlib.sha256(), 0
    for frame in selected:
        value = _file_digest(frame, check)
        total += value['bytes']
        digest.update(json.dumps({'name': frame.name, 'sha256': value['sha256'], 'bytes': value['bytes']}, sort_keys=True).encode())
        witness.update(json.dumps({'name': frame.name, **value}, sort_keys=True).encode())
    return {'path': str(path), 'files': len(selected), 'bytes': total, 'sha256': digest.hexdigest(), 'witness_sha256': witness.hexdigest()}


def _summary(cases):
    return {'success': bool(cases) and all(not row['required'] or (row['status'] == 'measured' and row['success']) for row in cases)
                and all(row['cleanup']['verified'] for row in cases),
            'expected_cases': len(cases),
            **{name + '_cases': sum(row['status'] == name for row in cases)
               for name in ('measured', 'not_verified', 'failed', 'cancelled')},
            'required_cases': sum(row['required'] for row in cases),
            'required_measured_cases': sum(row['required'] and row['status'] == 'measured' and row['success'] for row in cases),
            'cleanup_verified': all(row['cleanup']['verified'] for row in cases)}


def _row(case, status, reason=None, *, root=None, result=None, seconds=None):
    result = result or {}
    cleanup = result.get('cleanup', {'verified': True, 'owners': []})
    return {'id': case['id'], 'scenario': case['scenario'], 'required': case['required'],
            'status': status, 'success': status == 'measured' and result.get('success') is True,
            'reason': reason, 'error': result.get('error'), 'artifact_dir': str(root) if root is not None else None,
            'cleanup': cleanup, 'evidence': result, 'duration_seconds': seconds}


def _safe_parent(report_path, protected):
    parent = report_path.parent.resolve(strict=True)
    if not parent.is_dir() or report_path.exists() or report_path.is_symlink():
        raise ValueError('Select an existing output parent and a new report destination')
    for raw in protected:
        path = Path(raw).resolve()
        if parent == path or parent.is_relative_to(path) or report_path.resolve() == path:
            raise ValueError('Stress output aliases a protected source/input/output/assets tree')
    return parent


def _declared_paths(batch, parent):
    """Protect declared inputs even if later batch validation finds a missing file."""
    result = []
    if isinstance(batch.get('weights'), str):
        result.append(_path(batch['weights'], parent))
    for job in batch.get('jobs', []) if isinstance(batch.get('jobs'), list) else []:
        if not isinstance(job, dict):
            continue
        for key in ('input', 'output_dir', 'hint_first_frame', 'hint_video'):
            if isinstance(job.get(key), str):
                result.append(_path(job[key], parent))
        settings = job.get('settings')
        if isinstance(settings, dict):
            for key in ('corrections_path', 'temp_dir'):
                if isinstance(settings.get(key), str):
                    result.append(_path(settings[key], parent))
    return result


def _artifact_bytes(roots):
    total, entries = 0, 0
    for root in roots:
        for directory, folders, files in os.walk(root, followlinks=False):
            entries += len(folders) + len(files)
            if entries > MAX_SOURCE_FILES:
                raise ValueError('Stress artifact entry admission exceeded')
            for name in [*folders, *files]:
                path = Path(directory) / name
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                elif not stat.S_ISDIR(info.st_mode):
                    raise ValueError('Stress artifacts contain an unsupported link/device/pipe')
    return total


def _write_new_json(path, data, maximum):
    raw = json.dumps(json_value(data), sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if len(raw) > maximum:
        raise ValueError('Stress evidence admission exceeded before publication')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def _provenance():
    from .benchmark import environment, repository_identity
    result = {'environment': environment()}
    try:
        result['repository'] = repository_identity()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result['repository'] = {'status': 'unavailable', 'reason': str(exc)[:4096]}
    return result


def _asset_paths(weights, asset_root):
    paths = RuntimePaths.from_package(Path(__file__).resolve().parents[1], weights, asset_root=asset_root)
    return [getattr(paths, name) for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir')]


def _asset_membership(path, check):
    members, metadata_bytes = [], 0
    for directory, folders, names in os.walk(path, followlinks=False):
        check()
        for name in [*folders, *names]:
            item = Path(directory) / name
            info = item.lstat()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError('Model asset directory contains a link/device/pipe')
            relative = str(item.relative_to(path))
            metadata_bytes += len(relative.encode()) + 16
            if len(members) >= MAX_SOURCE_FILES or metadata_bytes > MAX_REPORT_BYTES:
                raise ValueError('Model asset directory inventory admission exceeded')
            members.append(('directory' if stat.S_ISDIR(info.st_mode) else 'file', relative))
    return sorted(members)


def _asset_provenance(paths, snapshots, check):
    entries, directories, metadata_bytes, total_members = [], {}, 0, 0
    for path in sorted(set(paths)):
        check()
        if not path.exists():
            entries.append({'path': str(path), 'status': 'missing'})
            continue
        files = [path]
        if path.is_dir():
            inventory = _asset_membership(path, check)
            directories[str(path)] = inventory
            total_members += len(inventory)
            metadata_bytes += len(json.dumps(inventory).encode())
            files = [path / name for kind, name in inventory if kind == 'file']
        else:
            total_members += 1
        if total_members > MAX_SOURCE_FILES or metadata_bytes > MAX_REPORT_BYTES:
            raise ValueError('Aggregate model asset inventory admission exceeded')
        for item in sorted(files):
            name = str(item)
            value = {'path': name, 'files': 1, **_file_digest(item, check)}
            metadata_bytes += len(json.dumps(value).encode())
            if metadata_bytes > MAX_REPORT_BYTES:
                raise ValueError('Aggregate model asset provenance admission exceeded')
            snapshots[name] = value
            entries.append(value)
        if path.is_dir() and _asset_membership(path, check) != directories[str(path)]:
            raise ValueError('Model asset directory membership changed during capture')
    return {'status': 'captured', 'files': entries, 'directories': directories,
            'scope': 'Selected local model file bytes; no tensor compatibility or quality verdict'}


def run_stress_suite(manifest_path, output_path, *, check=None, progress=None,
                     worker_factory=None, observer_factory=None, scenario_runner=None):
    """Execute the declared plan; missing prerequisites never become passing cases."""
    started = time.monotonic()
    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).absolute()
    report: dict[str, Any] = {'schema_version': 1, 'kind': 'production_stress', 'status': 'failed', 'cases': [],
              'manifest': {'path': str(manifest_path), 'sha256': None, 'bytes': None}, 'limits': {},
              'artifact_dir': None, 'scope': 'Injected Python fixtures' if worker_factory or scenario_runner else 'Production shared engine',
              'acceptance_verdict': None, 'created_utc': datetime.now(UTC).isoformat(), 'error': None}
    report['summary'] = _summary([])
    owned_report = False
    deadline = started + 7200
    def tick():
        if check is not None:
            check()
        if time.monotonic() >= deadline:
            raise CancelledError('Stress suite deadline exceeded')
    def persist():
        data = json_value(report)
        raw = json.dumps(data, allow_nan=False, indent=2).encode()
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError('Stress report exceeds16MiB evidence limit')
        if owned_report:
            write_json_atomic(output_path, data)
    cases, prepared, snapshots, protected = [], {}, {}, {str(manifest_path)}
    artifact_roots, selected_assets = [], set()
    try:
        tick()
        plan, raw = load_stress_manifest(manifest_path)
        deadline = started + plan['limits']['deadline_seconds']
        report['manifest'].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        report['limits'] = plan['limits']
        cases = list(plan['cases'])
        configured = {case['scenario'] for case in cases if case['required']}
        cases.extend({'id': '$missing:' + scenario, 'scenario': scenario, 'required': True}
                     for scenario in SCENARIOS if scenario not in configured)
        metadata_bytes = len(raw)
        if plan['asset_root'] is not None:
            protected.add(plan['asset_root'])
        for case in plan['cases']:
            tick()
            batch_path = Path(case['batch_manifest'])
            protected.add(str(batch_path))
            try:
                batch_data, batch_raw = read_object(batch_path, max_bytes=min(4 * 1024**2, MAX_METADATA_BYTES - metadata_bytes), regular_file=True)
                for raw_path in _declared_paths(batch_data, batch_path.parent):
                    protected.add(raw_path)
                metadata_bytes += len(batch_raw)
                if metadata_bytes > MAX_METADATA_BYTES:
                    raise ValueError('Aggregate stress metadata admission exceeded')
                jobs, weights = jobs_from_data(batch_data, batch_path)
                assets = _asset_paths(weights, plan['asset_root'])
                protected.update(str(path) for path in assets)
                selected_assets.update(assets)
                verified_batch = read_regular_bytes(batch_path, max_bytes=len(batch_raw))
                captured_batch = {'path': str(batch_path), 'files': 1, 'bytes': len(batch_raw),
                                  'sha256': hashlib.sha256(batch_raw).hexdigest(),
                                  'identity': list(_identity(batch_path.stat()))}
                if verified_batch != batch_raw:
                    raise ValueError('Batch manifest changed while jobs were frozen')
                snapshots[str(batch_path)] = captured_batch
                if len(jobs) > plan['limits']['max_jobs']:
                    raise ValueError('Stress batch job admission exceeded64')
                selected = {}
                for job in jobs:
                    config, _ = job.configurations()
                    paths = [job.input_path, job.initial_hint_path, job.hint_video_path,
                             config.corrections_path, job.output_dir, config.temp_dir, weights]
                    protected.update(str(Path(p).resolve()) for p in paths if p is not None)
                    for path in (job.input_path, job.initial_hint_path, job.hint_video_path, config.corrections_path):
                        if path is not None:
                            name = str(Path(path).resolve())
                            if name not in snapshots:
                                snapshots[name] = _source_digest(name, tick)
                            selected[name] = snapshots[name]
                case = dict(case, source_identities=[snapshots[str(Path(job.input_path).resolve())] for job in jobs],
                            batch_identity=captured_batch)
                prepared[case['id']] = (case, jobs, weights, None)
            except FileNotFoundError as exc:
                prepared[case['id']] = (case, [], None, ('not_verified', f'Missing local prerequisite: {exc}'))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                prepared[case['id']] = (case, [], None, ('failed', f'Invalid local batch/source: {type(exc).__name__}: {exc}'))
        # Default asset locations remain protected even when every batch is invalid.
        protected.update(str(path) for path in _asset_paths(None, plan['asset_root']))
        parent = _safe_parent(output_path, protected)
        tick()
        asset_provenance: dict[str, Any] = ({'status': 'not_verified', 'scope': 'Injected Python fixtures; production assets not used'}
                            if worker_factory or scenario_runner else _asset_provenance(selected_assets, snapshots, tick))
        # Claim the exact report path exclusively before any scenario starts.
        with output_path.open('xb') as stream:
            stream.write(b'{}\n')
        owned_report = True
        artifact_root = parent / ('stress-' + uuid.uuid4().hex)
        artifact_root.mkdir(mode=0o700)
        artifact_roots.append(artifact_root)
        report['artifact_dir'] = str(artifact_root)
        plan_record = {'manifest': report['manifest'], 'limits': plan['limits'], 'provenance': _provenance(),
                       'model_assets': asset_provenance,
                       'cases': [value[0] for value in prepared.values()],
                       'requested_jobs': {name: [json_value(job) for job in value[1]] for name, value in prepared.items()}}
        _write_new_json(artifact_root / 'frozen-plan.json', plan_record, min(MAX_REPORT_BYTES, plan['limits']['max_artifact_bytes']))
        prerequisite_cache = {}
        production_check = None
        if scenario_runner is None:
            from .stress_runtime import execute_scenario, production_prerequisites
            scenario_runner = execute_scenario
            if worker_factory is None:
                production_check = production_prerequisites
        for index, case in enumerate(cases):
            tick()
            if case['id'] not in prepared:
                row = _row(case, 'not_verified', 'No case configured for this required specification scenario')
            else:
                normalized, jobs, weights, unavailable = prepared[case['id']]
                prerequisite = []
                if unavailable is None and production_check is not None:
                    key = (str(weights) if weights is not None else None, plan['asset_root'])
                    if key not in prerequisite_cache:
                        prerequisite_cache[key] = production_check(weights=weights, asset_root=plan['asset_root'])
                    prerequisite = prerequisite_cache[key]
                if unavailable is not None:
                    row = _row(case, unavailable[0], unavailable[1])
                elif prerequisite:
                    row = _row(case, 'not_verified', '; '.join(prerequisite)[:4096])
                else:
                    used = _artifact_bytes(artifact_roots)
                    remaining_bytes = plan['limits']['max_artifact_bytes'] - used
                    if remaining_bytes <= 0:
                        raise ValueError('Suite artifact ceiling exhausted before next case')
                    case_parent = artifact_root
                    if case['scenario'] == 'disk_full' and normalized['fault_volume'] is not None:
                        volume = Path(normalized['fault_volume'])
                        if volume.is_dir():
                            _safe_parent(volume / ('stress-check-' + uuid.uuid4().hex), protected)
                            case_parent = volume
                    case_root = case_parent / (case['id'] + '-' + uuid.uuid4().hex)
                    normalized = dict(normalized, timeout_seconds=min(normalized['timeout_seconds'], max(1, deadline - time.monotonic())))
                    case_started = time.monotonic()
                    try:
                        case_root.mkdir(mode=0o700)
                        if case_parent != artifact_root:
                            artifact_roots.append(case_root)
                        result = scenario_runner(normalized, jobs, weights, case_root, check=tick, progress=progress,
                            worker_factory=worker_factory, observer_factory=observer_factory,
                            asset_root=plan['asset_root'], limits={**plan['limits'], 'max_artifact_bytes': remaining_bytes})
                        if (not isinstance(result, dict) or result.get('status') not in ('measured', 'not_verified', 'failed', 'cancelled')
                                or type(result.get('success')) is not bool
                                or (result['status'] == 'measured' and result['success'] is not True)):
                            raise TypeError('Scenario driver returned no canonical terminal outcome')
                        cleanup = result.get('cleanup')
                        if (not isinstance(cleanup, dict) or set(cleanup) != {'verified', 'owners'}
                                or type(cleanup['verified']) is not bool or not isinstance(cleanup['owners'], list)):
                            raise TypeError('Scenario driver returned no explicit cleanup evidence')
                        row = _row(case, result['status'], result.get('reason'), root=case_root, result=result,
                                   seconds=time.monotonic() - case_started)
                        reference = _write_new_json(case_root / ('evidence-' + uuid.uuid4().hex + '.json'),
                                                    result, plan['limits']['max_evidence_bytes'])
                        row['evidence'] = {**reference, 'coverage': result.get('coverage', {}),
                                           'scope': result.get('scope'), 'memory_screen': result.get('memory_screen')}
                        if _artifact_bytes(artifact_roots) > plan['limits']['max_artifact_bytes']:
                            row.update(status='failed', success=False, error='Suite cooperative artifact ceiling exceeded')
                    except (CancelledError, KeyboardInterrupt):
                        report['cases'].append(_row(case, 'cancelled', 'Scenario interrupted without terminal cleanup evidence',
                            root=case_root, result={'cleanup': {'verified': False, 'owners': []}}))
                        raise
                    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                        row = _row(case, 'failed', str(exc)[:4096], root=case_root,
                                   result={'error': f'{type(exc).__name__}: {exc}'[:4096],
                                           'cleanup': {'verified': False, 'owners': []}}, seconds=time.monotonic() - case_started)
            report['cases'].append(row)
            report['summary'] = _summary(report['cases'])
            persist()
            if progress is not None:
                progress(ProgressEvent('stress_cases', index + 1, len(cases)))
        for path, before in snapshots.items():
            if _source_digest(path, tick) != before:
                raise ValueError('Original stress source/hint/correction changed during execution')
        for path, before in asset_provenance.get('directories', {}).items():
            if not Path(path).is_dir() or _asset_membership(Path(path), tick) != before:
                raise ValueError('Model asset directory membership changed during execution')
        if hashlib.sha256(read_object(manifest_path, max_bytes=1024 * 1024, max_depth=32, regular_file=True)[1]).hexdigest() != report['manifest']['sha256']:
            raise ValueError('Stress manifest changed during execution')
        report['status'] = 'failed' if any(row['status'] == 'failed' or not row['cleanup']['verified'] or (row['status'] == 'measured' and not row['success']) for row in report['cases']) else 'cancelled' if any(row['status'] == 'cancelled' for row in report['cases']) else 'complete'
        if report['status'] == 'failed':
            report['error'] = 'One or more stress cases failed or cleanup is unverified; inspect case evidence'
    except (CancelledError, KeyboardInterrupt) as exc:
        report['status'], report['error'] = 'cancelled', str(exc)[:4096]
        existing = {row['id'] for row in report['cases']}
        report['cases'].extend(_row(case, 'cancelled', 'Suite cancelled before this case completed') for case in cases if case['id'] not in existing)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'[:4096]
    existing = {row['id'] for row in report['cases']}
    report['cases'].extend(_row(case, 'not_verified', 'Suite aborted before this case ran')
                           for case in cases if case['id'] not in existing)
    report['summary'] = _summary(report['cases'])
    if report['status'] != 'complete':
        report['summary']['success'] = False
    persist()
    return report


def main(argv=None, *, worker_factory=None, observer_factory=None, scenario_runner=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    token = CancellationToken()
    def interrupt(signum, frame):
        token.cancel('Stress suite interrupted')
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        report = run_stress_suite(args.manifest, args.output, check=token.check,
                                 worker_factory=worker_factory, observer_factory=observer_factory,
                                 scenario_runner=scenario_runner)
    finally:
        signal.signal(signal.SIGINT, previous)
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'error': report['error']}, allow_nan=False))
    return 130 if report['status'] == 'cancelled' else 1 if report['status'] == 'failed' else 0 if report['summary']['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
