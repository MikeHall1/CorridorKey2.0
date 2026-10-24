"""Separate owned profiled resource attempts; their timing is never throughput evidence."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import benchmark
from .asset_inventory import AssetSelection
from .batch import jobs_from_data
from .benchmark_report import read_report
from .calibration import _model_files, canonical
from .calibration_task import _metrics, _size
from .control import CancellationToken, CancelledError
from .identities import sequence_identity
from .local_json import read_object
from .preview_sequence import scan_sequence
from .project_collection import (
    FilePin,
    admit_file,
    assert_pin,
    fresh_directory,
    read_file,
    real_path,
)
from .runtime_paths import RuntimePaths
from .worker_client import WorkerClient
from .worker_protocol import job_to_wire

METRICS = {'peak_allocated_vram_bytes', 'peak_reserved_vram_bytes'}
RESOURCE_SCOPE = 'processing after model load; process allocation peaks, not whole-device usage'
RESET_SCOPE = 'reset before current processing, after model load; resident allocations/reservations remain included'
MAX_RECEIPT_BYTES = 32 * 1024**2


def validate_policy(value):
    if not isinstance(value, dict) or set(value) != {'constraints', 'limits'}:
        raise ValueError('Resource policy requires constraints and explicit attempt limits')
    limits = value['limits']
    bounds = {'max_jobs': 1024, 'max_frames': 122880, 'max_output_bytes': 2**40,
              'read_budget': 2**40, 'max_files': 100000}
    if (not isinstance(limits, dict) or set(limits) != {*bounds, 'max_wall_seconds'}
            or any(type(limits[k]) is not int or not 1 <= limits[k] <= high for k, high in bounds.items())
            or type(limits['max_wall_seconds']) not in (int, float)
            or not math.isfinite(limits['max_wall_seconds']) or not 0 < limits['max_wall_seconds'] <= 86400):
        raise ValueError('Invalid bounded resource-phase limits')
    constraints = value['constraints']
    if not isinstance(constraints, list) or not 1 <= len(constraints) <= 32:
        raise ValueError('Resource policy requires1..32 per-job allocator constraints')
    seen = set()
    for row in constraints:
        if (not isinstance(row, dict) or set(row) != {'job', 'metric', 'maximum_bytes'}
                or type(row['job']) is not int or row['job'] < 0 or row['metric'] not in METRICS
                or type(row['maximum_bytes']) is not int or not 1 <= row['maximum_bytes'] <= 2**63 - 1
                or (row['job'], row['metric']) in seen):
            raise ValueError('Resource limits require distinct jobs, supported allocator metric and positive byte threshold')
        seen.add((row['job'], row['metric']))


def _live(reference, *, check=None, max_bytes=None, query_environment=True):
    """Use the existing benchmark producer; do not fabricate absent driver metadata."""
    repository = benchmark.repository_identity(check=check, max_bytes=max_bytes)
    if not query_environment:
        expected = reference['environment']
        if repository != reference['repository']:
            raise ValueError('Current implementation differs from the original unprofiled pair')
        return {'repository': repository, 'environment': {k: expected[k] for k in ('python', 'os', 'dependencies')},
                'cpu': expected['hardware'].get('cpu'), 'ram_bytes': expected['hardware'].get('ram_bytes'),
                'driver': expected['hardware'].get('driver')}
    env = benchmark.environment()
    driver = benchmark.command('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader').decode().strip()
    expected = reference['environment']
    hardware = expected['hardware']
    if (not driver or driver != hardware.get('driver') or repository != reference['repository']
            or any(env.get(k) != expected.get(k) for k in ('python', 'os', 'dependencies'))
            or any(env['hardware'].get(k) != hardware.get(k) for k in ('cpu', 'ram_bytes'))):
        raise ValueError('Live implementation/environment/driver differs from the unprofiled pair')
    return {'repository': repository, 'environment': {k: env[k] for k in ('python', 'os', 'dependencies')},
            'cpu': env['hardware'].get('cpu'), 'ram_bytes': env['hardware'].get('ram_bytes'), 'driver': driver}


def _files(path, maximum, check):
    root = real_path(path)
    if root.is_file():
        return [admit_file(root)]
    index = scan_sequence(root, check=check, max_entries=maximum, max_frames=maximum)
    return [admit_file(p) for p in index.paths]


def _membership_fence(membership, maximum, check):
    for root, expected in membership['sources'].items():
        if check is not None:
            check()
        current = _files(root, maximum, check)
        kind = 'sequence' if Path(root).is_dir() else 'file'
        if kind != expected['kind'] or [p.path for p in current] != expected['files']:
            raise ValueError('Source/hint membership changed')
    for expected in membership['models']:
        current, _ = _model_files(AssetSelection(Path(expected['root']), Path(expected['weights'])), check)
        if [p.path for p in current] != expected['files']:
            raise ValueError('Model membership changed')


def _record(record, job, original, reused):
    diagnostics = record.get('diagnostics', {})
    source = original['diagnostics']['source']
    measured = _metrics(record, job, original['diagnostics']['frames_submitted'],
                        'warm_session' if reused else 'cold_process', source['native_hw'])
    if (not measured['neural_gpu_verified'] or diagnostics.get('profiling_enabled') is not True
            or diagnostics.get('source') != source or diagnostics.get('execution') != original['diagnostics']['execution']
            or diagnostics.get('output_contracts') != original['diagnostics']['output_contracts']
            or record.get('hints') != original.get('hints') or record.get('asset_paths') != original.get('asset_paths')):
        raise ValueError('Profiled record lacks matched actual neural CUDA/source/colour/model/profile evidence')
    if not any(s.get('method') == 'synchronized_wall' and s.get('status') == 'complete' for s in diagnostics.get('stages', [])):
        raise ValueError('Completed synchronized stage evidence is missing')
    resources = diagnostics.get('resources')
    execution = diagnostics['execution']
    if (not isinstance(resources, dict) or resources.get('scope') != RESOURCE_SCOPE or resources.get('reset_scope') != RESET_SCOPE
            or resources.get('device_index') != execution.get('device_index')
            or resources.get('reset_device_index') != execution.get('device_index')
            or resources.get('gpu') != execution.get('gpu') or resources.get('cuda_version') != execution.get('cuda_version')):
        raise ValueError('Allocator reset/device/scope evidence is missing or mismatched')
    allocated, reserved = (resources.get(k) for k in ('peak_allocated_vram_bytes', 'peak_reserved_vram_bytes'))
    if any(type(v) is not int or not 0 < v <= 2**63 - 1 for v in (allocated, reserved)):
        raise ValueError('Allocator observations require positive byte integers')
    assert isinstance(allocated, int) and isinstance(reserved, int)
    if reserved < allocated:
        raise ValueError('Allocator reserved bytes must be >= allocated bytes')
    expected_settings = dict(original['requested_settings'])
    expected_settings['temp_dir'] = json.loads(job.config_json)['temp_dir']
    profiles = expected_settings.get('shot_profiles')
    if isinstance(profiles, dict) and profiles.get('enabled'):
        from .shot_profiles import execution_clone
        template = replace(job, input_path=original['input'], output_dir=original['output_dir'],
            config_json=canonical(original['requested_settings']).decode(),
            writer_json=canonical(original['writer_settings']).decode(),
            initial_hint_path=original['hints']['first_frame'], hint_video_path=original['hints']['full'],
            profile_stages=False)
        expected = execution_clone(template, owned_root=Path(job.output_dir).parent,
            temp_dir=Path(expected_settings['temp_dir']), output_dir=Path(job.output_dir), profile_stages=True)
        if (job.input_path, job.writer_json, job.initial_hint_path, job.hint_video_path) != (
                expected.input_path, expected.writer_json, expected.initial_hint_path, expected.hint_video_path):
            raise ValueError('Resource clone changed original shot-plan source/writer/conditioning')
        expected_settings = json.loads(expected.config_json)
    if json.loads(job.config_json) != expected_settings:
        raise ValueError('Resource phase changed non-ephemeral requested settings')
    return resources


def _validate_original(original, template):
    """Benchmark outputs relocate; validate that actual owner without rekeying the template."""
    diagnostics = original['diagnostics']
    if diagnostics.get('profiling_enabled') is not False:
        raise ValueError('Original throughput must remain unprofiled')
    if json.loads(template.config_json) != original['requested_settings'] or original.get('hints') != {
            'first_frame': template.initial_hint_path, 'full': template.hint_video_path}:
        raise ValueError('Frozen source/settings/hints cannot be reproduced exactly')
    bound = replace(template, output_dir=original['output_dir'])
    measured = _metrics(original, bound, diagnostics['frames_submitted'],
                        'warm_session' if original['model_reused'] else 'cold_process', diagnostics['source']['native_hw'])
    if not measured['neural_gpu_verified']:
        raise ValueError('Original CPU/unknown provider cannot qualify for neural memory acceptance')


def _prepare(report_path, policy_path, policy_sha256, check, *, query_environment=True):
    from .regression_gate import _policy, context, digest

    policy_pin, pair_pin = admit_file(policy_path), admit_file(report_path)
    policy, policy_raw = _policy(policy_path, policy_sha256)
    if policy['schema_version'] not in (3, 4) or 'resources' not in policy:
        raise ValueError('Resource phase requires an unchanged predeclared version3/4 resource policy')
    report, raw = read_report(Path(pair_pin.path), regular_file=True)
    if report.get('kind') != 'inference_pair' or report.get('status') != 'complete' or digest(context(report)) != policy['context_sha256']:
        raise ValueError('Resource phase requires the exact completed unprofiled pair context')
    limits = policy['resources']['limits']
    reference = report['pairs'][0]['samples']['baseline']
    snapshots = None
    source_roots = set()
    membership = {'sources': {}, 'models': []}
    pins = {p.path: p for p in (pair_pin, policy_pin)}
    expected_hashes = {pair_pin.path: hashlib.sha256(raw).hexdigest(), policy_pin.path: hashlib.sha256(policy_raw).hexdigest()}
    jobs_by_side, asset_roots = {}, {}
    jobs_count, frames_count = 0, 0
    for side in ('baseline', 'candidate'):
        frozen = report['provenance'][side]['normalized_batch']
        jobs, _ = jobs_from_data({'schema_version': 1, 'kind': 'inference_batch', **frozen}, Path(report_path))
        originals = report['pairs'][0]['samples'][side]['runs'][0]['jobs']
        for job, original in zip(jobs, originals, strict=True):
            config, _ = job.configurations()
            if not Path(job.input_path).is_dir():
                raise ValueError('Resource phase initially supports explicit image sequences only; video is not profiled')
            if config.shot_profiles is not None and config.shot_profiles['enabled']:
                from .shot_profiles import ProfilePlan
                ProfilePlan.parse(config.shot_profiles).validate_recipe(job)
            _validate_original(original, job)
            paths = original.get('asset_paths')
            if not isinstance(paths, dict) or 'foundation_stats' not in paths:
                raise ValueError('Original actual model path metadata is missing')
            root = real_path(Path(paths['foundation_stats']).parent.parent, directory=True)
            runtime_paths = RuntimePaths.from_package(benchmark.ROOT, paths['weights'], asset_root=root)
            if paths != {k: str(getattr(runtime_paths, k)) for k in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir')}:
                raise ValueError('Original model roots cannot be reproduced')
            asset_roots.setdefault(side, str(root))
            if asset_roots[side] != str(root):
                raise ValueError('Per-side model roots drift')
            models, _ = _model_files(AssetSelection(root, Path(paths['weights'])), check)
            model_member = {'root': str(root), 'weights': paths['weights'], 'files': [p.path for p in models]}
            if model_member not in membership['models']:
                membership['models'].append(model_member)
            if {p.path for p in models} != set(reference['weights']):
                raise ValueError('Actual model membership differs from paired weights')
            for pin in models:
                pins.setdefault(pin.path, pin)
            expected_hashes.update(reference['weights'])
            for source in (job.input_path, job.initial_hint_path, job.hint_video_path):
                if source is not None:
                    source_roots.add(source)
                    source_pins = _files(source, limits['max_files'], check)
                    membership['sources'][source] = {'kind': 'sequence' if Path(source).is_dir() else 'file',
                                                      'files': [p.path for p in source_pins]}
                    for pin in source_pins:
                        pins.setdefault(pin.path, pin)
            if config.corrections_path is not None:
                correction = report['pairs'][0]['samples'][side]['corrections'][originals.index(original)]
                if not isinstance(correction, dict) or correction.get('path') != str(config.corrections_path):
                    raise ValueError('Original correction provenance is missing')
                bindings = [(correction['path'], correction['manifest_sha256'])]
                bindings.extend((asset['path'], asset['sha256']) for asset in correction['assets'])
                for path, expected_sha in bindings:
                    pin = admit_file(path)
                    prior = expected_hashes.get(pin.path)
                    if prior is not None and prior != expected_sha:
                        raise ValueError('Conflicting canonical correction/content identities')
                    expected_hashes[pin.path] = expected_sha
                    pins.setdefault(pin.path, pin)
        jobs_by_side[side] = jobs
    repository_names = set(benchmark.command('git', 'ls-files', '-z').split(b'\0')) | set(
        benchmark.command('git', 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0'))
    if len(repository_names) > limits['max_files']:
        raise ValueError('Resource implementation inventory exceeds file budget')
    for name in repository_names - {b''}:
        path = benchmark.ROOT / name.decode()
        if path.exists():
            pin = admit_file(path)
            pins.setdefault(pin.path, pin)
    for path in sorted(Path(__file__).parent.glob('*.py')):
        pin = admit_file(path)
        pins.setdefault(pin.path, pin)
    if any(c['job'] >= len(jobs) for c in policy['resources']['constraints'] for jobs in jobs_by_side.values()):
        raise ValueError('Resource policy selects a missing original job')
    for pair in report['pairs']:
        for side, sample in pair['samples'].items():
            warm = sample['model_state'] == 'warm_session'
            for index, original in enumerate(sample['runs'][0]['jobs']):
                _validate_original(original, jobs_by_side[side][index])
                first = report['pairs'][0]['samples'][side]['runs'][0]['jobs'][index]
                if (original.get('asset_paths') != first.get('asset_paths')
                        or original['diagnostics']['source'] != first['diagnostics']['source']
                        or original['diagnostics']['output_contracts'] != first['diagnostics']['output_contracts']):
                    raise ValueError('Repeated source/model/output contracts differ from the verified original')
                jobs_count += 2 if warm else 1
                frames_count += original['diagnostics']['frames_submitted'] * (2 if warm else 1)
                execution = original['diagnostics']['execution']
                hw = sample['environment']['hardware']
                if (execution.get('gpu') != hw.get('gpu') or execution.get('device_index') != hw.get('device_index')
                        or execution.get('cuda_version') != hw.get('cuda_version')):
                    raise ValueError('Unprofiled selected GPU identity is incomplete')
    if jobs_count > limits['max_jobs'] or frames_count > limits['max_frames'] or len(pins) > limits['max_files']:
        raise ValueError('Resource declared job/frame/file workload exceeds budget')
    required = 6 * sum(p.size for p in pins.values()) + 3 * limits['max_output_bytes']
    if required > limits['read_budget']:
        raise ValueError('Resource provenance verification read budget exceeded')
    verified = []
    for pin in sorted(pins.values(), key=lambda p: p.path):
        sha, _ = read_file(pin, check=check)
        if pin.path in expected_hashes and sha != expected_hashes[pin.path]:
            raise ValueError('Resource source/model/report/policy content differs from the original pair')
        verified.append(replace(pin, sha256=sha))
    if set(reference['assets']) != source_roots:
        raise ValueError('Original asset inventory differs from actual source/hint roots')
    verified_hashes = {p.path: p.sha256 for p in verified}
    for path, expected in reference['assets'].items():
        member = membership['sources'][path]
        if member['kind'] == 'file':
            actual = {'path': path, 'sha256': verified_hashes[member['files'][0]]}
        else:
            actual = sequence_identity(path, [{'name': Path(p).name, 'sha256': verified_hashes[p]} for p in member['files']])
        if actual != expected:
            raise ValueError('Original source/hint content changed')
    _membership_fence(membership, limits['max_files'], check)
    for pin in verified:
        assert_pin(pin)
    snapshots = _live(reference, check=check, max_bytes=limits['read_budget'], query_environment=query_environment)
    return policy, report, jobs_by_side, asset_roots, verified, snapshots, required, membership


@dataclass(frozen=True)
class ResourceAttempt:
    payload_json: bytes
    report_path: Path
    retained_owner: object = None

    def to_dict(self):
        return json.loads(self.payload_json)


def run_phase(report_path, policy_path, policy_sha256, output_parent, *, worker_factory=None,
              cancellation=None, clock=time.monotonic):
    """Explicit separate profiled attempt. Injected providers never qualify as production evidence."""
    token = cancellation if cancellation is not None else CancellationToken()
    token.check()
    started = clock()
    from .regression_gate import _policy
    declared, _ = _policy(policy_path, policy_sha256)
    def admission_check():
        token.check()
        if clock() - started > declared['resources']['limits']['max_wall_seconds']:
            raise TimeoutError('Resource admission/execution wall budget exceeded')
    policy, pair, jobs, roots, pins, before, read_bytes, membership = _prepare(report_path, policy_path, policy_sha256, admission_check)
    from .output_paths import prepare_report_destination

    parent = real_path(output_parent, directory=True)
    # A missing sentinel admits the entire new owned attempt, never an original asset/export root.
    protected = [p.path for p in pins]
    directories = [j['output_dir'] for p in pair['pairs'] for s in p['samples'].values() for j in s['runs'][0]['jobs']]
    for observation in pair['pairs']:
        for sample in observation['samples'].values():
            for original in sample['runs'][0]['jobs']:
                directories.append(original['asset_paths']['cradio_dir'])
    for side_jobs in jobs.values():
        for job in side_jobs:
            if Path(job.input_path).is_dir():
                directories.append(job.input_path)
    guard = prepare_report_destination(parent / '.resource-admission', (), protected_paths=protected, protected_directories=directories)
    guard()
    root = fresh_directory(parent, 'resources-')
    receipt = {'schema_version': 1, 'kind': 'regression_resource_evidence', 'status': 'running',
               'policy_sha256': policy_sha256, 'pair_sha256': benchmark.sha256(report_path), 'context_sha256': policy['context_sha256'],
               'producer': 'owned_production_worker' if worker_factory is None else 'fixture_provider', 'attempt_root': str(root),
               'before': before, 'after': None, 'membership': membership, 'inputs': [asdict(p) for p in pins], 'verification_read_bytes': read_bytes,
               'limits': policy['resources']['limits'], 'trials': [], 'cleanup_verified': True, 'error': None,
               'scope': 'Separate synchronized per-job CUDA allocator peaks; no resource-phase timing is throughput evidence, RSS, whole-device usage or a hard cap.'}
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    current = None
    factory = worker_factory if worker_factory is not None else WorkerClient

    def check():
        token.check()
        info = real_path(root, directory=True).stat()
        if (info.st_dev, info.st_ino) != root_identity:
            raise ValueError('Owned resource attempt root changed')
        if clock() - started > receipt['limits']['max_wall_seconds']:
            raise TimeoutError('Resource wall budget exceeded')
        _size(root, receipt['limits']['max_output_bytes'])
        for pin in pins:
            assert_pin(pin)

    def run(client, original, template, pair_index, side, job_index, phase, reused):
        check()
        directory = root / f'{len(receipt["trials"]):04d}-{phase}'
        directory.mkdir()
        from .shot_profiles import execution_clone
        job = execution_clone(template, owned_root=root, temp_dir=directory / 'scratch',
                              output_dir=directory / 'outputs', profile_stages=True)
        request = client.submit_preview(job)
        while True:
            try:
                check()
            except BaseException:
                try:
                    client.cancel(request, 'Resource phase cancelled/budget exceeded')
                except (OSError, RuntimeError, ValueError, BufferError):
                    logging.getLogger(__name__).exception('Resource cancellation dispatch failed; close retains retirement ownership')
                raise
            try:
                event = client.next_event(timeout=.1)
            except TimeoutError:
                continue
            if event.get('request_id') != request or event.get('kind') not in ('result', 'failed', 'cancelled', 'rejected', 'lost'):
                continue
            record = event.get('payload', {}).get('record')
            if not isinstance(record, dict):
                raise TypeError(f"Resource worker {event['kind']} lacks an actual job record")
            path = root / f'record-{len(receipt["trials"]):04d}.json'
            raw = canonical(record, 4 * 1024**2)
            with path.open('xb') as stream:
                stream.write(raw)
            receipt['trials'].append({'pair': pair_index, 'side': side, 'job': job_index, 'phase': phase,
                'original_job_id': original['job_id'], 'profiled_job_id': record.get('job_id'), 'job_request': job_to_wire(job),
                'record': {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}, 'status': 'not_verified', 'resources': None})
            if event['kind'] != 'result':
                raise ValueError(f"Resource worker {event['kind']}; raw diagnostic record retained")
            resources = _record(record, job, original, reused)
            receipt['trials'][-1].update(status='complete', resources=resources)
            check()
            return

    try:
        for pair_index, observation in enumerate(pair['pairs']):
            for side in observation['order']:
                check()
                _membership_fence(membership, receipt['limits']['max_files'], admission_check)
                current = factory(capacity=1, weights=observation['samples'][side]['runs'][0]['jobs'][0]['asset_paths']['weights'],
                                  asset_root=roots[side], startup_timeout=min(10., max(.1, receipt['limits']['max_wall_seconds'] - (clock() - started))), shutdown_timeout=5.)
                try:
                    sample = observation['samples'][side]
                    warm = sample['model_state'] == 'warm_session'
                    if warm:
                        for i, template in enumerate(jobs[side]):
                            run(current, sample['runs'][0]['jobs'][i], template, pair_index, side, i, 'warmup', i > 0)
                    for i, template in enumerate(jobs[side]):
                        run(current, sample['runs'][0]['jobs'][i], template, pair_index, side, i, 'measured', warm or i > 0)
                finally:
                    try:
                        retired = current.close()
                    except BaseException:
                        receipt['cleanup_verified'] = False
                        raise
                    if not retired or getattr(current, 'cleanup_error', None):
                        receipt['cleanup_verified'] = False
                        raise RuntimeError('Resource worker retirement NOT VERIFIED')
                    current = None
        check()
        for pin in pins:
            sha, _ = read_file(pin, check=admission_check)
            if sha != pin.sha256:
                raise ValueError('Resource pinned bytes changed during execution')
        _membership_fence(membership, receipt['limits']['max_files'], admission_check)
        receipt['after'] = _live(pair['pairs'][0]['samples']['baseline'], check=admission_check, max_bytes=receipt['limits']['read_budget'])
        check()
        if receipt['after'] != before:
            raise ValueError('Resource provenance changed during execution')
        receipt['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        logging.getLogger(__name__).exception('Resource phase stopped')
        if getattr(exc, 'cleanup_error', None):
            receipt['cleanup_verified'] = False
        receipt['status'] = 'cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'not_verified'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
    artifacts = []
    receipt['artifact_verification'] = 'not_verified'
    if receipt['status'] == 'complete':
        try:
            _size(root, receipt['limits']['max_output_bytes'])
            for index, path in enumerate(root.rglob('*')):
                admission_check()
                if index >= receipt['limits']['max_files']:
                    raise ValueError('Resource artifact file-count budget exceeded')
                real_path(path)
                if path.is_file():
                    pin = admit_file(path)
                    sha, _ = read_file(pin, check=admission_check)
                    artifacts.append(asdict(replace(pin, sha256=sha)))
            receipt['artifact_verification'] = 'complete'
        except (Exception, KeyboardInterrupt) as exc:
            logging.getLogger(__name__).exception('Resource artifact finalization failed')
            receipt.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'not_verified',
                           error=f'Artifact finalization: {type(exc).__name__}: {exc}')
    receipt['artifacts'] = artifacts
    receipt['phase_elapsed_seconds'] = clock() - started
    destination = root / 'resource-evidence.json'
    guard()
    try:
        for pin in pins:
            assert_pin(pin)
    except (ValueError, OSError) as exc:
        receipt.update(status='not_verified', error=f'Input finalization: {exc}')
    info = real_path(root, directory=True).stat()
    if (info.st_dev, info.st_ino) != root_identity:
        raise ValueError('Owned resource attempt changed before publication')
    for artifact in artifacts:
        assert_pin(FilePin(artifact['path'], tuple(artifact['fingerprint']), artifact['sha256']))
    receipt['error'] = receipt['error'][:4096] if isinstance(receipt['error'], str) else None
    raw = canonical(receipt, MAX_RECEIPT_BYTES)
    if _size(root, receipt['limits']['max_output_bytes']) + len(raw) > receipt['limits']['max_output_bytes']:
        raise ValueError('Resource attempt has no remaining output budget for its terminal receipt; artifacts retained')
    with destination.open('xb') as stream:
        stream.write(raw)
    return ResourceAttempt(raw, destination, current)


def summarize(rows, constraints, pairs):
    """Arithmetic over admitted measured allocator observations only, no RSS aliases."""
    results = []
    for constraint in constraints:
        selected = [r for r in rows if r['phase'] == 'measured' and r['job'] == constraint['job']]
        if len(selected) != pairs * 2:
            raise ValueError('Missing resource job/side/repetition coverage')
        values = [r['resources'][constraint['metric']] for r in selected]
        peak = max(values)
        results.append({**constraint, 'observed_max_bytes': peak, 'raw_bytes': values,
                        'status': 'pass' if peak <= constraint['maximum_bytes'] else 'fail', 'scope': RESOURCE_SCOPE})
    return results


def admit(receipt_path, receipt_sha256, report_path, policy_path, policy_sha256, *, check=None):
    """Offline receipt admission never queries a live GPU or substitutes RSS."""
    pin = admit_file(receipt_path)
    receipt, raw = read_object(Path(pin.path), max_bytes=MAX_RECEIPT_BYTES, max_depth=128, regular_file=True)
    if hashlib.sha256(raw).hexdigest() != receipt_sha256:
        raise ValueError('Resource receipt byte identity differs')
    if (type(receipt.get('schema_version')) is not int or receipt.get('schema_version') != 1 or receipt.get('kind') != 'regression_resource_evidence'
            or receipt.get('policy_sha256') != policy_sha256 or receipt.get('producer') != 'owned_production_worker'
            or receipt.get('status') != 'complete' or receipt.get('cleanup_verified') is not True
            or receipt.get('artifact_verification') != 'complete'):
        raise ValueError('Resource receipt is incomplete, fixture/unknown producer or cleanup is not verified')
    policy, pair, jobs, _roots, inputs, signature, _required, membership = _prepare(
        report_path, policy_path, policy_sha256, check, query_environment=False)
    expected_inputs = [asdict(p) for p in inputs]
    if (receipt.get('inputs') != json.loads(json.dumps(expected_inputs)) or receipt.get('before') != signature
            or receipt.get('after') != signature or receipt.get('limits') != policy['resources']['limits']
            or receipt.get('membership') != membership or receipt.get('verification_read_bytes') != _required
            or receipt.get('pair_sha256') != benchmark.sha256(report_path) or receipt.get('context_sha256') != policy['context_sha256']):
        raise ValueError('Resource receipt input/code/environment/driver/workload bindings differ')
    root = real_path(receipt['attempt_root'], directory=True)
    if not Path(pin.path).is_relative_to(root):
        raise ValueError('Resource receipt escapes its owned attempt')
    artifacts = receipt.get('artifacts')
    if not isinstance(artifacts, list) or len(artifacts) > receipt['limits']['max_files']:
        raise ValueError('Missing/excessive resource artifact inventory')
    protected = [asdict(p) for p in inputs] + [asdict(replace(pin, sha256=receipt_sha256))]
    seen_artifacts = set()
    for asset in artifacts:
        artifact = FilePin(asset['path'], tuple(asset['fingerprint']), asset['sha256'])
        if artifact.path in seen_artifacts or not real_path(artifact.path).is_relative_to(root):
            raise ValueError('Duplicate or escaping resource artifact')
        seen_artifacts.add(artifact.path)
        sha, _ = read_file(artifact, check=check)
        if sha != artifact.sha256:
            raise ValueError('Resource artifact changed')
        protected.append(asdict(artifact))
    rows = receipt.get('trials')
    if not isinstance(rows, list) or not 1 <= len(rows) <= receipt['limits']['max_jobs']:
        raise ValueError('Resource trials are missing/excessive')
    expected = []
    for observation in pair['pairs']:
        for side in observation['order']:
            sample = observation['samples'][side]
            warm = sample['model_state'] == 'warm_session'
            if warm:
                expected.extend((observation['index'], side, i, 'warmup', i > 0) for i in range(len(jobs[side])))
            expected.extend((observation['index'], side, i, 'measured', warm or i > 0) for i in range(len(jobs[side])))
    if len(rows) != len(expected):
        raise ValueError('Resource trial lifecycle coverage differs')
    used_ids, used_outputs = set(), set()
    for row, (pair_index, side, job_index, phase, reused) in zip(rows, expected, strict=True):
        if check is not None:
            check()
        if tuple(row.get(k) for k in ('pair', 'side', 'job', 'phase')) != (pair_index, side, job_index, phase) or row.get('status') != 'complete':
            raise ValueError('Resource order/side/job/phase coverage differs')
        original = pair['pairs'][pair_index]['samples'][side]['runs'][0]['jobs'][job_index]
        from .worker_protocol import job_from_wire
        job = job_from_wire(row['job_request'])
        if not job.profile_stages or not Path(job.output_dir).is_relative_to(root):
            raise ValueError('Resource job is unprofiled or escapes owned attempt')
        if row['record']['path'] not in seen_artifacts:
            raise ValueError('Raw resource record is not an admitted artifact')
        raw_record, record_bytes = read_object(Path(row['record']['path']), max_bytes=4 * 1024**2, regular_file=True)
        if hashlib.sha256(record_bytes).hexdigest() != row['record']['sha256']:
            raise ValueError('Raw resource record hash differs')
        if (row.get('original_job_id') != original['job_id'] or row.get('profiled_job_id') != raw_record.get('job_id')
                or raw_record.get('job_id') in used_ids or job.output_dir in used_outputs):
            raise ValueError('Resource original/new job identity or fresh destination differs')
        used_ids.add(raw_record['job_id']); used_outputs.add(job.output_dir)
        if _record(raw_record, job, original, reused) != row['resources']:
            raise ValueError('Resource observations differ from actual raw record')
    _size(root, receipt['limits']['max_output_bytes'])
    _membership_fence(membership, receipt['limits']['max_files'], check)
    for p in inputs:
        assert_pin(p)
    assert_pin(pin)
    observations = summarize(rows, policy['resources']['constraints'], len(pair['pairs']))
    return {'name': 'observed_memory', 'status': 'fail' if any(r['status'] == 'fail' for r in observations) else 'pass',
            'observations': observations, 'receipt_sha256': receipt_sha256, 'scope': RESOURCE_SCOPE}, protected, str(root)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--output-parent', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        outcome = run_phase(args.report, args.policy, args.policy_sha256, args.output_parent)
        result = outcome.to_dict()
        print(json.dumps({'status': result['status'], 'report_path': str(outcome.report_path), 'error': result['error']}))
        return 130 if result['status'] == 'cancelled' else 0 if result['status'] == 'complete' else 1
    except (Exception, KeyboardInterrupt) as exc:
        logging.getLogger(__name__).exception('Resource admission failed')
        print(json.dumps({'status': 'cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'not_verified',
                          'error': f'{type(exc).__name__}: {exc}'}))
        return 130 if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
