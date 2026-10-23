"""Opt-in EXR recovery with verified outputs, IID frames and chronological RVM checkpoints."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from .batch import load_jobs
from .control import CancelledError, JobControl
from .diagnostics import json_value
from .lifecycle import cleanup_all
from .recovery_identity import canonical, job_identity
from .recovery_journal import (
    AttemptJournal,
    export_paths,
    inventory,
    remap_outputs,
    verify_inventory,
)
from .recovery_store import BundleStore, CacheCapacity, CacheInvalid, write_record
from .session import InferenceSession

if TYPE_CHECKING:
    from .durable_writer_prefix import DurableWriterPrefix


class _WriterRunOptions(TypedDict, total=False):
    durable_writer_prefix: DurableWriterPrefix


class _Receipt(TypedDict):
    bytes: int
    sha256: str


class _PublicationOptions(TypedDict, total=False):
    expected_inventory: dict[str, _Receipt]


ROOT = Path(__file__).resolve().parents[1]


def _validate_cache_context(identify, job, expected):
    if identify(job) != expected:
        raise CacheInvalid('job dependencies changed before durable cache access')


def _fresh_destination(requested):
    path = Path(requested).resolve()
    while path.exists():
        path = Path(requested).resolve().with_name(Path(requested).name + '.recovery-' + uuid.uuid4().hex)
    return path


def _separate_workspace(root, jobs, assets, *, check=None):
    """Admit dependencies before constructing a store that can prune/evict."""
    from .corrections import MAX_CORRECTION_BYTES, CorrectionDocument
    from .local_json import read_object

    check = check or (lambda: None)
    root = Path(root).resolve()

    def within(child, parent):
        check()
        if child.is_relative_to(parent):
            return True
        try:
            parent_info = parent.stat()
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(parent_info.st_mode):
            return False
        identity = parent_info.st_dev, parent_info.st_ino
        # Existing-directory identity also closes aliases/case variants that
        # native Path spelling normalization need not collapse.
        for depth, ancestor in enumerate(chain((child,), child.parents)):
            check()
            if depth >= 1024:
                raise ValueError('recovery dependency ancestry inspection limit exceeded')
            try:
                info = ancestor.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == identity:
                return True
        return False

    inputs = [path for job in jobs for path in (job.input_path, job.initial_hint_path, job.hint_video_path)
              if path is not None] + list(assets.values())
    for job in jobs:
        check()
        output = Path(job.output_dir).resolve()
        if within(output, root) or within(root, output):
            raise ValueError('recovery workspace must be separate from render destinations')
        config, _ = job.configurations()
        if config.corrections_path is not None:
            path = Path(config.corrections_path).resolve()
            inputs.append(path)
            data, _ = read_object(path, max_bytes=MAX_CORRECTION_BYTES, regular_file=True)
            check()
            document = CorrectionDocument.parse(data, path.parent)
            for asset_path in document.asset_paths():
                check()
                inputs.append(asset_path)
    for path in inputs:
        check()
        path = Path(path).resolve()
        if within(path, root) or (path.is_dir() and within(root, path)):
            raise ValueError('recovery workspace must be separate from source/hint/model/correction dependencies')


def run_recoverable(jobs, *, workspace, max_cache_bytes, assets, environment, implementation_root=ROOT,
                    weights=None, resume=False, durable_foundations=False, durable_rvm=False, durable_hann=False,
                    durable_stage2=False, durable_writer_prefix=False,
                    session_factory: Callable[..., InferenceSession] = InferenceSession, cancellation=None, progress=None):
    if not jobs:
        raise ValueError('recoverable batch requires jobs')
    if type(durable_foundations) is not bool:
        raise ValueError('durable_foundations must be a boolean')
    if type(durable_rvm) is not bool:
        raise ValueError('durable_rvm must be a boolean')
    if type(durable_hann) is not bool:
        raise ValueError('durable_hann must be a boolean')
    if type(durable_stage2) is not bool:
        raise ValueError('durable_stage2 must be a boolean')
    if type(durable_writer_prefix) is not bool:
        raise ValueError('durable_writer_prefix must be a boolean')
    policy = {name: True for name, enabled in
              (('durable_foundations', durable_foundations), ('durable_rvm', durable_rvm),
               ('durable_hann', durable_hann), ('durable_stage2', durable_stage2),
               ('durable_writer_prefix', durable_writer_prefix)) if enabled}
    identity_environment = {**environment, 'recovery_policy': policy} if policy else environment
    control = JobControl(cancellation, progress)
    root = Path(workspace).resolve()
    from .output_paths import validate_batch_output_paths

    validate_batch_output_paths(
        jobs, weights=weights, protected_paths=assets.values(),
        protected_directories=(assets['cradio_dir'],) if 'cradio_dir' in assets else (), check=control.check)
    _separate_workspace(root, jobs, assets, check=control.check)
    store_owner = (BundleStore(root, max_cache_bytes, structural_admission=True) if durable_writer_prefix
                   else BundleStore(root, max_cache_bytes))
    with store_owner as store:
        if durable_writer_prefix:
            from . import cache_maintenance

            _, snapshot = store._structural_scan(control.check)
            # A new journal, final run report and atomic metadata replacement
            # must fit even when no optional bundles can be cached (quota 0).
            if len(snapshot['files']) + 3 > cache_maintenance.MAX_FILES:
                raise CacheCapacity('managed workspace lacks journal/report structural headroom')
        journal = AttemptJournal(store, [json_value(job) for job in jobs], resume=resume)
        reports = root / 'reports'
        if reports.is_symlink():
            raise ValueError('recovery reports directory cannot be a symlink')
        reports.mkdir(exist_ok=True)
        report_path = reports / f'run-{uuid.uuid4().hex}.json'
        report = {'schema_version': 1, 'kind': 'recoverable_exr_batch', 'status': 'running', 'jobs': [],
                  'journal': str(journal.path), 'report_path': str(report_path), 'environment': environment,
                  'note': 'Restored/retained bytes are not new neural inference. Origin timings are historical. '
                          'Optional IID/RVM/Hann/Stage2 entries avoid verified neural work; '
                          'source/state reconstruction and rendering still run. Verified EXR prefixes may be copied '
                          'only on explicit resume; native writes cover the remaining suffix in fresh output.',
                  'durable_foundations_enabled': durable_foundations, 'durable_rvm_enabled': durable_rvm,
                  'durable_hann_enabled': durable_hann, 'durable_stage2_enabled': durable_stage2,
                  'durable_writer_prefix_enabled': durable_writer_prefix}

        def identify(job):
            return job_identity(job, assets=assets, environment=identity_environment,
                                implementation_root=implementation_root, check=control.check)

        resource_guard = None
        resource_record = None
        try:
            with session_factory(weights=weights, package_root=implementation_root) as session:
                batch_model_context = None
                for index, requested in enumerate(jobs):
                    control.check_hook = None
                    resource_guard = None
                    resource_record = None
                    requested_config, _ = requested.configurations()
                    if requested_config.process_tree_rss_stop_bytes is not None:
                        from .memory_guard import ProcessTreeRssGuard

                        resource_guard = ProcessTreeRssGuard(
                            requested_config.process_tree_rss_stop_bytes, cancel_check=control.cancellation.check)
                        resource_record = {'index': index, 'policy': resource_guard.snapshot()}
                        report.setdefault('resource_observations', []).append(resource_record)
                        control.check_hook = resource_guard.check
                        resource_guard.check(force=True)
                    started = time.perf_counter()
                    control.checkpoint('recovery_identity', index, len(jobs))
                    destination = _fresh_destination(requested.output_dir)
                    job = replace(requested, output_dir=str(destination))
                    identity = identify(job)
                    identity_data = identity.to_dict()
                    model_context = canonical({name: identity_data[name] for name in ('assets', 'environment', 'implementation')})
                    if batch_model_context is None:
                        batch_model_context = model_context
                    elif model_context != batch_model_context:
                        raise CacheInvalid('model assets/backend/implementation changed during batch; resume with a fresh session')
                    completed = journal.completed(index, identity, check=control.check)
                    if completed is not None:
                        if resource_guard is not None:
                            resource_guard.check(force=True)
                            assert resource_record is not None
                            resource_record['policy'] = resource_guard.snapshot()
                        destination = Path(completed['output_dir'])
                        origin = completed['origin_record']
                        report['jobs'].append({'index': index, 'mode': 'verified_existing', 'new_session_run': False,
                                               'key': identity.key, 'output_dir': str(destination),
                                               'requested_output_dir': requested.output_dir,
                                               'outputs': remap_outputs(origin, destination), 'origin_record': origin,
                                               'operation_seconds': time.perf_counter() - started})
                        continue
                    cache_note = None
                    try:
                        cached = store.lookup(identity, check=control.check)
                    except CacheInvalid as exc:
                        cache_note = str(exc)
                        store.discard(identity)  # Only recognized owned derived files can be removed.
                        cached = None
                    mode = 'verified_cache_restore' if cached is not None else 'render'
                    writer_prefix_store: DurableWriterPrefix | None = None
                    attempt = journal.start(index, identity, destination, mode)
                    try:
                        if cached is not None:
                            control.checkpoint('recovery_restore', index, len(jobs))
                            manifest = store.restore(identity, destination, check=control.check)
                            if manifest is None:
                                raise CacheInvalid('sealed cache disappeared under exclusive ownership')
                            origin = manifest['origin_record']
                            artifacts = [{name: item[name] for name in ('name', 'bytes', 'sha256')}
                                         for item in manifest['files']]
                        else:
                            destination.mkdir(parents=True, exist_ok=False)
                            frame_store = None
                            if durable_foundations:
                                from .durable_foundations import DurableFeatureFrames

                                frame_store = DurableFeatureFrames(
                                    store, identity,
                                    validate_context=partial(_validate_cache_context, identify, job, identity),
                                    check=control.check)
                            rvm_store = None
                            if durable_rvm:
                                from .durable_rvm import DurableRvmChunks

                                rvm_store = DurableRvmChunks(
                                    store, identity,
                                    validate_context=partial(_validate_cache_context, identify, job, identity),
                                    check=control.check)
                            hann_store = None
                            if durable_hann:
                                from .durable_hann import DurableHannWindows

                                hann_store = DurableHannWindows(
                                    store, identity,
                                    validate_context=partial(_validate_cache_context, identify, job, identity),
                                    check=control.check)
                            stage2_store = None
                            if durable_stage2:
                                from .durable_stage2 import DurableStage2Batches

                                stage2_store = DurableStage2Batches(
                                    store, identity,
                                    validate_context=partial(_validate_cache_context, identify, job, identity),
                                    check=control.check)
                            writer_options: _WriterRunOptions = {}
                            if durable_writer_prefix:
                                from .durable_writer_prefix import DurableWriterPrefix

                                writer_prefix_store = DurableWriterPrefix(
                                    store, identity,
                                    validate_context=partial(_validate_cache_context, identify, job, identity),
                                    check=control.check, resume=resume)
                                writer_options['durable_writer_prefix'] = writer_prefix_store
                            if resource_guard is None:
                                result = session.run(job, cancellation=control.cancellation, progress=control.progress,
                                                     durable_foundations=frame_store, durable_rvm=rvm_store,
                                                     durable_hann=hann_store, durable_stage2=stage2_store, **writer_options)
                            else:
                                result = session.run(job, cancellation=control.cancellation, progress=control.progress,
                                                     durable_foundations=frame_store, durable_rvm=rvm_store,
                                                     durable_hann=hann_store, durable_stage2=stage2_store,
                                                     resource_guard=resource_guard, **writer_options)
                            origin = result.to_dict()
                            if Path(origin['output_dir']).resolve() != destination or origin.get('input') != job.input_path:
                                raise ValueError('completed result does not match the owned recovery request')
                            loaded_paths = origin.get('asset_paths')
                            if loaded_paths is not None or environment['backend'] == 'pytorch-cuda':
                                expected_paths = {name: item['path'] for name, item in identity_data['assets'].items()}
                                if (not isinstance(loaded_paths, dict) or set(loaded_paths) != set(expected_paths) or
                                        any(str(Path(loaded_paths[name]).resolve()) != path for name, path in expected_paths.items())):
                                    raise CacheInvalid('actual session-loaded asset paths differ from fingerprinted assets')
                            files = export_paths(origin)
                            config, _ = job.configurations()
                            if set(origin['diagnostics']['outputs']) != set(config.outputs):
                                raise ValueError('completed outputs differ from requested outputs')
                            if identify(job) != identity:
                                raise CacheInvalid('job dependencies changed during render; refusing cache publication')
                            control.checkpoint('recovery_publish', index, len(jobs))
                            publication_options: _PublicationOptions = {}
                            expected_files: dict[str, _Receipt] | None = None
                            if writer_prefix_store is not None:
                                expected_files = {item['name']: {'bytes': item['bytes'], 'sha256': item['sha256']}
                                                  for item in inventory(files, check=control.check)}
                                restored_receipts = writer_prefix_store.verify_restored_prefix()
                                if not set(restored_receipts) <= set(files):
                                    raise CacheInvalid('restored writer prefix differs from completed output inventory')
                                for name, receipt in restored_receipts.items():
                                    expected_files[name] = {'bytes': receipt['bytes'], 'sha256': receipt['sha256']}
                                publication_options['expected_inventory'] = expected_files
                            try:
                                manifest = store.publish(identity, files, origin, check=control.check, **publication_options)
                                artifacts = [{name: item[name] for name in ('name', 'bytes', 'sha256')}
                                             for item in manifest['files']]
                            except CacheCapacity as exc:
                                cache_note = str(exc)
                                artifacts = ([{'name': name, 'bytes': receipt['bytes'], 'sha256': receipt['sha256']}
                                              for name, receipt in sorted(expected_files.items())]
                                             if expected_files is not None else inventory(files, check=control.check))
                        if identify(job) != identity:
                            raise CacheInvalid('job dependencies changed during recovery')
                        if not verify_inventory(destination, artifacts, check=control.check):
                            raise CacheInvalid('completed output inventory changed before journal seal')
                        control.check()
                        if resource_guard is not None:
                            resource_guard.check(force=True)
                        if writer_prefix_store is not None:
                            writer_prefix_store.verify_restored_prefix()
                        journal.finish(index, attempt, status='complete', artifacts=artifacts, origin_record=origin, check=control.check)
                        report['jobs'].append({'index': index, 'mode': mode, 'new_session_run': mode == 'render',
                                               'key': identity.key, 'requested_output_dir': requested.output_dir,
                                               'output_dir': str(destination), 'outputs': remap_outputs(origin, destination),
                                               'origin_record': origin, 'cache_note': cache_note,
                                               'operation_seconds': time.perf_counter() - started})
                        if resource_guard is not None:
                            assert resource_record is not None
                            resource_record['policy'] = resource_guard.snapshot()
                    except BaseException as exc:
                        if session.last_result is not None:
                            failed = session.last_result.to_dict()
                            if failed.get('output_dir') == str(destination):
                                report['failed_attempt'] = failed
                        status = 'cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed'
                        cleanup_all([partial(journal.finish, index, attempt, status=status,
                                             error={'type': type(exc).__name__, 'message': str(exc)})],
                                    primary_error=exc)
                        raise
            control.check_hook = None
            report['status'] = 'complete'
            control.checkpoint('recovery_complete', len(jobs), len(jobs))
        except BaseException as exc:
            report['status'] = 'cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed'
            report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            if resource_guard is not None and resource_record is not None:
                resource_record['policy'] = resource_guard.snapshot()
            cleanup_all([lambda: write_record(report_path, report)], primary_error=sys.exception())
        return report


def cuda_environment():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('new recoverable inference requires the validated CUDA runtime; no CPU/MPS fallback')
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    if not driver:
        raise RuntimeError('CUDA driver identity is unavailable; refusing unverified replay context')
    dependencies = {}
    for distribution in importlib.metadata.distributions():
        dependencies.setdefault(distribution.metadata['Name'], set()).add(distribution.version)
    return {'backend': 'pytorch-cuda', 'precision': 'model bfloat16, baton/rvm float16, IID float32',
            'dependencies': {name: sorted(versions) for name, versions in sorted(dependencies.items())},
            'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                        'gpu': properties.name, 'gpu_uuid': str(getattr(properties, 'uuid', None)),
                        'capability': [properties.major, properties.minor], 'vram_bytes': properties.total_memory,
                        'cuda': torch.version.cuda, 'driver': driver, 'cudnn': torch.backends.cudnn.version(),
                        'deterministic': torch.are_deterministic_algorithms_enabled(),
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                        'default_dtype': str(torch.get_default_dtype()),
                        'matmul_precision': torch.get_float32_matmul_precision()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    run = subparsers.add_parser('run')
    run.add_argument('--batch', required=True, type=Path)
    run.add_argument('--workspace', required=True, type=Path)
    run.add_argument('--max-cache-bytes', required=True, type=int)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--durable-foundations', action='store_true',
                     help='reuse sealed float32 IID C-RADIO/MoGe frames')
    run.add_argument('--durable-rvm', action='store_true',
                     help='reuse exact chronological RVM chunks and state')
    run.add_argument('--durable-hann', action='store_true',
                     help='replay exact Stage1 window events to reconstruct Hann state; Stage2/writers restart')
    run.add_argument('--durable-stage2', action='store_true',
                     help='reuse verified exact Stage2 prediction batches; upstream state, rendering and writers restart')
    run.add_argument('--durable-writer-prefix', action='store_true',
                     help='seal checked EXR prefixes; copy only on explicit resume, with full chronology/render replay')
    status = subparsers.add_parser('status')
    status.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == 'status':
        root = args.workspace.resolve()
        marker = json.loads((root / 'workspace.json').read_text())
        if marker.get('kind') != 'corridorkey_recovery':
            parser.error('not a CorridorKey recovery workspace')
        print(json.dumps({'workspace': str(root), 'journals': [str(p) for p in sorted((root / 'journals').glob('*.json'))],
                          'scope': 'metadata listing only; payload/backend/outputs NOT VERIFIED'}, indent=2))
        return 0
    if args.max_cache_bytes < 0:
        parser.error('max-cache-bytes must be nonnegative')
    jobs, weights = load_jobs(args.batch)
    if any(config.output_format != 'exr' or config.auto_vram for job in jobs for config, _ in (job.configurations(),)):
        parser.error('recovery requires EXR jobs with explicit scheduling (auto_vram=false)')
    from .runtime import RuntimePaths

    paths = RuntimePaths.from_package(ROOT, weights)
    paths.validate()
    assets = {name: getattr(paths, name) for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir')}
    report = run_recoverable(jobs, workspace=args.workspace, max_cache_bytes=args.max_cache_bytes,
                             assets=assets, environment=cuda_environment(), weights=weights, resume=args.resume,
                             durable_foundations=args.durable_foundations, durable_rvm=args.durable_rvm,
                             durable_hann=args.durable_hann, durable_stage2=args.durable_stage2,
                             durable_writer_prefix=args.durable_writer_prefix)
    print(f"recoverable EXR batch: {report['status']}; report: {report['report_path']}")
    for job in report['jobs']:
        print(f"{job['mode']}: {job['output_dir']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
