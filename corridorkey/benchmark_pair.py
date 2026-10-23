"""Alternate matched local inference measurements; no media leaves this machine."""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import benchmark
from .local_json import read_object

LIMIT = 4 * 1024 * 1024
SIDES = ('baseline', 'candidate')
# Changes to these fields would change the timed workload or reference alignment.
FIXED_SETTINGS = {'outputs', 'output_format', 'start_frame', 'num_frames', 'frame_dir_fps',
                  'corrections_path', 'temp_dir', 'profile_stages', 'start_mode', 'seed', 'exr_dtype'}


def _object(path):
    return read_object(Path(path), max_bytes=LIMIT)[0]


def _load(path):
    path = Path(path).resolve()
    value = _object(path)
    if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('kind') != 'inference_pair'
            or set(value) - {'schema_version', 'kind', 'baseline', 'candidate', 'repetitions', 'changed_settings'}):
        raise ValueError('Expected schema_version=1, kind=inference_pair and known fields.')
    repetitions = value.get('repetitions', 3)
    if type(repetitions) is not int or repetitions < 3:
        raise ValueError('At least three actual paired repetitions are required.')
    fields = value.get('changed_settings')
    if (not isinstance(fields, list) or not fields or any(not isinstance(f, str) or not f for f in fields)
            or len(set(fields)) != len(fields) or set(fields) & FIXED_SETTINGS):
        raise ValueError('changed_settings must declare distinct non-workload settings to compare.')
    manifests = {}
    for side in SIDES:
        name = value.get(side)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'{side} requires an inference manifest path.')
        source = (path.parent / name).resolve()
        manifest = benchmark.load_manifest(source)
        if manifest['kind'] != 'inference':
            raise ValueError('Paired runs require inference manifests.')
        manifests[side] = (source, manifest)
    return value, manifests


def _freeze(manifests, root):
    from .batch import load_jobs
    frozen, provenance, batches, watched = {}, {}, {}, {}
    for side, (path, manifest) in manifests.items():
        batch_path = (path.parent / manifest['batch_manifest']).resolve()
        jobs, weights = load_jobs(batch_path)
        normalized = {'schema_version': 1, 'kind': 'inference_batch',
                      'weights': str(weights.resolve()) if weights is not None else None, 'jobs': []}
        for index, job in enumerate(jobs):
            if job.profile_stages:
                raise ValueError('Profiling must be disabled for paired throughput measurements.')
            normalized['jobs'].append({'input': job.input_path, 'output_dir': str(root / side / str(index)),
                                      'settings': json.loads(job.config_json), 'writer': json.loads(job.writer_json),
                                      'hint_first_frame': job.initial_hint_path, 'hint_video': job.hint_video_path})
            for source in (job.input_path, job.initial_hint_path, job.hint_video_path):
                if source is not None:
                    watched.setdefault(source, benchmark.asset_identity(source))
        batch_copy = root / f'{side}-batch.json'
        batch_copy.write_text(json.dumps(normalized, allow_nan=False))
        copied: dict[str, Any] = dict(manifest, batch_manifest=str(batch_copy), repetitions=1)
        if 'quality_manifests' in copied:
            copied['quality_manifests'] = [str((path.parent / p).resolve()) if p is not None else None
                                           for p in copied['quality_manifests']]
            from .benchmark_quality import prepare_quality_templates
            templates = prepare_quality_templates(copied['quality_manifests'], jobs, manifest_dir=path.parent)
            for template in templates or []:
                if template is not None:
                    for asset in template['asset_hashes']:
                        watched.setdefault(asset, benchmark.asset_identity(asset))
            for quality in copied['quality_manifests']:
                if quality is not None:
                    watched.setdefault(quality, benchmark.asset_identity(quality))
        manifest_copy = root / f'{side}.json'
        manifest_copy.write_text(json.dumps(copied, allow_nan=False))
        frozen[side] = (manifest_copy, copied)
        batches[side] = normalized
        provenance[side] = {'manifest_path': str(path), 'manifest_sha256': benchmark.sha256(path),
                            'batch_path': str(batch_path), 'batch_sha256': benchmark.sha256(batch_path),
                            'normalized_batch': normalized, 'frozen_manifest': copied}
    return frozen, provenance, batches, watched


def _match(manifests, batches, fields):
    left, right = (batches[side] for side in SIDES)
    if left['weights'] != right['weights'] or len(left['jobs']) != len(right['jobs']):
        raise ValueError('Paired weights and job counts must match.')
    a, b = (manifests[side][1] for side in SIDES)
    if a['model_state'] != b['model_state'] or a.get('quality_manifests') != b.get('quality_manifests'):
        raise ValueError('Paired model state and quality templates must match.')
    changed = set()
    for a, b in zip(left['jobs'], right['jobs']):
        if a['settings'].get('auto_vram') or b['settings'].get('auto_vram'):
            raise ValueError('Controlled pairs require auto_vram=false; automatic selection is a separate experiment.')
        if {k: v for k, v in a.items() if k not in {'output_dir', 'settings'}} != {
                k: v for k, v in b.items() if k not in {'output_dir', 'settings'}}:
            raise ValueError('Paired inputs, hints and output writer must match.')
        differences = {k for k in a['settings'].keys() | b['settings'].keys()
                       if a['settings'].get(k) != b['settings'].get(k)}
        if differences - set(fields):
            raise ValueError(f'Undeclared settings differ: {sorted(differences - set(fields))}')
        changed.update(differences)
    if changed != set(fields):
        raise ValueError('Every declared changed setting must actually differ.')


def _sample(report, count):
    if report.get('inference', {}).get('status') == 'cancelled':
        raise KeyboardInterrupt('Inference measurement cancelled.')
    if not report.get('summary', {}).get('success') or report.get('inference', {}).get('status') != 'measured':
        raise ValueError('Inference measurement or requested quality evaluation failed.')
    runs = report.get('runs', [])
    if len(runs) != 1 or len(runs[0].get('jobs', [])) != count:
        raise ValueError('Measurement has missing or extra runs/jobs.')
    if any(job.get('status') != 'complete' for job in runs[0]['jobs']):
        raise ValueError('Measurement contains incomplete jobs.')
    elapsed = runs[0].get('elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Measurement duration must be finite and positive.')
    return elapsed



def _effective(report, jobs):
    effective = []
    for measured, frozen in zip(report['runs'][0]['jobs'], jobs, strict=True):
        settings = measured.get('diagnostics', {}).get('effective_settings')
        if not isinstance(settings, dict) or set(settings) != set(frozen['settings']):
            raise ValueError('Complete effective runtime settings are required for every measured job.')
        if settings.get('auto_vram') is not False:
            raise ValueError('Controlled pairs require manual effective settings.')
        if measured.get('requested_settings') != frozen['settings']:
            raise ValueError('Measured requested settings differ from the frozen configuration.')
        effective.append(settings)
    return effective


def _match_effective(current, previous, allowed):
    for a, b in zip(current, previous, strict=True):
        differences = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if differences - set(allowed):
            raise ValueError(f'Unmatched effective runtime settings: {sorted(differences - set(allowed))}')


def _identity(report):
    # Storage-free bytes change with exports; compare hardware/dependencies instead.
    env = report.get('environment', {})
    hardware = env.get('hardware', {})
    return {key: report.get(key) for key in ('repository', 'assets', 'weights', 'corrections')} | {
        'environment': {key: env.get(key) for key in ('python', 'os', 'dependencies')},
        'hardware': {key: hardware.get(key) for key in ('cpu', 'ram_bytes', 'gpu', 'cuda_version', 'driver', 'device_index')}}


def run(manifest_path):
    """Return terminal evidence even on failure; partial runs never yield a speedup."""
    report = {'schema_version': 1, 'kind': 'inference_pair', 'status': 'running',
              'created_utc': datetime.now(UTC).isoformat(), 'pairs': [],
              'summary': {'success': False, 'speedup': None},
              'conditions': {'order': 'AB, BA, AB, ...', 'selection': 'manual; auto_vram is refused', 'filesystem': 'uncontrolled; hashing reads inputs',
                             'output': 'fresh destination for each actual sample',
                             'warm_session': 'independent session plus full unmeasured warmup per side/sample',
                             'timing': 'shared runner measured elapsed only; preparation/quality/warmup excluded',
                             'quality': 'raw per-side quality retained; missing references remain unavailable'},
              'warnings': ['Configuration comparison uses the same engine/interpreter; cross-revision comparison is unsupported.',
                           'Timing alone does not establish quality, memory improvement or production acceptance.']}
    try:
        value, manifests = _load(manifest_path)
        report['manifest_sha256'] = benchmark.sha256(manifest_path)
        with tempfile.TemporaryDirectory(prefix='corridorkey-pair-') as folder:
            frozen, provenance, batches, watched = _freeze(manifests, Path(folder))
            report['provenance'] = provenance
            _match(frozen, batches, value['changed_settings'])
            identity = None
            effective_by_side = {}
            for index in range(value.get('repetitions', 3)):
                pair = {'index': index, 'order': list(SIDES if index % 2 == 0 else reversed(SIDES)), 'samples': {}}
                report['pairs'].append(pair)
                for side in pair['order']:
                    for path, expected in watched.items():
                        if benchmark.asset_identity(path) != expected:
                            raise ValueError('Input or quality template changed during paired experiment.')
                    path, manifest = frozen[side]
                    raw = benchmark.run_inference_benchmark(path, dict(manifest))
                    try:
                        json.dumps(raw, allow_nan=False)
                    except (ValueError, TypeError):
                        pair['samples'][side] = {'rejected_report_repr': repr(raw)}
                        raise ValueError('Measurement report is not finite JSON data.') from None
                    pair['samples'][side] = raw
                    _sample(raw, len(batches[side]['jobs']))
                    effective = _effective(raw, batches[side]['jobs'])
                    if side in effective_by_side:
                        _match_effective(effective, effective_by_side[side], [])
                    other = 'candidate' if side == 'baseline' else 'baseline'
                    if other in effective_by_side:
                        _match_effective(effective, effective_by_side[other], value['changed_settings'])
                    effective_by_side.setdefault(side, effective)
                    current = _identity(raw)
                    if identity is not None and identity != current:
                        raise ValueError('Repository, hardware, assets, weights or corrections changed between samples.')
                    identity = current
                    for weight, expected in raw.get('weights', {}).items():
                        if benchmark.sha256(weight) != expected:
                            raise ValueError('Weight changed during paired experiment.')
                    for correction, job in zip(raw.get('corrections', []), batches[side]['jobs']):
                        if correction is not None and benchmark.correction_identity(
                                correction['path'], job['input']) != correction:
                            raise ValueError('Correction changed during paired experiment.')
                    for path, expected in watched.items():
                        if benchmark.asset_identity(path) != expected:
                            raise ValueError('Input or quality template changed during paired experiment.')
            durations = {side: [_sample(p['samples'][side], len(batches[side]['jobs'])) for p in report['pairs']]
                         for side in SIDES}
            stats = {side: {'raw_seconds': times, 'median_seconds': statistics.median(times),
                            'min_seconds': min(times), 'max_seconds': max(times),
                            'stdev_seconds': statistics.stdev(times)} for side, times in durations.items()}
            report['quality_pairs'] = [
                {side: p['samples'][side]['inference'].get('quality') for side in SIDES}
                for p in report['pairs']]
            summary = {'success': True, **stats,
                                 'speedup': stats['baseline']['median_seconds'] / stats['candidate']['median_seconds'],
                                 'paired_ratios': [a / b for a, b in zip(durations['baseline'], durations['candidate'])]}
            json.dumps(summary, allow_nan=False)
            report['summary'] = summary
            report['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        logging.getLogger(__name__).exception('Paired benchmark terminated')
        report['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='New local JSON report; existing destinations are refused.')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation precedes expensive work; source/report files cannot be overwritten.
    with output.open('x', encoding='utf-8') as stream:
        report = run(args.manifest)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Paired benchmark {report['status']}; report: {output}")
    return 130 if report['status'] == 'cancelled' else 0 if report['summary']['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
