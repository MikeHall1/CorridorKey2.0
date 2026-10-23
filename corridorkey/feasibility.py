"""Bounded projection of recorded inference observations, never a realtime claim."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TypeGuard

from .benchmark_report import MAX_VIEW_CHARS, read_report

MAX_SAMPLES = 256
MAX_JOBS = 4096
MAX_STAGES = 4096
NOT_VERIFIED = 'NOT VERIFIED'
LANES = {
    'decode': {'decode'}, 'transfer': {'transfer'},
    'foundations': {'iid_foundations', 'foundation_prepass'},
    'temporal': {'rvm_prepass', 'stage1_stream', 'stage1_stream_including_stage2_render_submit'},
    'neural_refinement': {'stage2_inference'}, 'postprocess': {'render'},
    'encoding': {'encoding'}, 'write_submission': {'writer_submit'},
    'writer_finalize_cleanup': {'writer_finalize_and_cleanup'},
    'delivery': set(), 'queue': set(), 'synchronization': set(),
}


def _number(value, *, positive=False) -> TypeGuard[int | float]:
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0))
    except OverflowError:
        return False


def _rate(count, seconds):
    if type(count) is not int or count <= 0 or not _number(seconds, positive=True):
        return None
    try:
        rate = count / seconds
        return rate if _number(rate, positive=True) and _number(1 / rate, positive=True) else None
    except (OverflowError, ZeroDivisionError):
        return None


def _observation(value, scope, kind=None):
    if value is None or value == {} or value == []:
        return {'status': NOT_VERIFIED, 'scope': scope}
    if not isinstance(value, dict):
        return {'status': NOT_VERIFIED, 'scope': scope, 'reason': 'Observation is not a structured record'}
    valid = True
    if kind == 'resources':
        peaks = [value[k] for k in ('peak_allocated_vram_bytes', 'peak_reserved_vram_bytes') if k in value]
        valid = bool(peaks) and all(type(v) is int and v >= 0 for v in peaks) and isinstance(value.get('scope'), str)
    elif kind == 'resource_policy':
        valid = (type(value.get('samples')) is int and value['samples'] >= 0 and isinstance(value.get('scope'), str)
                 and (value.get('observed_peak_rss_bytes') is None or
                      type(value['observed_peak_rss_bytes']) is int and value['observed_peak_rss_bytes'] >= 0))
    elif kind == 'output_telemetry':
        valid = (value.get('status') in ('complete', 'partial', 'unavailable')
                 and type(value.get('native_completion')) is bool
                 and all(isinstance(value[k], dict) for k in ('queue_accept_latency', 'native_write_latency',
                         'render_ready_offsets', 'native_completion_offsets', 'completion_spacing') if k in value))
    if not valid:
        return {'status': NOT_VERIFIED, 'scope': scope, 'reason': 'Observation does not match the recorded diagnostic shape'}
    if len(json.dumps(value, allow_nan=False)) > 65536:
        return {'status': NOT_VERIFIED, 'scope': scope, 'reason': 'Observation exceeds projection size limit'}
    return {'status': 'recorded', 'scope': scope, 'value': value}


def _job(job, index, frame_budget):
    if not isinstance(job, dict):
        raise TypeError('Inference jobs must be records')
    diagnostics = job.get('diagnostics') or {}
    if not isinstance(diagnostics, dict):
        raise TypeError('Job diagnostics must be a record')
    stages = diagnostics.get('stages') or []
    if not isinstance(stages, list) or len(stages) > MAX_STAGES:
        raise ValueError('Stage observation count exceeds projection limit')
    valid = []
    for stage in stages:
        if (not isinstance(stage, dict) or not isinstance(stage.get('name'), str)
                or len(stage['name']) > 256 or not _number(stage.get('seconds'))
                or stage.get('method') not in ('host_wall', 'synchronized_wall')
                or stage.get('status') not in ('complete', 'failed')):
            raise ValueError('Invalid recorded stage observation')
        valid.append({k: stage[k] for k in ('name', 'seconds', 'method', 'status')})
    first = diagnostics.get('first_rendered_batch_seconds')
    elapsed = job.get('processing_seconds')
    frames = diagnostics.get('frames_submitted')
    complete = job.get('status') == 'complete'
    fps = _rate(frames, elapsed) if complete else None
    return {'job': index, 'status': job.get('status'),
            'processing_fps': fps, 'processing_scope': 'Submitted frames / session processing wall time; excludes model load, not presentation.',
            'first_rendered_batch_seconds': first if _number(first) else None,
            'first_rendered_batch_scope': 'Since job diagnostics start; CPU-rendered batch, not encoded, durable or presented.',
            'model_load_seconds': job.get('model_load_seconds') if _number(job.get('model_load_seconds')) else None,
            'model_initialization_seconds': job.get('model_initialization_seconds') if _number(job.get('model_initialization_seconds')) else None,
            'profiling_enabled': diagnostics.get('profiling_enabled'),
            'stages': valid,
            'stage_lanes': {lane: [s for s in valid if s['name'] in names] or NOT_VERIFIED for lane, names in LANES.items()},
            'target_budget_seconds': frame_budget,
            'output_telemetry': _observation(diagnostics.get('output_telemetry'),
                'Recorded render-ready/queue/native-write milestones only; native write is not fsync or playback. Partial populations retain their recorded status.', 'output_telemetry'),
            'resources': _observation(diagnostics.get('resources'),
                'Recorded runtime resource snapshot/allocation peaks, not continuous process-tree or hard-cap proof.', 'resources'),
            'resource_policy': _observation(diagnostics.get('resource_policy'), 'Recorded observed policy samples, not an unsampled hard memory cap.', 'resource_policy'),
            'settings': _observation({'requested': job.get('requested_settings'), 'effective': diagnostics.get('effective_settings')}, 'Recorded settings; missing members are unknown.'),
            'cache': _observation(diagnostics.get('foundation_cache'), 'Per-job foundation cache observations only.')}


def project(report, target_fps=24.0):
    """Project an admitted report without measuring hardware or summing stage scopes."""
    if not _number(target_fps, positive=True) or not 0.01 <= target_fps <= 1000:
        raise ValueError('Target FPS must be finite, in0.01..1000')
    result = {'schema_version': 1, 'kind': 'realtime_feasibility', 'target_fps': target_fps,
              'frame_budget_seconds': 1 / target_fps, 'status': NOT_VERIFIED, 'samples': [],
              'reported_status': report.get('status'), 'reported_success': report.get('summary', {}).get('success'),
              'reported_inference_status': report.get('inference', {}).get('status'),
              'sustained_presentation': NOT_VERIFIED, 'instrumentation_overhead': NOT_VERIFIED,
              'warnings': ['Offline throughput is not sustained delivery or presentation.',
                           'Inclusive/nested/overlapped stages and queue timings are never summed.',
                           'Synchronized profiling changes execution; overhead needs a matched measured experiment.',
                           'CPU correctness duration is never inference throughput. Missing observations are NOT VERIFIED.']}
    if report.get('kind') not in ('inference', 'inference_pair'):
        result['reason'] = 'Selected report does not contain inference measurements'
        return result
    if report['kind'] == 'inference_pair':
        pairs = report.get('pairs')
        if not isinstance(pairs, list) or len(pairs) > MAX_SAMPLES // 2:
            raise ValueError('Pair observation count exceeds projection limit')
        sources = [(f"pair{p['index']}/{s}", p['samples'][s]) for p in pairs for s in ('baseline', 'candidate') if s in p['samples']]
        result['pair_conditions'] = _observation(report.get('conditions'), 'Raw paired measurement conditions; no cross-side speedup inferred.')
    else:
        sources = [('inference', report)]
    jobs_seen = 0
    samples = result['samples']
    for label, source in sources:
        runs = source.get('runs')
        if runs is None or runs == []:
            if len(samples) >= MAX_SAMPLES:
                raise ValueError('Sample observation count exceeds projection limit')
            samples.append({'sample': label, 'status': NOT_VERIFIED, 'offline_fps': None, 'jobs': [],
                            'reason': 'No measurement runs; failed/rejected sample diagnostics remain in the raw report.'})
            continue
        if not isinstance(runs, list) or len(samples) + len(runs) > MAX_SAMPLES:
            raise ValueError('Run observation count exceeds projection limit')
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict) or not isinstance(run.get('jobs'), list):
                raise TypeError('Inference runs require job records')
            jobs_seen += len(run['jobs'])
            if jobs_seen > MAX_JOBS:
                raise ValueError('Job observation count exceeds projection limit')
            jobs = [_job(j, i, 1 / target_fps) for i, j in enumerate(run['jobs'])]
            elapsed = run.get('elapsed_seconds')
            counts = [(j.get('diagnostics') or {}).get('frames_submitted') for j in run['jobs']]
            valid_counts = [n for n in counts if type(n) is int and n > 0]
            measured = (source.get('summary', {}).get('success') is True
                        and source.get('inference', {}).get('status') == 'measured'
                        and bool(jobs) and all(j['status'] == 'complete' for j in jobs)
                        and len(valid_counts) == len(counts) and _number(elapsed, positive=True))
            fps = _rate(sum(valid_counts), elapsed) if measured else None
            measured = measured and fps is not None
            state = source.get('model_state')
            scope = ('Fresh interpreter wall time including imports, model loading and shutdown.' if state == 'cold_process'
                     else 'Warm session after full unmeasured warmup; measured run wall time.' if state == 'warm_session'
                     else 'Model lifecycle unavailable; measured run wall time only.')
            samples.append({'sample': label, 'run': run.get('index', run_index), 'model_state': state or NOT_VERIFIED,
                            'status': 'measured' if measured else NOT_VERIFIED,
                            'elapsed_seconds': elapsed if _number(elapsed, positive=True) else None,
                            'offline_fps': fps, 'offline_seconds_per_frame': 1 / fps if fps else None,
                            'offline_target_comparison': ('at_or_above_target' if fps >= target_fps else 'below_target') if fps else NOT_VERIFIED,
                            'timing_scope': scope, 'jobs': jobs,
                            'cache_conditions': _observation(source.get('cache'), 'Reported filesystem/intermediate/output cache conditions.'),
                            'environment': _observation(source.get('environment'), 'Recorded environment; no live hardware query.'),
                            'repository': _observation(source.get('repository'), 'Recorded revision/source identity.')})
    if any(s['status'] == 'measured' for s in samples):
        result['status'] = 'observations_available'
    return result


def inspect(path: Path, target_fps=24.0):
    report, raw = read_report(path, regular_file=True)
    result = project(report, target_fps)
    result['source'] = {'path': str(path.absolute()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return result


def view(projection):
    text = json.dumps(projection, indent=2, allow_nan=False)
    if len(text) > MAX_VIEW_CHARS:
        return text[:MAX_VIEW_CHARS] + '\n[Feasibility view truncated; inspect the original local report.]'
    return text
