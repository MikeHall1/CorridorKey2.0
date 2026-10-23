"""Screen existing paired measurements against an explicitly pinned local policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .benchmark_report import read_report
from .control import CancelledError
from .evaluation_dossier import (
    METRICS,
    _compatibility,
    _quality,
    _repetition_contract,
    _sample_identity,
)
from .local_json import read_object

MAX_PAIRS = 128
SIDES = ('baseline', 'candidate')


def digest(value):
    """Digest a JSON context, not a claim about when it was declared."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _check(check):
    if check is not None:
        check()


def _job_contract(job, expected):
    if (job.get('input') != expected['input'] or job.get('writer_settings') != expected['writer']
            or job.get('hints') != {'first_frame': expected.get('hint_first_frame'), 'full': expected.get('hint_video')}
            or job.get('requested_settings') != expected['settings']):
        raise ValueError('Measured source/writer/hints/settings differ from frozen job')
    actual = job.get('diagnostics', {}).get('effective_settings')
    if (not isinstance(actual, dict) or set(actual) != set(expected['settings'])
            or actual.get('auto_vram') is not False or actual.get('profile_stages')):
        raise ValueError('Complete unprofiled manual effective settings required')
    if type(job.get('model_reused')) is not bool:
        raise ValueError('Measured model reuse metadata is missing')
    return actual


def _quality_contract(sample, index, job):
    quality = sample.get('inference', {}).get('quality')
    if quality is None:
        return None
    entries = quality.get('jobs')
    if not isinstance(entries, list):
        raise TypeError('Quality jobs missing')
    matches = [q for q in entries if q.get('run') == 0 and q.get('job') == index]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate quality job mapping')
    if matches[0].get('report') is None:
        return {'status': matches[0].get('status'), 'reason': matches[0].get('reason')}
    report = _quality(sample, index)
    binding = report.get('candidate_binding', {})
    source = job['diagnostics'].get('source', {})
    if (not job.get('job_id') or not job.get('output_dir')
            or binding.get('job_id') != job['job_id'] or binding.get('output_dir') != job['output_dir']
            or report.get('source', {}).get('path') != job['input']
            or binding.get('source_start_frame') != source.get('start_frame')
            or binding.get('frames_submitted') != job['diagnostics'].get('frames_submitted')
            or report.get('fps') != source.get('native_fps')):
        raise ValueError('Quality source/range/output/job binding differs from measured job')
    return _repetition_contract(report)


def context(report):
    """Bind result-free quality, actual lifecycle, jobs and identities to policy."""
    from .benchmark_pair import FIXED_SETTINGS, _match, _match_effective

    provenance = report.get('provenance')
    if not isinstance(provenance, dict) or set(provenance) != set(SIDES):
        raise ValueError('Missing frozen pair provenance')
    batches, manifests = {}, {}
    for side in SIDES:
        entry = provenance[side]
        batches[side] = entry['normalized_batch']
        manifests[side] = (None, entry['frozen_manifest'])
        if entry['frozen_manifest'].get('model_state') not in ('cold_process', 'warm_session'):
            raise ValueError('Unsupported model/cache lifecycle')
        if not batches[side]['jobs'] or len(batches[side]['jobs']) > 4096:
            raise ValueError('Missing or excessive jobs')
    fields = set()
    for a, b in zip(batches['baseline']['jobs'], batches['candidate']['jobs'], strict=True):
        fields.update(k for k in a['settings'].keys() | b['settings'].keys()
                      if a['settings'].get(k) != b['settings'].get(k))
    if not fields or fields & FIXED_SETTINGS:
        raise ValueError('Pair must change declared non-workload settings only')
    _match(manifests, batches, sorted(fields))
    identity, effective, contracts = None, {}, {}
    for pair in report['pairs']:
        for side in SIDES:
            sample = pair['samples'][side]
            current = _sample_identity(sample)
            if not current['hardware'].get('cpu') or not current['environment']['dependencies']:
                raise ValueError('Hardware/dependency evidence is incomplete')
            if identity is not None and current != identity:
                raise ValueError('Hardware/repository/assets/weights/corrections drift')
            identity = current
            frozen = batches[side]['jobs']
            jobs = sample['runs'][0]['jobs']
            manifest = manifests[side][1]
            if sample.get('model_state') != manifest['model_state'] or sample.get('manifest') != manifest:
                raise ValueError('Actual model lifecycle/manifest differs from frozen protocol')
            cache = sample.get('cache')
            if not isinstance(cache, dict) or set(cache) != {'filesystem', 'intermediate', 'output'} or any(not isinstance(v, str) or not v for v in cache.values()):
                raise ValueError('Actual cache lifecycle is missing')
            warmup = sample.get('warmup')
            if (not isinstance(warmup, list) or (manifest['model_state'] == 'warm_session' and len(warmup) != len(frozen))
                    or (manifest['model_state'] == 'cold_process' and warmup)):
                raise ValueError('Actual full warmup records differ from protocol')
            warm_settings = []
            for index, (job, expected) in enumerate(zip(warmup, frozen)):
                if job.get('status') != 'complete' or job.get('model_reused') != (index > 0):
                    raise ValueError('Warmup lifecycle is incomplete or inconsistent')
                warm_settings.append(_job_contract(job, expected))
            if len(jobs) != len(frozen):
                raise ValueError('Measured/frozen job counts differ')
            settings = []
            for index, (job, expected) in enumerate(zip(jobs, frozen, strict=True)):
                if job.get('model_reused') != (manifest['model_state'] == 'warm_session' or index > 0):
                    raise ValueError('Measured model reuse differs from lifecycle')
                settings.append(_job_contract(job, expected))
            contract = {'cache': cache, 'warmup_settings': warm_settings,
                        'quality_templates': manifest.get('quality_manifests'),
                        'quality': [_quality_contract(sample, i, j) for i, j in enumerate(jobs)]}
            if side in contracts and contract != contracts[side]:
                raise ValueError('Measurement/quality contracts drift across repetitions')
            contracts[side] = contract
            if side in effective:
                _match_effective(settings, effective[side], [])
            effective[side] = settings
            if len(effective) == 2:
                _match_effective(effective['baseline'], effective['candidate'], fields)
    conditions = report.get('conditions')
    if not isinstance(conditions, dict) or any(not conditions.get(k) for k in
            ('order', 'selection', 'filesystem', 'output', 'warm_session', 'timing', 'quality')):
        raise ValueError('Missing timing/cache/warmup conditions')
    return {'identity': identity, 'conditions': conditions, 'measurement_contracts': contracts,
            'batches': {s: {**batches[s], 'jobs': [{k: v for k, v in j.items() if k != 'output_dir'}
                         for j in batches[s]['jobs']]} for s in SIDES},
            'model_state': manifests['baseline'][1]['model_state'], 'effective_settings': effective,
            'changed_settings': sorted(fields), 'repetitions': len(report['pairs'])}


def _policy(path, expected):
    policy, raw = read_object(Path(path), max_bytes=1024 * 1024, max_depth=32, regular_file=True)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Policy bytes differ from the externally pinned SHA256')
    keys = {'schema_version', 'kind', 'context_sha256', 'repetitions', 'performance',
            'quality', 'required_evidence', 'visual_review_required', 'accepted_tradeoffs'}
    if (set(policy) != keys or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or policy['kind'] != 'regression_policy'):
        raise ValueError('Unsupported regression policy schema')
    if (not isinstance(policy['context_sha256'], str) or len(policy['context_sha256']) != 64
            or type(policy['repetitions']) is not int or not 3 <= policy['repetitions'] <= MAX_PAIRS):
        raise ValueError('Policy requires exact context digest and 3..128 repetitions')
    perf = policy['performance']
    if (not isinstance(perf, dict) or set(perf) != {'max_elapsed_regression', 'max_relative_spread'}
            or any(not _number(v) or v < 0 for v in perf.values())):
        raise ValueError('Performance requires finite nonnegative relative limits')
    quality = policy['quality']
    if not isinstance(quality, list) or len(quality) > 256:
        raise ValueError('Quality tolerances must be a bounded list')
    seen = set()
    for item in quality:
        if (not isinstance(item, dict) or set(item) != {'job', 'metric', 'units', 'max_increase'}
                or type(item['job']) is not int or item['job'] < 0 or item['metric'] not in METRICS
                or item['units'] != METRICS[item['metric']] or not _number(item['max_increase'])
                or item['max_increase'] < 0 or (item['job'], item['metric']) in seen):
            raise ValueError('Quality requires distinct supported job/metric/unit/absolute tolerance')
        seen.add((item['job'], item['metric']))
    if (type(policy['visual_review_required']) is not bool
            or not isinstance(policy['required_evidence'], list)
            or len(policy['required_evidence']) > 32
            or any(v not in ('numerical_parity', 'observed_memory') for v in policy['required_evidence'])
            or len(set(policy['required_evidence'])) != len(policy['required_evidence'])
            or not isinstance(policy['accepted_tradeoffs'], list) or len(policy['accepted_tradeoffs']) > 32
            or any(not isinstance(v, str) or not v.strip() or len(v) > 4096 for v in policy['accepted_tradeoffs'])):
        raise ValueError('Invalid explicit evidence/review/tradeoff contract')
    return policy, raw


def evaluate(report_path, policy_path, policy_sha256, *, check=None):
    """Return immutable JSON evidence; invalid/absent evidence never becomes a pass."""
    result = {'schema_version': 1, 'kind': 'regression_gate', 'status': 'not_verified', 'checks': [],
              'predeclaration': 'NOT VERIFIED: pinned bytes do not prove chronological predeclaration.',
              'scope': 'Report-level screen, not product acceptance or hardware performance certification.'}
    try:
        _check(check)
        policy, policy_raw = _policy(policy_path, policy_sha256)
        result['policy'] = {'path': str(Path(policy_path).absolute()), 'sha256': hashlib.sha256(policy_raw).hexdigest(),
                            'bytes': len(policy_raw), 'value': policy}
        report, raw = read_report(Path(report_path), regular_file=True)
        result['source'] = {'path': str(Path(report_path).absolute()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        if report['kind'] != 'inference_pair' or report.get('status') != 'complete':
            raise ValueError('Only completed inference_pair evidence can be screened')
        if len(report['pairs']) != policy['repetitions']:
            raise ValueError('Actual repetitions differ from declared policy')
        actual = context(report)
        result['context_sha256'] = digest(actual)
        if result['context_sha256'] != policy['context_sha256']:
            raise ValueError('Workload/hardware/settings/cache context differs from policy')
        checks = result['checks']
        times = {s: [p['samples'][s]['runs'][0]['elapsed_seconds'] for p in report['pairs']] for s in SIDES}
        ratios = [b / a for a, b in zip(times['baseline'], times['candidate'], strict=True)]
        if any(not _number(r) or r <= 0 for r in ratios):
            raise ValueError('Timing ratios are not representable as finite positive numbers')
        spread = max((max(v) - min(v)) / statistics.median(v) for v in [*times.values(), ratios])
        limit = 1 + policy['performance']['max_elapsed_regression']
        if not _number(spread) or not _number(limit):
            raise ValueError('Derived timing spread/limit is not finite')
        noisy = spread > policy['performance']['max_relative_spread']
        # The observed envelope is descriptive, not a confidence interval.
        status = ('inconclusive' if noisy or min(ratios) <= limit < max(ratios)
                  else 'fail' if min(ratios) > limit else 'pass')
        checks.append({'name': 'elapsed_seconds', 'status': status, 'candidate_over_baseline': ratios,
                       'raw_seconds': times, 'relative_spread': spread, 'limit_ratio': limit})
        for item in policy['quality']:
            _check(check)
            row = {'name': item['metric'], 'job': item['job'], 'status': 'not_verified'}
            checks.append(row)
            try:
                deltas, prior = [], None
                for pair in report['pairs']:
                    reports = [_quality(pair['samples'][s], item['job']) for s in SIDES]
                    signature = [_repetition_contract(q) for q in reports]
                    if prior is not None and signature != prior:
                        raise ValueError('Quality measurement/reference contracts drift across repetitions')
                    prior = signature
                    a, b = _compatibility(reports[0], reports[1], item['metric'])
                    deltas.append(b['value'] - a['value'])
                row.update(status='fail' if max(deltas) > item['max_increase'] else 'pass',
                           raw_increases=deltas, max_increase=item['max_increase'], units=item['units'])
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                row['reason'] = str(exc)
        for name in policy['required_evidence'] + (['visual_review'] if policy['visual_review_required'] else []):
            checks.append({'name': name, 'status': 'not_verified',
                           'reason': 'No supported independently bound evidence in this inference_pair schema'})
        statuses = {r['status'] for r in checks}
        result['status'] = next((s for s in ('fail', 'not_verified', 'inconclusive') if s in statuses), 'pass')
        _check(check)
    except (CancelledError, KeyboardInterrupt):
        result.update(status='cancelled', error='Gate evaluation cancelled')
    except (ValueError, KeyError, TypeError, AttributeError, OSError, OverflowError, ZeroDivisionError) as exc:
        result.update(status='not_verified', error=str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True, help='Externally pinned exact policy byte digest; not chronology proof.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from .output_paths import prepare_report_destination

    guard = prepare_report_destination(args.output, (), protected_paths=(args.report, args.policy))
    guard()
    result = evaluate(args.report, args.policy, args.policy_sha256)
    guard()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Regression screen {result['status']}; report: {args.output}")
    return 130 if result['status'] == 'cancelled' else 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
