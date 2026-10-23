"""Bounded strict inspection/export of real benchmark reports, without metrics synthesis."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .local_json import read_object

MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_VIEW_CHARS = 2 * 1024 * 1024
MAX_REPORT_DEPTH = 128


def read_report(path: Path, *, regular_file: bool = False) -> tuple[dict[str, Any], bytes]:
    try:
        report, raw = read_object(path, max_bytes=MAX_REPORT_BYTES, max_depth=MAX_REPORT_DEPTH,
                                  regular_file=regular_file)
    except TypeError as exc:
        raise ValueError('Benchmark report must be an object.') from exc
    if not isinstance(report, dict) or type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('Unsupported benchmark report schema.')
    if not isinstance(report.get('kind'), str) or report['kind'] not in {'cpu_correctness', 'inference', 'inference_pair', 'production_stress', 'decode_reuse'}:
        raise ValueError('Unsupported benchmark report kind.')
    summary = report.get('summary')
    if not isinstance(summary, dict) or type(summary.get('success')) is not bool:
        raise ValueError('Benchmark report requires actual summary and runs.')
    if report['kind'] == 'decode_reuse':
        from .benchmark_decode_reuse import validate_report

        validate_report(report)
    elif report['kind'] == 'production_stress':
        _validate_stress(report)
    elif report['kind'] == 'inference_pair':
        _validate_pair(report)
    elif not isinstance(report.get('runs'), list):
        raise ValueError('Benchmark report requires actual runs.')
    return report, raw



def _validate_pair(report: dict[str, Any]) -> None:
    pairs, summary = report.get('pairs'), report['summary']
    if (not isinstance(pairs, list) or not isinstance(report.get('status'), str)
            or report['status'] not in {'complete', 'failed', 'cancelled'}):
        raise ValueError('Paired report requires terminal status and actual pairs.')
    success = summary['success']
    speedup = summary.get('speedup')
    if success:
        if (report['status'] != 'complete' or len(pairs) < 3 or type(speedup) not in (int, float)
                or not math.isfinite(speedup) or speedup <= 0):
            raise ValueError('Completed comparison requires at least three pairs and a finite positive speedup.')
    elif report['status'] == 'complete' or speedup is not None:
        raise ValueError('Partial or failed comparison cannot report a completed speedup.')
    for index, pair in enumerate(pairs):
        order = ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline']
        if (not isinstance(pair, dict) or type(pair.get('index')) is not int or pair['index'] != index
                or pair.get('order') != order or not isinstance(pair.get('samples'), dict)):
            raise ValueError('Paired report has invalid ordering or samples.')
        samples = pair['samples']
        if set(samples) - {'baseline', 'candidate'} or (success and set(samples) != set(order)):
            raise ValueError('Paired report has missing or unknown sides.')
        if any(not isinstance(sample, dict) for sample in samples.values()):
            raise ValueError('Paired report samples must contain actual report objects.')
        if success:
            for sample in samples.values():
                runs = sample.get('runs')
                sample_summary, inference = sample.get('summary'), sample.get('inference')
                if (not isinstance(sample_summary, dict) or sample_summary.get('success') is not True
                        or not isinstance(inference, dict) or inference.get('status') != 'measured'
                        or not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict)):
                    raise ValueError('Completed pairs require successful measured sample reports.')
                elapsed, jobs = runs[0].get('elapsed_seconds'), runs[0].get('jobs')
                if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                        or not math.isfinite(elapsed) or elapsed <= 0
                        or not isinstance(jobs, list) or not jobs
                        or any(not isinstance(job, dict) or job.get('status') != 'complete' for job in jobs)):
                    raise ValueError('Completed pairs require finite timings and completed jobs.')


STRESS_SCENARIOS = {'long_clip', 'batch50', 'repeated', 'concurrent_isolated', 'cancel',
                    'interruption_recovery', 'memory_growth', 'encoder_failure', 'disk_full'}


def _validate_stress(report: dict[str, Any]) -> None:
    required = {'schema_version', 'kind', 'status', 'summary', 'cases', 'manifest', 'limits',
                'artifact_dir', 'scope', 'acceptance_verdict'}
    if (not required <= set(report) or set(report) - required - {'created_utc', 'error'}
            or report['status'] not in ('complete', 'failed', 'cancelled')
            or not isinstance(report['scope'], str) or not report['scope'].strip()
            or not isinstance(report['limits'], dict) or report['acceptance_verdict'] is not None):
        raise ValueError('Stress report requires the complete bounded terminal schema and no acceptance verdict.')
    for key in ('error', 'created_utc'):
        if report.get(key) is not None and not isinstance(report[key], str):
            raise ValueError('Stress terminal error/time must be text or null.')
    artifact = report['artifact_dir']
    if artifact is not None and (not isinstance(artifact, str) or not artifact.strip()):
        raise ValueError('Stress artifact directory must be text or null.')
    manifest = report['manifest']
    if (not isinstance(manifest, dict) or set(manifest) != {'path', 'sha256', 'bytes'}
            or not isinstance(manifest['path'], str) or not manifest['path'].strip()):
        raise ValueError('Stress report requires captured manifest provenance.')
    valid_manifest = (isinstance(manifest['sha256'], str)
                      and re.fullmatch('[0-9a-f]{64}', manifest['sha256']) is not None
                      and type(manifest['bytes']) is int and manifest['bytes'] >= 0)
    if not valid_manifest and not (report['status'] == 'failed' and manifest['sha256'] is None
                                   and manifest['bytes'] is None and report.get('error')):
        raise ValueError('Stress report manifest hash/size is invalid.')
    rows = report['cases']
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('Stress report allows at most 64 canonical case records.')
    counts = dict.fromkeys(('measured', 'not_verified', 'failed', 'cancelled'), 0)
    required_cases = required_measured = 0
    cleanup_verified = True
    required_scenarios, seen = set(), set()
    fields = {'id', 'scenario', 'required', 'status', 'success', 'reason', 'error',
              'artifact_dir', 'cleanup', 'evidence'}
    required_success = True
    for row in rows:
        if (not isinstance(row, dict) or not fields <= set(row) or set(row) - fields - {'duration_seconds'}
                or not isinstance(row['id'], str) or row['id'] in seen
                or not isinstance(row['scenario'], str) or row['scenario'] not in STRESS_SCENARIOS
                or not isinstance(row['status'], str) or row['status'] not in counts
                or type(row['required']) is not bool or type(row['success']) is not bool
                or not isinstance(row['evidence'], dict)):
            raise ValueError('Stress cases require unique identities and exact scenario/status/evidence fields.')
        identity = row['id']
        if (re.fullmatch('[A-Za-z0-9_.-]{1,64}', identity) is None
                and not (identity == '$missing:' + row['scenario'] and row['required'] is True
                         and row['success'] is False
                         and (row['status'] == 'not_verified'
                              or row['status'] == 'cancelled' and report['status'] == 'cancelled'))):
            raise ValueError('Stress case identifier is invalid.')
        seen.add(identity)
        for key in ('reason', 'error', 'artifact_dir'):
            if row[key] is not None and not isinstance(row[key], str):
                raise ValueError('Stress case reasons/errors/artifacts must be text or null.')
        duration = row.get('duration_seconds')
        if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
            raise ValueError('Stress case duration must be finite nonnegative seconds or null.')
        cleanup = row['cleanup']
        if (not isinstance(cleanup, dict) or set(cleanup) != {'verified', 'owners'}
                or type(cleanup['verified']) is not bool or not isinstance(cleanup['owners'], list)
                or len(cleanup['owners']) > 256):
            raise ValueError('Stress cleanup requires a verified flag and bounded owner inventory.')
        if row['success'] and (row['status'] != 'measured' or not cleanup['verified']):
            raise ValueError('Only measured cases with verified cleanup can report success.')
        counts[row['status']] += 1
        cleanup_verified = cleanup_verified and cleanup['verified']
        if row['required']:
            required_cases += 1
            required_measured += row['status'] == 'measured'
            required_success = required_success and row['status'] == 'measured' and row['success']
            required_scenarios.add(row['scenario'])
    summary = report['summary']
    expected = {'expected_cases': len(rows), 'required_cases': required_cases,
                'required_measured_cases': required_measured,
                **{status + '_cases': count for status, count in counts.items()}}
    if (set(summary) != set(expected) | {'success', 'cleanup_verified'}
            or type(summary['cleanup_verified']) is not bool or summary['cleanup_verified'] != cleanup_verified
            or any(type(summary.get(key)) is not int or summary[key] != value for key, value in expected.items())):
        raise ValueError('Stress summary counts and cleanup must match the recorded cases exactly.')
    if report['status'] == 'complete':
        if required_scenarios != STRESS_SCENARIOS or counts['failed'] or counts['cancelled'] or not cleanup_verified:
            raise ValueError('Completed stress reports require all nine required scenarios and verified cleanup.')
        if any(row['status'] == 'measured' and not row['success'] for row in rows):
            raise ValueError('Unsuccessful measured stress cases require a failed terminal report.')
    if summary['success'] != (report['status'] == 'complete' and required_success and cleanup_verified):
        raise ValueError('Stress success requires measured successful required cases and verified cleanup.')
    if report['status'] == 'failed' and not report.get('error'):
        raise ValueError('Failed stress reports require an explicit terminal error.')


def report_view(report: dict[str, Any]) -> str:
    def omit_logs(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: omit_logs(item) for key, item in value.items() if key not in {'stdout', 'stderr'}}
        if isinstance(value, list):
            return [omit_logs(item) for item in value]
        return value
    priority = ('kind', 'status', 'scope', 'summary', 'cases', 'results', 'model_state', 'summary', 'inference', 'error', 'warnings', 'pairs', 'quality_pairs', 'runs')
    ordered = {key: report[key] for key in priority if key in report}
    ordered.update({key: value for key, value in report.items() if key not in ordered})
    text = json.dumps(omit_logs(ordered), indent=2, allow_nan=False)
    if report.get('kind') == 'production_stress':
        lines = ['Production stress: ' + report['status'], 'Scope: ' + report['scope'],
                 'Local observations only; full product acceptance is not asserted.']
        summary = report['summary']
        lines.append(f"Required cases measured: {summary['required_measured_cases']} / {summary['required_cases']}; NOT VERIFIED: {summary['not_verified_cases']}; failed: {summary['failed_cases']}; cancelled: {summary['cancelled_cases']}.")
        lines.append('Cleanup: ' + ('verified' if summary['cleanup_verified'] else 'NOT VERIFIED'))
        for row in report['cases']:
            state = row['status'].upper().replace('_', ' ')
            cleanup = 'verified' if row['cleanup']['verified'] else 'NOT VERIFIED'
            lines.append(f"{row['id']} ({row['scenario']}): {state}; cleanup {cleanup}. {row['reason'] or row['error'] or ''}")
        text = '\n'.join(lines) + '\n\nRecorded report:\n' + text
    if len(text) > MAX_VIEW_CHARS:
        text = text[:MAX_VIEW_CHARS] + '\n[Inspection truncated; open the full local report. No metrics were recomputed.]'
    return text


def export_report(source: Path, destination: Path) -> None:
    _, raw = read_report(source)
    if destination.suffix.lower() != '.json':
        raise ValueError('Choose a new .json file for report export.')
    # Exclusive creation prevents overwriting media, source manifests or old reports.
    with destination.open('xb') as stream:
        stream.write(raw)
