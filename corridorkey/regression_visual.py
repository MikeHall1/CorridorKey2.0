"""Bound local reviewer declarations, never authenticated human inspection."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .local_json import _depth, read_object
from .project_collection import admit_file, assert_pin, check_now
from .regression_numerical import fence

MAX_DECLARATION_BYTES = 256 * 1024
MAX_RECEIPT_BYTES = 4 * 1024**2
TRUST_SCOPE = ('Reviewer name, authorship, actual inspection, calibrated display and chronological approval '
               'are not authenticated; only the local declaration and selected byte identities are bound.')


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_policy(value):
    if (not isinstance(value, dict) or set(value) != {'mode', 'selections'} or value['mode'] not in
            ('bound_declaration_required', 'verified_human_review_required')):
        raise ValueError('Visual policy requires an explicit declaration/human requirement and selections')
    _selections(value['selections'])


def _selections(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError('Visual selections require1..128 pair/job rows')
    previous, total = (-1, -1), 0
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'pair', 'job', 'source_frames'}
                or type(row['pair']) is not int or not 0 <= row['pair'] < 128
                or type(row['job']) is not int or not 0 <= row['job'] < 256
                or (row['pair'], row['job']) <= previous):
            raise ValueError('Visual pair/job selections must be bounded, unique and ordered')
        previous = (row['pair'], row['job'])
        frames = row['source_frames']
        if (not isinstance(frames, list) or not 1 <= len(frames) <= 120
                or any(type(f) is not int or not 0 <= f < 2**63 for f in frames)
                or frames != sorted(set(frames))):
            raise ValueError('Visual absolute source frames must be bounded, ordered and unique')
        total += len(frames)
        if total > 1024:
            raise ValueError('Visual selected-frame budget exceeds1024')


def _read(path, expected, maximum):
    if not _sha(expected):
        raise ValueError('Visual input requires an externally supplied exact SHA256')
    pin = admit_file(path)
    value, raw = read_object(Path(pin.path), max_bytes=maximum, max_depth=64, regular_file=True)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Visual input byte receipt differs')
    assert_pin(pin)
    return value, {'path': pin.path, 'fingerprint': list(pin.fingerprint), 'sha256': expected}


def _declaration(value, policy, policy_sha, numerical_sha):
    fields = {'schema_version', 'kind', 'reviewer', 'outcome', 'policy_sha256',
              'numerical_evidence_sha256', 'selections', 'accepted_tradeoffs', 'observations'}
    if (set(value) != fields or type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['kind'] != 'regression_visual_declaration'
            or not isinstance(value['reviewer'], str) or not value['reviewer'].strip() or len(value['reviewer']) > 256
            or value['outcome'] not in ('accept', 'reject', 'unable_to_assess')
            or not isinstance(value['observations'], str) or len(value['observations']) > 4096):
        raise ValueError('Expected bounded reviewer-authored visual declaration fields')
    _selections(value['selections'])
    if (value['policy_sha256'] != policy_sha or value['numerical_evidence_sha256'] != numerical_sha
            or value['selections'] != policy['visual']['selections']
            or value['accepted_tradeoffs'] != policy['accepted_tradeoffs']):
        raise ValueError('Reviewer-authored policy/numerical/selections/tradeoffs differ; capture never fills claims')


def _bindings(numerical, selections):
    jobs = {(j['pair'], j['job']): j for j in numerical['jobs']}
    assets = {a['path']: a['sha256'] for a in numerical['assets']}
    bindings = []
    for selection in selections:
        key = selection['pair'], selection['job']
        if key not in jobs:
            raise ValueError('Visual selection names an unmeasured pair/job')
        sides = {}
        for name in ('baseline', 'candidate'):
            job = jobs[key]['sides'][name]
            start = job['source']['start_frame']
            frames = {}
            for absolute in selection['source_frames']:
                local = str(absolute - start)
                if any(local not in files for files in job['exports'].values()):
                    raise ValueError('Visual selection is outside actual pinned native exports')
                frames[str(absolute)] = {kind: {'path': files[local], 'sha256': assets[files[local]]}
                                         for kind, files in job['exports'].items()}
            sides[name] = {'job_id': job['job_id'], 'input': job['input'], 'source': job['source'],
                           'execution': job['execution'], 'contracts': job['contracts'], 'dtype': job['dtype'],
                           'frames': frames}
        bindings.append({**selection, 'sides': sides})
    return bindings


def capture(report_path, policy_path, policy_sha256, numerical_path, numerical_sha256,
            declaration_path, declaration_sha256, *, check=None):
    """Bind existing explicit claims; never synthesize reviewer identity or approval."""
    from .regression_gate import _policy, capture_numerical
    check_now(check)
    policy_pin = admit_file(policy_path)
    policy, _ = _policy(policy_path, policy_sha256)
    if policy['schema_version'] != 4:
        raise ValueError('Visual capture requires a version4 visual policy')
    supplied, numerical_pin = _read(numerical_path, numerical_sha256, MAX_RECEIPT_BYTES)
    declaration, declaration_pin = _read(declaration_path, declaration_sha256, MAX_DECLARATION_BYTES)
    _declaration(declaration, policy, policy_sha256, numerical_sha256)
    current = capture_numerical(report_path, policy_path, policy_sha256, check=check)
    if supplied != current:
        raise ValueError('Visual numerical receipt/report/policy/export identities differ')
    bindings = _bindings(current, policy['visual']['selections'])
    policy_asset = {'path': policy_pin.path, 'fingerprint': list(policy_pin.fingerprint), 'sha256': policy_sha256}
    assets = {a['path']: a for a in [*current['assets'], policy_asset, numerical_pin, declaration_pin]}
    result = {'schema_version': 1, 'kind': 'regression_visual_evidence',
              'policy_sha256': policy_sha256, 'numerical_evidence_sha256': numerical_sha256,
              'report_sha256': current['report_sha256'], 'context_sha256': current['context_sha256'],
              'requirement': policy['visual']['mode'],
              'declaration': {'path': declaration_pin['path'], 'sha256': declaration_sha256, 'value': declaration},
              'bindings': bindings, 'assets': [assets[p] for p in sorted(assets)],
              'protected_directories': current['protected_directories'], 'trust_scope': TRUST_SCOPE}
    encoded = json.dumps(result, allow_nan=False).encode()
    _depth(encoded, 64)
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError('Visual receipt exceeds4MiB metadata bound')
    fence(result, check=check)
    check_now(check)
    return result


def admit(path, expected_sha, report_path, policy_path, policy_sha,
          numerical_path, numerical_sha, *, check=None):
    """Reconstruct current inputs rather than trusting the supplied observation fields."""
    supplied, receipt_pin = _read(path, expected_sha, MAX_RECEIPT_BYTES)
    declaration = supplied.get('declaration')
    if (not isinstance(declaration, dict) or set(declaration) != {'path', 'sha256', 'value'}
            or not isinstance(declaration['path'], str)):
        raise ValueError('Visual receipt lacks its reviewer-authored declaration input')
    current = capture(report_path, policy_path, policy_sha, numerical_path, numerical_sha,
                      declaration['path'], declaration['sha256'], check=check)
    if supplied != current:
        raise ValueError('Visual receipt/current declaration/export bindings differ')
    outcome = current['declaration']['value']['outcome']
    status = 'fail' if outcome == 'reject' else 'pass' if outcome == 'accept' else 'not_verified'
    if current['requirement'] == 'verified_human_review_required' and status == 'pass':
        status = 'not_verified'
    row = {'name': 'visual_review', 'status': status, 'requirement': current['requirement'],
           'declared_outcome': outcome, 'reviewer_provided_name': current['declaration']['value']['reviewer'],
           'human_inspection_status': 'not_verified', 'trust_scope': TRUST_SCOPE,
           'evidence_sha256': expected_sha}
    current['assets'].append(receipt_pin)
    fence(current, check=check)
    return row, current
