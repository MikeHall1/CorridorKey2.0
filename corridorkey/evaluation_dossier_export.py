"""Fresh local analysis exports and assignment-blinded human-review packets."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import logging
import os
import secrets
import stat
import uuid
from pathlib import Path
from typing import Any

from .control import CancelledError, ProgressEvent

MAX_BYTES = (1 << 63) - 1
NOTICE = ('Human review pending. Assignment labels are concealed; embedded media metadata '
          'anonymity is not verified. Share only this review directory with reviewers. '
          'View raw media with a suitable colour-managed viewer; no HDR-to-display conversion is implied.')


def _check(check):
    if check is not None:
        check()


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode('utf-8') + b'\n'


def _write(path: Path, data: bytes):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _cell(value):
    text = '' if value is None else str(value)
    if isinstance(value, str) and text.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n')):
        text = "'" + text
    return text


def _analysis_csv(report):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['group', 'id', 'split', 'metric', 'units', 'baseline_median',
                     'candidate_median', 'absolute_delta', 'relative_delta', 'status', 'screen'])
    for group in ('cases', 'categories'):
        for entry in report.get(group, []):
            for metric, row in entry.get('metrics', {}).items():
                values = [group, entry.get('id', entry.get('category')), entry.get('split'), metric]
                values.extend(row.get(key) for key in ('units', 'baseline_median', 'candidate_median',
                                                       'absolute_delta', 'relative_delta', 'status', 'screen'))
                writer.writerow([_cell(value) for value in values])
    return stream.getvalue().encode('utf-8')


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _view_contract(row):
    contract = row.get('view_contract', {})
    values = {'colour_space': contract.get('colour_space'),
              'association': contract.get('associations', {}).get('candidate'),
              'rgb_layout': contract.get('rgb_layouts', {}).get(row['role']),
              'alpha_channel': contract.get('alpha_channels', contract.get('channels', {})).get(row['role'])}
    allowed = {'colour_space': {'linear_srgb_bt709'}, 'association': {'straight', 'premultiplied'},
               'rgb_layout': {'rgb', 'rgba'}, 'alpha_channel': {'gray', 'alpha', 'replicated_gray'}}
    return {key: value if value in allowed[key] else 'unavailable' for key, value in values.items()}


def _copy_media(source: Path, destination: Path, expected: dict, check):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(source, flags)
    with os.fdopen(fd, 'rb') as incoming:
        before = os.fstat(incoming.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected['bytes']:
            raise ValueError('Review media is not a regular file of the captured size.')
        digest, remaining = hashlib.sha256(), before.st_size
        with destination.open('xb') as outgoing:
            while remaining:
                _check(check)
                block = incoming.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Review media shortened during export.')
                digest.update(block)
                outgoing.write(block)
                remaining -= len(block)
            if incoming.read(1):
                raise ValueError('Review media grew during export.')
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
        if (_identity(before) != _identity(after) or _identity(after) != _identity(source.stat())
                or digest.hexdigest() != expected['sha256']):
            raise ValueError('Review media changed or differs from measured media.')
    return _identity(after)



def _sealed(destination, marker, files):
    """Recover an actual commit after an OS operation committed then raised."""
    try:
        with (destination / 'COMPLETE.json').open('rb') as stream:
            if stream.read(len(marker) + 1) != marker:
                return False
        for row in files:
            path = destination / row['path']
            with path.open('rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != row['bytes']:
                    return False
                digest, remaining = hashlib.sha256(), row['bytes']
                while remaining:
                    block = stream.read(min(remaining, 1024 * 1024))
                    if not block:
                        return False
                    digest.update(block)
                    remaining -= len(block)
                if stream.read(1) or digest.hexdigest() != row['sha256']:
                    return False
                if (_identity(info) != _identity(os.fstat(stream.fileno())) or
                        _identity(info) != _identity(path.stat())):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyboardInterrupt, SystemExit):
        return False

def export_dossier(report: dict[str, Any], existing_parent, *, include_media=False,
                   max_media_bytes=0, check=None, progress=None):
    """Return a terminal result; an unsealed partial directory is never usable."""
    destination = None
    marker = None
    files = []
    try:
        from .evaluation_dossier import revalidate_inputs

        _check(check)
        if type(include_media) is not bool or type(max_media_bytes) is not int or not 0 <= max_media_bytes <= MAX_BYTES:
            raise ValueError('Media selection must be boolean and budget an integer in [0,2^63-1].')
        report = revalidate_inputs(report, check=check)
        parent = Path(existing_parent).expanduser().resolve(strict=True)
        if not parent.is_dir():
            raise ValueError('Select an existing output parent directory.')
        for raw in report['protected_paths']:
            protected = Path(raw).resolve()
            if parent == protected or parent.is_relative_to(protected):
                raise ValueError('Output parent is inside a protected input/source/output tree.')
        media = report.get('review_media', []) if include_media else []
        if include_media:
            if not media or any(type(row.get('bytes')) is not int or row['bytes'] < 0 for row in media):
                raise ValueError('Selected review media is missing or has no captured byte size.')
            if sum(row['bytes'] for row in media) > max_media_bytes or max_media_bytes == 0:
                raise ValueError('Explicit media byte budget does not cover selected review media.')
            for row in media:
                if Path(row['path']).suffix.lower() not in ('.png', '.exr', '.npy'):
                    raise ValueError('Review media must retain its measured PNG/EXR/NPY format.')
            groups = {}
            for row in media:
                groups.setdefault(row['case_id'], {}).setdefault(row['side'], set()).add((row['frame'], row['role']))
            expected_cases = {case['id'] for case in report['cases']}
            if set(groups) != expected_cases or any(set(sides) != {'baseline', 'candidate'} or
                    sides['baseline'] != sides['candidate'] for sides in groups.values()):
                raise ValueError('Review media must cover both assignments with matching frames/roles for every case.')
        _check(check)
        destination = parent / ('evaluation-dossier-' + uuid.uuid4().hex)
        destination.mkdir(mode=0o700)
        def write(path, data):
            _write(path, data)
            files.append({'path': str(path.relative_to(destination)), 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})

        review = destination / 'review'
        review.mkdir(mode=0o700)
        assignments = {case['id']: {'opaque_id': uuid.uuid4().hex,
                                    'A': 'candidate' if secrets.randbits(1) else 'baseline'}
                       for case in report['cases']}
        copied, public = [], []
        for index, row in enumerate(media):
            _check(check)
            case_id = row['case_id']
            assignment = assignments[case_id]
            label = 'A' if row['side'] == assignment['A'] else 'B'
            role = {'candidate': 'alpha', 'candidate_alpha': 'alpha', 'candidate_rgb': 'rgb'}.get(row['role'])
            if role is None or row['side'] not in ('baseline', 'candidate'):
                raise ValueError('Unsupported review role or engine assignment.')
            filename = f"{assignment['opaque_id']}-{label}-{uuid.uuid4().hex}-{role}{Path(row['path']).suffix.lower()}"
            identity = _copy_media(Path(row['path']), review / filename, row, check)
            copied.append((row, identity))
            files.append({'path': 'review/' + filename, 'bytes': row['bytes'], 'sha256': row['sha256']})
            public.append({'case': assignment['opaque_id'], 'assignment': label, 'media': filename,
                           'frame': row['frame'], 'role': role, 'fps': row['fps'],
                           'view_contract': _view_contract(row)})
            if progress is not None:
                progress(ProgressEvent('copy_review_media', index + 1, len(media)))
        public.sort(key=lambda row: (row['case'], row['frame'], row['role'], row['assignment']))
        write(destination / 'analysis.json', _json(report))
        write(destination / 'analysis.csv', _analysis_csv(report))
        body = html.escape(json.dumps({key: report.get(key) for key in ('coverage', 'policy', 'categories', 'cases')},
                                      indent=2, allow_nan=False))
        write(destination / 'analysis.html', ('<!doctype html><meta charset="utf-8"><title>Evaluation dossier</title>'
                 '<h1>Evaluation dossier</h1><p>Metric screens only. Human review and product acceptance pending.</p>'
                 '<pre>' + body + '</pre>').encode('utf-8'))
        write(destination / 'private-assignment-key.json', _json({'notice': 'Private: do not share with blinded reviewers.',
                 'assignments': assignments, 'source_media': media, 'dossier_token': report['token']}))
        write(review / 'packet.json', _json({'schema_version': 1, 'notice': NOTICE, 'review_status': 'pending',
                 'media_status': 'prepared' if media else 'not_prepared',
                 'cases': sorted(item['opaque_id'] for item in assignments.values()), 'media': public}))
        links = ''.join('<li>' + html.escape(row['case']) + ' ' + row['assignment'] + ': <a href="' +
                        html.escape(row['media'], quote=True) + '">' + html.escape(row['role']) + '</a></li>' for row in public)
        write(review / 'index.html', ('<!doctype html><meta charset="utf-8"><title>Paired review</title><h1>Paired review</h1><p>'
                 + html.escape(NOTICE) + '</p><p>' + ('Media prepared; ratings pending.' if media else
                 'Metadata template only; media preparation and ratings pending.') + '</p><ul>' + links + '</ul>').encode('utf-8'))
        ratings = io.StringIO(newline='')
        writer = csv.writer(ratings)
        writer.writerow(['case', 'observer', 'date', 'confidence', 'preference_A_B_tie',
                         'hair_edges', 'spill', 'temporal', 'composites', 'artifact_notes'])
        for assignment in assignments.values():
            writer.writerow([assignment['opaque_id']] + [''] * 9)
        write(review / 'ratings.csv', ratings.getvalue().encode('utf-8'))
        marker = _json({'dossier_token': report['token'], 'review_status': 'pending',
                        'media_status': 'prepared' if media else 'not_prepared', 'files': files})
        _write(destination / '.complete-pending.json', marker)
        _check(check)
        revalidate_inputs(report, check=check)
        for row, identity in copied:
            if _identity(Path(row['path']).stat()) != identity:
                raise ValueError('Review media changed before export sealing.')
        _check(check)
        os.rename(destination / '.complete-pending.json', destination / 'COMPLETE.json')
        return {'status': 'complete', 'destination': str(destination), 'usable': True, 'error': None,
                'warning': NOTICE, 'partial_path': None}
    except BaseException as exc:
        logging.getLogger(__name__).exception('Evaluation dossier export interrupted or failed')
        if destination is not None and marker is not None and _sealed(destination, marker, files):
            return {'status': 'complete', 'destination': str(destination), 'usable': True, 'error': None,
                    'warning': NOTICE + f' Export sealed before late {type(exc).__name__}.', 'partial_path': None}
        return {'status': 'cancelled' if isinstance(exc, CancelledError) else 'failed', 'destination': None,
                'usable': False, 'error': f'{type(exc).__name__}: {exc}', 'warning': None,
                'partial_path': str(destination) if destination is not None and destination.exists() else None}
