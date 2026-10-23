"""Checksummed per-batch attempt journals; only sealed outputs can satisfy recovery."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

from .identities import file_sha256
from .recovery_identity import canonical
from .recovery_store import CacheInvalid, relative_name, write_record


def inventory(files, *, check=None):
    return [{'name': relative_name(name), 'bytes': Path(path).stat().st_size,
             'sha256': file_sha256(path, check=check)} for name, path in sorted(files.items())]


def verify_inventory(root, items, *, check=None):
    root = Path(root).resolve()
    if not isinstance(items, list) or not items or not root.is_dir():
        return False
    expected = {}
    for item in items:
        name = relative_name(item['name'])
        path = root / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root) or
                type(item['bytes']) is not int or item['bytes'] < 0 or path.stat().st_size != item['bytes'] or
                file_sha256(path, check=check) != item['sha256']):
            return False
        expected.setdefault(path.parent, set()).add(path.name)
    for directory, names in expected.items():
        if directory.is_symlink() or {p.name for p in directory.iterdir()} - {'.replay-complete.json'} != names:
            return False
    return True


def export_paths(record, *, check=None):
    if record.get('status') != 'complete':
        raise ValueError('only completed shared-engine jobs can be sealed')
    root = Path(record['output_dir']).resolve()
    diagnostics = record['diagnostics']
    count = diagnostics['frames_submitted']
    if type(count) is not int or count <= 0:
        raise ValueError('completed exports require a positive submitted-frame count')
    files = {}
    outputs = diagnostics['outputs']
    if not outputs:
        raise ValueError('completed job has no export outputs')
    for name, path in outputs.items():
        if check is not None:
            check()
        if diagnostics['output_contracts'][name]['format'] != 'exr':
            raise ValueError('durable replay currently accepts EXR exports only')
        directory = Path(path)
        if (directory.is_symlink() or not directory.is_dir() or directory.resolve() == root or
                not directory.resolve().is_relative_to(root)):
            raise ValueError('export directories must be regular children of the job output root')
        indices = set()
        for file in directory.iterdir():
            if check is not None:
                check()
            match = re.fullmatch(r'.+_(\d{6,})\.exr', file.name)
            if file.is_symlink() or not file.is_file() or match is None:
                raise ValueError('unexpected/nonregular file in completed EXR output')
            frame = int(match.group(1))
            if frame in indices:
                raise ValueError('duplicate completed EXR frame index')
            indices.add(frame)
            files[file.resolve().relative_to(root).as_posix()] = file
        if len(indices) != count or indices != set(range(count)):
            raise ValueError('completed EXR outputs must cover every submitted frame')
    return files


def remap_outputs(record, destination):
    original = Path(record['output_dir']).resolve()
    return {name: str(Path(destination) / Path(path).resolve().relative_to(original))
            for name, path in record['diagnostics']['outputs'].items()}


class AttemptJournal:
    def __init__(self, store, requests, *, resume=False):
        if type(resume) is not bool:
            raise TypeError('resume must be explicit boolean')
        store._require_owner()
        self.store = store
        self.request_key = hashlib.sha256(canonical(requests).encode()).hexdigest()
        directory = store.root / 'journals'
        if directory.is_symlink():
            raise ValueError('journal directory cannot be a symlink')
        directory.mkdir(exist_ok=True)
        self.path = directory / f'{self.request_key}.json'
        if self.path.exists():
            envelope = json.loads(self.path.read_text())
            if not isinstance(envelope, dict) or not isinstance(envelope.get('payload'), dict):
                raise CacheInvalid('invalid journal envelope')
            payload = envelope['payload']
            if (envelope['sha256'] != hashlib.sha256(canonical(payload).encode()).hexdigest() or
                    type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or payload.get('kind') != 'recovery_attempts' or
                    payload.get('owner') != store.owner or payload.get('request_key') != self.request_key or
                    not isinstance(payload.get('jobs'), list) or len(payload['jobs']) != len(requests)):
                raise CacheInvalid('journal checksum/ownership/request mismatch')
            if not resume and any(payload['jobs']):
                raise ValueError('this batch has an existing journal; use explicit resume')
            self.data = payload
            for attempts in self.data['jobs']:
                if not isinstance(attempts, list):
                    raise CacheInvalid('invalid journal attempt list')
                for attempt in attempts:
                    if attempt.get('status') not in {'running', 'complete', 'failed', 'cancelled', 'interrupted'}:
                        raise CacheInvalid('unknown journal attempt state')
                    if attempt['status'] == 'running':
                        attempt['status'] = 'interrupted'
        else:
            self.data = {'schema_version': 1, 'kind': 'recovery_attempts', 'owner': store.owner,
                         'request_key': self.request_key, 'jobs': [[] for _ in requests]}
        self.persist()

    def persist(self):
        self.store._require_owner()
        payload = json.loads(canonical(self.data))
        write_record(self.path, {'payload': payload, 'sha256': hashlib.sha256(canonical(payload).encode()).hexdigest()})

    def start(self, index, identity, output_dir, mode):
        attempt = {'id': uuid.uuid4().hex, 'key': identity.key, 'output_dir': str(Path(output_dir).resolve()),
                   'mode': mode, 'status': 'running'}
        self.data['jobs'][index].append(attempt)
        self.persist()
        return attempt['id']

    def finish(self, index, attempt_id, *, status, artifacts=None, origin_record=None, error=None, check=None):
        if status not in {'complete', 'failed', 'cancelled'}:
            raise ValueError('invalid terminal recovery status')
        attempt = self.data['jobs'][index][-1]
        if attempt['id'] != attempt_id or attempt['status'] != 'running':
            raise ValueError('journal transition must finish its current running attempt')
        if status == 'complete' and (not artifacts or not origin_record or origin_record.get('status') != 'complete'):
            raise ValueError('completed attempt requires sealed artifacts and a completed origin job')
        if status == 'complete' and not verify_inventory(attempt['output_dir'], artifacts, check=check):
            raise CacheInvalid('cannot complete an attempt with unverified/missing artifacts')
        attempt.update(status=status, artifacts=artifacts, origin_record=origin_record, error=error)
        self.persist()

    def completed(self, index, identity, *, check=None):
        for attempt in reversed(self.data['jobs'][index]):
            if (attempt['key'] == identity.key and attempt['status'] == 'complete' and
                    verify_inventory(attempt['output_dir'], attempt['artifacts'], check=check)):
                return json.loads(canonical(attempt))
        return None

    def snapshot(self):
        return json.loads(canonical(self.data))
