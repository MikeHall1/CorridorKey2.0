"""Bounded provenance for accepted guidance rasters, without optional providers.

Sample RGB hashes describe what the producer observed. Loading authenticates
the stored files and binding; it does not rerun a provider or decode samples.
"""
import hashlib
import json
import os
import stat
from pathlib import Path

MAX_GUIDANCE = 32
MAX_RESOURCES = 300
MAX_RESOURCE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 512 * 1024 * 1024
MAX_GUIDANCE_RECORD_BYTES = 256 * 1024


def _object(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'guidance {label} has unknown/missing fields')


def _name(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError('guidance identifiers must be bounded nonempty strings')


def _bounded_object(value, label):
    from .corrections import _primitive, canonical

    if not isinstance(value, dict):
        raise TypeError(f'guidance {label} requires a primitive object')
    _primitive(value)
    if len(canonical(value).encode()) > 16 * 1024:
        raise ValueError(f'guidance {label} exceeds 16 KiB')


def _resources(resources):
    from .corrections import _asset

    if not isinstance(resources, list) or len(resources) > MAX_RESOURCES:
        raise ValueError('guidance resources must be a bounded list')
    seen = set()
    for resource in resources:
        _object(resource, ('id', 'role', 'asset'), 'resource')
        _name(resource['id'])
        if resource['id'] in seen or resource['role'] not in ('reference', 'source_sample'):
            raise ValueError('guidance resource IDs/roles must be unique/supported')
        seen.add(resource['id'])
        _asset(resource['asset'])
        if len(resource['asset']['path']) > 4096:
            raise ValueError('guidance resource path exceeds 4096 characters')


def _parameters(parameters):
    from .corrections import _hash

    _object(parameters, ('contract', 'producer', 'settings'), 'parameters')
    if parameters['contract'] != 'corridorkey-guidance-v1':
        raise ValueError('unsupported guidance parameter contract')
    producer = parameters['producer']
    _object(producer, ('id', 'version', 'weights_sha256'), 'producer')
    _name(producer['id'])
    _name(producer['version'])
    weights = producer['weights_sha256']
    if not isinstance(weights, list) or len(weights) > 32:
        raise ValueError('guidance external weight hashes must be a bounded list')
    for value in weights:
        _hash(value)
    if len(set(weights)) != len(weights):
        raise ValueError('guidance external weight hashes must be unique')
    _bounded_object(parameters['settings'], 'settings')


def _observation(observation, *, source, entry, resources, kind):
    from .corrections import _hash, _integer

    _object(observation, ('source_samples', 'reference_samples', 'result_asset_sha256', 'diagnostics'), 'observation')
    if observation['result_asset_sha256'] != entry['asset']['sha256']:
        raise ValueError('guidance result hash differs from generated correction asset')
    _hash(observation['result_asset_sha256'])
    samples = observation['source_samples']
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_RESOURCES:
        raise ValueError('guidance source samples must be a nonempty bounded list')
    frames = []
    for sample in samples:
        _object(sample, ('frame', 'source_frame', 'rgb_sha256'), 'source sample')
        _integer(sample['frame'], 'guidance sample frame')
        _integer(sample['source_frame'], 'guidance source frame')
        if sample['frame'] >= source['frame_count'] or sample['source_frame'] != source['start_frame'] + sample['frame']:
            raise ValueError('guidance source sample has a stale source offset')
        _hash(sample['rgb_sha256'])
        frames.append(sample['frame'])
    if frames != sorted(set(frames)) or entry['frame'] not in frames:
        raise ValueError('guidance samples must be ordered, unique and include generation frame')
    samples = observation['reference_samples']
    if not isinstance(samples, list) or len(samples) > MAX_RESOURCES:
        raise ValueError('guidance reference samples must be a bounded list')
    references = {resource['id'] for resource in resources if resource['role'] == 'reference'}
    sampled, pairs = set(), set()
    for sample in samples:
        _object(sample, ('resource_id', 'frame', 'rgb_sha256'), 'reference sample')
        _name(sample['resource_id'])
        _integer(sample['frame'], 'guidance reference frame')
        _hash(sample['rgb_sha256'])
        pair = sample['resource_id'], sample['frame']
        if sample['resource_id'] not in references or pair in pairs:
            raise ValueError('guidance reference sample requires a unique declared reference/frame')
        pairs.add(pair)
        sampled.add(sample['resource_id'])
    if sampled != references or (kind == 'clean_plate' and not references):
        raise ValueError('guidance must record samples for every reference; clean plate requires a reference')
    _bounded_object(observation['diagnostics'], 'diagnostics')


def guidance_binding(document_data, entry_id, *, kind, resources, parameters, decode, observation):
    """Detached path-free generation inputs; application controls remain editable."""
    from .corrections import _decode_contract, canonical

    _name(entry_id)
    if kind not in ('clean_plate', 'subject_clicks'):
        raise ValueError('unsupported guidance kind')
    entries = [entry for entry in document_data['entries'] if entry['id'] == entry_id]
    if len(entries) != 1 or entries[0]['kind'] not in ('keep', 'remove', 'matte', 'trimap'):
        raise ValueError('guidance must bind one generated raster correction')
    entry = entries[0]
    _resources(resources)
    _parameters(parameters)
    _object(decode, ('source', 'reference'), 'decode')
    _decode_contract(decode['source'])
    if kind == 'clean_plate':
        _decode_contract(decode['reference'])
    elif decode['reference'] is not None:
        raise ValueError('subject-click guidance has no reference decoder')
    _observation(observation, source=document_data['source'], entry=entry, resources=resources, kind=kind)
    return json.loads(canonical({
        'contract': 'corridorkey-guidance-binding-v1', 'source': document_data['source'],
        'entry': {'id': entry_id, 'kind': entry['kind'], 'frame': entry['frame'],
                  'asset': {'sha256': entry['asset']['sha256']}},
        'kind': kind, 'resources': [{'id': item['id'], 'role': item['role'],
                                    'sha256': item['asset']['sha256']} for item in resources],
        'parameters': parameters, 'decode': decode, 'observation': observation,
    }))


def parse_guidance(data):
    """Validate origin/entry bijection and reject stale generation bindings."""
    from .corrections import _hash, canonical

    origins = data['guidance']
    if not isinstance(origins, list) or len(origins) > MAX_GUIDANCE:
        raise ValueError('guidance origins must be a bounded list')
    seen, owners, total = set(), set(), 0
    for origin in origins:
        _object(origin, ('id', 'entry_id', 'kind', 'resources', 'parameters', 'decode',
                         'observation', 'binding', 'binding_sha256'), 'origin')
        _name(origin['id'])
        _name(origin['entry_id'])
        if origin['id'] in seen or origin['entry_id'] in owners:
            raise ValueError('guidance IDs and generated entry owners must be unique')
        seen.add(origin['id'])
        owners.add(origin['entry_id'])
        expected = guidance_binding(data, origin['entry_id'], kind=origin['kind'], resources=origin['resources'],
                                    parameters=origin['parameters'], decode=origin['decode'], observation=origin['observation'])
        _hash(origin['binding_sha256'])
        if origin['binding'] != expected or hashlib.sha256(canonical(expected).encode()).hexdigest() != origin['binding_sha256']:
            raise ValueError('stale or corrupt guidance binding; regenerate or explicitly detach guidance')
        entry = next(entry for entry in data['entries'] if entry['id'] == origin['entry_id'])
        if entry.get('guidance_id') != origin['id']:
            raise ValueError('guidance origin must be marked by its generated entry')
        total += len(origin['resources'])
        if total > MAX_RESOURCES or len(canonical(origin).encode()) > MAX_GUIDANCE_RECORD_BYTES:
            raise ValueError('guidance document resources or origin metadata exceed supported bounds')
    for entry in data['entries']:
        if 'guidance_id' in entry:
            _name(entry['guidance_id'])
            if entry['id'] not in owners or entry['guidance_id'] not in seen:
                raise ValueError('generated entry has an orphan guidance marker')


def attach_guidance(document, entry_id, *, guidance_id, kind, resources, parameters, decode, observation):
    """Return a schema4 clone, leaving document and caller-owned inputs untouched.

    Replacing an existing origin requires explicit removal/detach first. This
    helper increments the document revision; it never edits generation assets.
    """
    from .corrections import CorrectionDocument, canonical

    if not isinstance(document, CorrectionDocument):
        raise TypeError('guidance attachment requires a CorrectionDocument instance')
    _name(guidance_id)
    data = document.to_dict()
    entries = [entry for entry in data['entries'] if entry['id'] == entry_id]
    if len(entries) != 1 or 'guidance_id' in entries[0]:
        raise ValueError('guidance requires one entry without an existing origin')
    binding = guidance_binding(data, entry_id, kind=kind, resources=resources, parameters=parameters,
                               decode=decode, observation=observation)
    data['schema_version'] = max(4, data['schema_version'])
    data.setdefault('tracks', [])
    data.setdefault('guidance', []).append({
        'id': guidance_id, 'entry_id': entry_id, 'kind': kind, 'resources': resources,
        'parameters': parameters, 'decode': decode, 'observation': observation,
        'binding': binding, 'binding_sha256': hashlib.sha256(canonical(binding).encode()).hexdigest(),
    })
    entries[0]['guidance_id'] = guidance_id
    data['revision'] += 1
    return CorrectionDocument.parse(data, document.base_dir)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _admit(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, 'st_file_attributes', 0) & 0x400
            or not 0 < info.st_size <= MAX_RESOURCE_BYTES):
        raise ValueError('guidance resource must be a bounded regular single-link file')
    return info


def verify_guidance_resources(data, base_dir, *, check=None):
    """Preflight all extents, then hash bounded chunks with mutation fences."""
    admitted, total = [], 0
    for origin in data.get('guidance', []):
        for resource in origin['resources']:
            if check is not None:
                check()
            path = Path(base_dir) / resource['asset']['path']
            info = _admit(path)
            total += info.st_size
            if total > MAX_TOTAL_RESOURCE_BYTES:
                raise ValueError('guidance resource bytes exceed 512 MiB')
            admitted.append((path, info, resource['asset']['sha256']))
    for path, info, expected in admitted:
        if check is not None:
            check()
        digest, remaining = hashlib.sha256(), info.st_size
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(info):
                raise ValueError('guidance resource changed before hashing')
            while remaining:
                if check is not None:
                    check()
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError('guidance resource truncated while hashing')
                digest.update(chunk)
                remaining -= len(chunk)
            if stream.read(1) or _identity(os.fstat(stream.fileno())) != _identity(info):
                raise ValueError('guidance resource changed while hashing')
        if _identity(_admit(path)) != _identity(info) or digest.hexdigest() != expected:
            raise ValueError('guidance resource changed or hash mismatch')
