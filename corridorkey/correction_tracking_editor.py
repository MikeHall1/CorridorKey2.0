"""Explicit sequence tracking preview/publication on the existing editor owner."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .control import JobControl
from .correction_rasters import CorrectionRasters
from .correction_tracking import (
    KERNEL_VERSION,
    TrackingAnchor,
    TrackingFrame,
    TrackingRequest,
    track_corrections,
)
from .correction_tracking_context import tracking_cuts, tracking_decode
from .corrections import CorrectionDocument, canonical, track_binding
from .identities import file_sha256

MAX_PREVIEW_BYTES = 512 * 1024 * 1024
MAX_CLEANUP_PENDING = 8
MAX_DIAGNOSTIC_BYTES = 64 * 1024 * 1024


def _cleanup(editor, directory):
    pending = getattr(editor, 'tracking_cleanup_pending', [])
    try:
        if directory.is_dir():
            shutil.rmtree(directory)
        else:
            directory.unlink()
    except FileNotFoundError:
        pending = [path for path in pending if path != directory]
    except OSError as exc:
        if directory not in pending:
            pending.append(directory)
        editor.tracking_cleanup_warnings = [*getattr(editor, 'tracking_cleanup_warnings', []), str(exc)[:1024]][-8:]
    else:
        pending = [path for path in pending if path != directory]
    editor.tracking_cleanup_pending = pending


def discard_tracking(editor):
    preview = getattr(editor, 'tracking_preview', None)
    editor.tracking_preview = None
    if preview is not None:
        # Only our unique unpublished preview directory; accepted assets stay.
        _cleanup(editor, preview['directory'])


def _source_hash(index, check):
    rows = []
    for frame in range(len(index.paths)):
        check()
        path = index.path_at(frame)
        rows.append({'name': Path(path).name, 'sha256': file_sha256(path, check=check)})
        index.check_path(frame)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def prepare_tracking(editor, *, anchor_ids, span, direction, unreliable_policy='request_anchor',
                     capture_diagnostics=False):
    from .preview_sequence import scan_sequence
    from .video_io import _read_frame_dir_views

    if type(capture_diagnostics) is not bool:
        raise TypeError('Diagnostic capture requires an explicit boolean.')
    config, _ = editor.job.configurations()
    for directory in list(getattr(editor, 'tracking_cleanup_pending', [])):
        _cleanup(editor, directory)
    if len(getattr(editor, 'tracking_cleanup_pending', [])) >= MAX_CLEANUP_PENDING:
        raise OSError('Tracking private cleanup remains blocked; inspect cleanup warnings before preparing more previews.')
    if not Path(editor.job.input_path).is_dir():
        raise ValueError('Tracking currently requires a local image sequence; video timing/reader tracking is unavailable.')
    data = editor.document.to_dict()
    if (not isinstance(anchor_ids, list) or not anchor_ids or len(anchor_ids) > 32
            or any(not isinstance(key, str) for key in anchor_ids) or len(set(anchor_ids)) != len(anchor_ids)):
        raise ValueError('Select one to 32 unique authored coverage anchors.')
    ids = set(anchor_ids)
    entries = [entry for entry in data['entries'] if entry['id'] in ids]
    if len(entries) != len(ids) or any(entry['kind'] not in {'keep', 'remove', 'matte'} for entry in entries):
        raise ValueError('Tracking anchors must be existing matte/keep/remove entries, not trimaps or resets.')
    if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
            or not 0 <= span[0] < span[1] <= data['source']['frame_count']):
        raise ValueError('Tracking range must lie inside the selected correction timeline.')
    cuts = tracking_cuts(config)
    barriers = sorted(set(cuts) | set(editor.document.resets) | {shot[0] for shot in data['shots']})
    if any(span[0] < point < span[1] for point in barriers):
        raise ValueError('Tracking range crosses a cut/reset/shot boundary; narrow the range and add an artist anchor.')
    request = TrackingRequest(data['source']['sha256'], *span, source_start=data['source']['start_frame'],
                              direction=direction, cuts=tuple(p for p in cuts if span[0] <= p <= span[1]),
                              resets=tuple(p for p in editor.document.resets if span[0] <= p <= span[1]),
                              unreliable_policy=unreliable_policy,
                              source_decode_identity=hashlib.sha256(canonical(tracking_decode(config, editor.job.input_path)).encode()).hexdigest())
    request.validate()
    if data['source']['height'] * data['source']['width'] > request.max_pixels:
        raise ValueError('Tracking exceeds the 4 Mi-pixel native canvas bound; no scaling is performed.')
    if len(entries) * data['source']['height'] * data['source']['width'] * 4 > request.max_anchor_bytes:
        raise ValueError('Artist anchors exceed the 64 MiB tracking byte bound; use fewer anchors or a smaller native canvas.')
    estimator_identity = f'opencv-{cv2.__version__}-farneback-v1'
    settings = {key: value for key, value in asdict(request).items() if key in {
        'direction', 'unreliable_policy', 'max_uncertain_fraction', 'motion', 'max_frames', 'max_pixels', 'max_anchor_bytes', 'max_anchors'}}
    settings.update(kernel_version=KERNEL_VERSION, estimator_identity=estimator_identity)
    binding = track_binding(data, [entry['id'] for entry in entries], cuts=cuts,
                            decode=tracking_decode(config, editor.job.input_path), tracking=settings)
    index = scan_sequence(editor.job.input_path, check=editor.check)
    if _source_hash(index, editor.check) != editor.binding['sha256']:
        raise ValueError('Source changed since editor binding; reopen before tracking.')
    anchors = []
    for entry in entries:
        # Reuse coverage admission/precision without mixing existing track output.
        isolated = {**entry, 'kind': 'matte', 'strength': 1., 'direction': 'both'}
        isolated.pop('guidance_id', None)
        isolated_data = {**data, 'entries': [isolated]}
        if isolated_data['schema_version'] == 4:
            isolated_data['guidance'] = []
        if isolated_data['schema_version'] >= 3:
            isolated_data['tracks'] = []
        rasters = CorrectionRasters(CorrectionDocument.parse(isolated_data, editor.document.base_dir), editor.meta, check=editor.check)
        try:
            mask = rasters.apply(np.zeros(editor.meta['native_hw'], np.float32), entry['frame'])
        finally:
            rasters.close()
        anchors.append(TrackingAnchor.create(entry['id'], entry['revision'], entry['frame'], entry['kind'], entry['strength'], mask))

    def read(absolute):
        path = index.path_at(absolute)
        proxy, _ = _read_frame_dir_views(path, exr_linear=config.frame_dir_linear,
                                        alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile)
        index.check_path(absolute)
        return TrackingFrame(editor.binding['sha256'], absolute, proxy)

    directory = editor.workspace / ('tracking-preview-' + uuid.uuid4().hex)
    directory.mkdir(exist_ok=False)
    rows, summaries, used = [], [], 0
    diagnostic_receipts, diagnostic_used = {}, 0
    diagnostic_directory = directory / 'diagnostics'
    try:
        if capture_diagnostics:
            diagnostic_directory.mkdir(exist_ok=False)
        for candidate in track_corrections(request, tuple(anchors), read, control=JobControl(check_hook=editor.check)):
            editor.check()
            provenance = json.loads(candidate.provenance_json)
            summaries.append({'frame': candidate.local_frame, 'source_frame': candidate.source_frame,
                              'status': candidate.status, 'candidate_identity': candidate.identity,
                              'source_proxy_sha256': provenance['source_proxy_sha256'],
                              'derived_mask_sha256': provenance['derived_mask_sha256'],
                              'diagnostics': json.loads(candidate.diagnostics_json)})
            if capture_diagnostics:
                from .tracking_overlays import build_tracking_overlays, write_bundle
                bundle = build_tracking_overlays(candidate, document_identity=editor.document.identity,
                                                 decode_identity=request.source_decode_identity, check=editor.check)
                if bundle is not None:
                    receipt = write_bundle(bundle, diagnostic_directory,
                                           remaining_budget=MAX_DIAGNOSTIC_BYTES - diagnostic_used, check=editor.check)
                    diagnostic_used += receipt.bytes_charged
                    diagnostic_receipts[candidate.local_frame] = receipt
            mask = candidate.mask()
            if mask is not None:
                used += mask.nbytes
                if used > MAX_PREVIEW_BYTES:
                    raise ValueError('Tracking preview exceeds the 512 MiB logical mask/encoded-byte budget; shorten the range.')
                ok, encoded = cv2.imencode('.exr', mask)
                if not ok:
                    raise ValueError('Could not encode float32 tracked coverage.')
                payload = encoded.tobytes()
                used += len(payload)
                if used > MAX_PREVIEW_BYTES:
                    raise ValueError('Tracking preview exceeds the 512 MiB logical mask/encoded-byte budget; shorten the range.')
                target = directory / f'{candidate.local_frame:09d}.exr'
                with target.open('xb') as stream:
                    stream.write(payload)
                rows.append({'frame': candidate.local_frame, 'asset': {'path': str(target), 'sha256': hashlib.sha256(payload).hexdigest()}})
        # Re-hash exact indexed files and re-index once to detect additions/removal.
        if (_source_hash(index, editor.check) != editor.binding['sha256']
                or scan_sequence(editor.job.input_path, check=editor.check).identity != index.identity):
            raise ValueError('Source changed during tracking; preview discarded.')
        complete = len(rows) == span[1] - span[0]
        track = {'id': uuid.uuid4().hex, 'owner_id': entries[0]['id'], 'anchor_ids': [entry['id'] for entry in entries],
                 'range': span.copy(), 'direction': direction, 'binding': binding,
                 'binding_sha256': hashlib.sha256(canonical(binding).encode()).hexdigest(),
                 'frames': sorted(rows, key=lambda row: row['frame']),
                 'observations': sorted(summaries, key=lambda row: row['frame'])}
        if complete:
            candidate_data = {**data, 'schema_version': max(3, data['schema_version']), 'tracks': [*data.get('tracks', []), track]}
            CorrectionDocument.parse(candidate_data, editor.document.base_dir)
        old = getattr(editor, 'tracking_preview', None)
        editor.tracking_preview = {'directory': directory, 'document_identity': editor.document.identity,
                                   'track': track, 'complete': complete, 'frames': summaries, 'bytes_charged': used,
                                   'inspected': False,
                                   'diagnostic_receipts': diagnostic_receipts,
                                   'diagnostic_directory': diagnostic_directory,
                                   'diagnostics': {
                                       'captured': capture_diagnostics, 'bytes_charged': diagnostic_used,
                                       'budget_bytes': MAX_DIAGNOSTIC_BYTES,
                                       'mask_budget_bytes': MAX_PREVIEW_BYTES,
                                       'available_frames': sorted(diagnostic_receipts),
                                       'kinds': ['reliability', 'uncertainty'] if diagnostic_receipts else [],
                                       'frame_events': [{'frame': item['frame'], 'source_frame': item['source_frame'],
                                                         'status': item['status'], 'available': item['frame'] in diagnostic_receipts,
                                                         'reason': str(item['diagnostics'].get('reason', ''))[:96]}
                                                        for item in summaries] if capture_diagnostics else [],
                                       'legend': ('Red opacity shows low heuristic reliability or the uncertain fraction in a proxy cell. '
                                                  'Flow/photometric heuristics are not calibrated confidence or proof of occlusion.'),
                                       'captured_request': {
                                           'document_identity': editor.document.identity,
                                           'source_identity': editor.binding['sha256'],
                                           'source_path': editor.job.input_path,
                                           'source_binding': data['source'],
                                           'decode_identity': request.source_decode_identity,
                                           'source_start': data['source']['start_frame'],
                                           'range': span.copy(), 'direction': direction},
                                       'availability': ('Captured maps are available only for frames where motion was computed.'
                                                        if capture_diagnostics else 'Not captured. Enable diagnostics before preparing a new tracking preview.')
                                   }}
    except BaseException:
        _cleanup(editor, directory)
        raise
    if old is not None:
        _cleanup(editor, old['directory'])


def accept_tracking(editor):
    preview = getattr(editor, 'tracking_preview', None)
    if preview is None or not preview['complete'] or not preview['inspected']:
        raise ValueError('A complete inspected tracking preview is required; narrow the range or add another anchor.')
    if editor.document.identity != preview['document_identity']:
        raise ValueError('Tracking preview is stale after an authored edit.')
    from .preview_sequence import scan_sequence
    if _source_hash(scan_sequence(editor.job.input_path, check=editor.check), editor.check) != editor.binding['sha256']:
        raise ValueError('Source changed after tracking; preview cannot be accepted.')
    data = editor.document.to_dict()
    data.update(schema_version=max(3, data['schema_version']), tracks=[*data.get('tracks', []), preview['track']])
    candidate = CorrectionDocument.parse(data, editor.document.base_dir)
    candidate.verify_assets(check=editor.check)
    editor._apply(data, preserve_tracks=True)
    editor.tracking_preview = None  # The accepted document now owns immutable assets.
    if preview.get('diagnostics', {}).get('captured'):
        _cleanup(editor, preview['diagnostic_directory'])


def tracking_summary(editor):
    preview = getattr(editor, 'tracking_preview', None)
    if preview is None:
        return None
    summary = {key: preview[key] for key in ('complete', 'frames', 'bytes_charged', 'document_identity', 'inspected')}
    track = preview['track']
    summary['diagnostics'] = preview.get('diagnostics', {
        'captured': False, 'available_frames': [], 'frame_events': [], 'kinds': [],
        'availability': 'Diagnostic maps were not captured for this preview; saved tracks have no map backfill.'})
    summary['request'] = {'range': track['range'], 'direction': track['direction'], 'anchor_ids': track['anchor_ids'],
                          'unreliable_policy': track['binding']['tracking']['unreliable_policy'],
                          'capture_diagnostics': summary['diagnostics']['captured']}
    return json.loads(canonical(summary))  # No cross-thread aliases to owned observations.


def tracking_view(editor, frame):
    preview = getattr(editor, 'tracking_preview', None)
    if preview is None or preview['document_identity'] != editor.document.identity:
        raise ValueError('No current tracking preview; prepare it again.')
    row = next((row for row in preview['track']['frames'] if row['frame'] == frame), None)
    if row is None:
        raise ValueError('This frame needs another anchor; there is no retained mask to inspect.')
    if file_sha256(row['asset']['path'], check=editor.check) != row['asset']['sha256']:
        raise ValueError('Tracking preview mask changed; prepare it again.')
    mask = cv2.imread(row['asset']['path'], cv2.IMREAD_UNCHANGED)
    if mask is None or mask.dtype != np.float32 or mask.shape != editor.meta['native_hw']:
        raise ValueError('Tracking preview mask no longer matches the native canvas.')
    display, _ = editor.view(frame)
    overlay = np.rint(cv2.resize(mask, (display.pixels.shape[1], display.pixels.shape[0]), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    preview['inspected'] = True
    return display, overlay


def tracking_overlay_view(editor, frame, kind, opacity=.55):
    """Inspect captured heuristic maps on a source copy, without accepting coverage."""
    from .desktop_preview import PreviewFrame
    from .preview_sequence import scan_sequence
    from .tracking_overlays import OverlayBinding, read_bundle, render_layer
    from .video_io import _read_frame_dir_views

    editor.check()
    if (type(frame) is not int or kind not in ('reliability', 'uncertainty')
            or type(opacity) not in (int, float) or not math.isfinite(opacity) or not 0 <= opacity <= 1):
        raise ValueError('Select a local tracking frame and an available diagnostic kind.')
    preview = getattr(editor, 'tracking_preview', None)
    if preview is None or preview['document_identity'] != editor.document.identity:
        raise ValueError('No current diagnostic preview. Saved tracks have no captured maps; prepare a new preview explicitly.')
    diagnostics = preview.get('diagnostics', {})
    receipt = preview.get('diagnostic_receipts', {}).get(frame)
    if not diagnostics.get('captured') or receipt is None:
        raise ValueError('Diagnostic maps are unavailable for this frame; authored anchors and uncomputed/stopped intervals have no fabricated maps.')
    config, _ = editor.job.configurations()
    decode_identity = hashlib.sha256(canonical(tracking_decode(config, editor.job.input_path)).encode()).hexdigest()
    captured = diagnostics['captured_request']
    data = editor.document.to_dict()
    track = preview['track']
    current_binding = track_binding(data, track['anchor_ids'], cuts=tracking_cuts(config),
                                    decode=tracking_decode(config, editor.job.input_path), tracking=track['binding']['tracking'])
    if (current_binding != track['binding'] or editor.job.input_path != captured['source_path']
            or decode_identity != captured['decode_identity']
            or data['source'] != captured['source_binding']
            or editor.binding != captured['source_binding']):
        raise ValueError('Tracking diagnostic source/decode binding changed; prepare a new preview.')
    row = next((item for item in preview['frames'] if item['frame'] == frame), None)
    if row is None:
        raise ValueError('No captured tracking event exists for this frame.')
    mask_row = next((item for item in preview['track']['frames'] if item['frame'] == frame), None)
    if mask_row is not None and file_sha256(mask_row['asset']['path'], check=editor.check) != mask_row['asset']['sha256']:
        raise ValueError('Tracking preview mask changed; prepare diagnostics again.')
    index = scan_sequence(editor.job.input_path, check=editor.check)
    if _source_hash(index, editor.check) != captured['source_identity']:
        raise ValueError('Source changed since diagnostic capture; prepare a new preview.')
    absolute = captured['source_start'] + frame
    if row['source_frame'] != absolute:
        raise ValueError('Tracking diagnostic local/source frame mapping changed.')
    proxy, _ = _read_frame_dir_views(index.path_at(absolute), exr_linear=config.frame_dir_linear,
                                    alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile)
    index.check_path(absolute)
    if proxy.shape != (*editor.meta['native_hw'], 3):
        raise ValueError('Tracking diagnostic source proxy geometry changed.')
    proxy_hash = hashlib.sha256(proxy.astype('<f4', copy=False).tobytes()).hexdigest()
    if row['source_proxy_sha256'].get(str(frame)) != proxy_hash:
        raise ValueError('Tracking diagnostic source proxy changed; prepare a new preview.')
    expected = OverlayBinding(source_identity=captured['source_identity'], document_identity=editor.document.identity,
                              candidate_identity=row['candidate_identity'], decode_identity=decode_identity,
                              source_frame=absolute, local_frame=frame,
                              native_width=editor.meta['native_hw'][1], native_height=editor.meta['native_hw'][0],
                              source_proxy_sha256=proxy_hash)
    bundle = read_bundle(receipt, preview['diagnostic_directory'], expected_binding=expected, check=editor.check)
    layer = render_layer(bundle, kind, opacity=opacity)
    display, _ = editor.view(frame)
    rgba = np.frombuffer(layer.pixels, np.uint8).reshape(layer.height, layer.width, 4)
    rgba = cv2.resize(rgba, (display.pixels.shape[1], display.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.
    pixels = np.rint(display.pixels.astype(np.float32) * (1 - alpha) + rgba[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    pixels.setflags(write=False)
    if (_source_hash(index, editor.check) != captured['source_identity']
            or scan_sequence(editor.job.input_path, check=editor.check).identity != index.identity):
        raise ValueError('Source changed while inspecting diagnostic maps; view discarded.')
    editor.check()
    note = (f'{kind.title()} heuristic · local frame {frame} / source frame {absolute} · '
            f'map proxy {layer.width}×{layer.height} · not calibrated confidence or occlusion proof. {display.note}')
    return PreviewFrame(pixels, display.count, display.width, display.height, display.fps, note), None
