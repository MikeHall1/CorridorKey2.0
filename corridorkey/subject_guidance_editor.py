"""Native subject clicks on the existing correction owner; optional CPU child."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .clean_plate_editor import _display
from .correction_tracking_context import tracking_decode
from .correction_tracking_editor import MAX_CLEANUP_PENDING, _cleanup, _source_hash
from .corrections import canonical
from .native_image_admission import admit_image
from .preview_sequence import scan_sequence
from .subject_guidance import Click, GuidanceFrame, GuidanceResult
from .subject_guidance_process import SubjectGuidanceProcess

MAX_CLICKS = 128
MAX_PIXELS = 4 * 1024**2


def invalidate_subject_result(editor):
    preview = getattr(editor, 'subject_preview', None)
    if preview is not None:
        preview.update(result=None, inspected=False)
        if preview['state'] == 'predicted':
            preview['state'] = 'prepared'


def discard_subject(editor, *, strict=False):
    preview = getattr(editor, 'subject_preview', None)
    if preview is None:
        return
    try:
        preview['owner'].close()
    except Exception as exc:
        # Keep the owner reachable, including after a successfully committed
        # correction. Cleanup cannot turn a committed edit into a failed edit.
        editor.subject_cleanup_error = f'{type(exc).__name__}: {exc}'[:2048]
        preview['state'] = 'retiring'
        preview['result'] = None
        logging.getLogger(__name__).exception('Subject provider owner cleanup remains pending')
        if strict:
            raise OSError('Subject provider cleanup remains pending; retry Discard before further guidance.') from exc
        return
    editor.subject_preview = None
    editor.subject_cleanup_error = None


def _current(editor):
    preview = getattr(editor, 'subject_preview', None)
    if preview is None or preview['document_identity'] != editor.document.identity:
        raise ValueError('Prepare the subject source again after correction changes.')
    if getattr(editor, 'subject_cleanup_error', None) or preview['state'] == 'retiring':
        raise OSError('Subject provider cleanup remains pending; retry Discard before further guidance.')
    return preview


def subject_summary(editor):
    preview = getattr(editor, 'subject_preview', None)
    if preview is None:
        return None
    result = preview['result']
    return json.loads(canonical({key: preview[key] for key in
        ('frame', 'source_frame', 'state', 'clicks', 'inspected', 'config_path', 'checkpoint_path')} | {
        'result': None if result is None else {'identity': result.identity, 'mask_format': result.mask_format,
                                               'provenance': json.loads(result.provenance_json)},
        'cleanup_error': getattr(editor, 'subject_cleanup_error', None)}))


def prepare_subject(editor, *, frame, config_path, checkpoint_path):
    from .video_io import _read_frame_dir_views

    if not Path(editor.job.input_path).is_dir():
        raise ValueError('Subject guidance currently requires an image sequence; video timing/decode acceptance is unavailable.')
    if type(frame) is not int or not 0 <= frame < editor.binding['frame_count']:
        raise ValueError('Choose a source frame inside the selected correction timeline.')
    height, width = editor.meta['native_hw']
    if height * width > MAX_PIXELS:
        raise ValueError('Subject guidance exceeds the 4 Mi-pixel native bound; no implicit resizing is performed.')
    # Retire the previous process before constructing another owner. Failed
    # preparation cannot silently leave two live segmentation models.
    discard_subject(editor, strict=True)
    index = scan_sequence(editor.job.input_path, check=editor.check)
    absolute = frame + editor.binding['start_frame']
    admitted = admit_image(index.path_at(absolute), expected_hw=(height, width), max_pixels=MAX_PIXELS, check=editor.check)
    if _source_hash(index, editor.check) != editor.binding['sha256']:
        raise ValueError('Source changed; reopen the correction editor before subject guidance.')
    config, _ = editor.job.configurations()
    display, linear = _read_frame_dir_views(str(admitted.path), exr_linear=config.frame_dir_linear,
        alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile,
        encoded_bytes=admitted.encoded_bytes)
    if not np.isfinite(linear).all() or not np.isfinite(display).all():
        raise ValueError('Subject guidance refuses nonfinite native source values.')
    index.check_path(absolute)
    if scan_sequence(editor.job.input_path, check=editor.check).identity != index.identity:
        raise ValueError('Source sequence membership changed during preparation.')
    native_hash = hashlib.sha256(np.ascontiguousarray(linear, dtype='<f4').tobytes()).hexdigest()
    decode = tracking_decode(config, editor.job.input_path)
    proxy_contract = canonical({'contract': 'native-assumed-sRGB-u8-v1', 'native_rgb_f32le_sha256': native_hash,
        'source_decode': decode, 'conversion': 'shared display RGB; clip[0,1]; rint(value*255); uint8; native HWC; no resize'})
    rgb = np.ascontiguousarray(np.rint(np.clip(display, 0, 1) * 255).astype(np.uint8)).tobytes()
    source = GuidanceFrame(editor.binding['sha256'], absolute, proxy_contract, width, height, rgb)
    source.validate(MAX_PIXELS)
    source_display = _display(linear, note='Captured source for optional CPU float32 coarse subject guidance')
    config_file, checkpoint_file = Path(config_path).absolute(), Path(checkpoint_path).absolute()
    owner = SubjectGuidanceProcess(config_file, checkpoint_file, workspace=editor.workspace)
    editor.subject_cleanup_error = None
    editor.subject_preview = {'owner': owner, 'frame': frame, 'source_frame': absolute, 'source': source,
        'source_display': source_display, 'native_hash': native_hash, 'decode': decode, 'index': index,
        'document_identity': editor.document.identity, 'result': None, 'clicks': [], 'inspected': False,
        'state': 'prepared', 'config_path': str(config_file), 'checkpoint_path': str(checkpoint_file)}


def _source_current(editor, preview):
    index = scan_sequence(editor.job.input_path, check=editor.check)
    if index.identity != preview['index'].identity or _source_hash(index, editor.check) != editor.binding['sha256']:
        raise ValueError('Source changed after subject preparation; discard and prepare again.')


def predict_subject(editor, *, clicks):
    preview = _current(editor)
    # Even invalid newer prompts supersede the old acceptable result.
    preview.update(result=None, inspected=False, state='prepared')
    if (not isinstance(clicks, list) or not 1 <= len(clicks) <= MAX_CLICKS
            or any(not isinstance(c, dict) or set(c) != {'x', 'y', 'positive'} for c in clicks)):
        raise ValueError('Supply one to 128 explicit native positive/negative clicks.')
    prompts = tuple(Click(**item) for item in clicks)
    for click in prompts:
        click.validate(preview['source'].width, preview['source'].height)
    if not any(c.positive for c in prompts):
        raise ValueError('Add at least one positive subject click before prediction.')
    # A failed/cancelled newer request must never leave an older acceptable mask.
    preview['clicks'] = json.loads(canonical(clicks))
    _source_current(editor, preview)
    result = preview['owner'].predict(preview['source'], prompts, check=editor.check)
    editor.check()
    if (not isinstance(result, GuidanceResult) or result.frame_identity != preview['source'].identity
            or result.source_sha256 != editor.binding['sha256'] or result.source_frame != preview['source_frame']
            or result.decode_identity != preview['source'].decode_identity
            or (result.height, result.width) != tuple(editor.meta['native_hw']) or result.mask_format != 'binary_u8'
            or type(result.mask_bytes) is not bytes or len(result.mask_bytes) != result.height * result.width
            or any(value not in (0, 255) for value in result.mask_bytes)):
        raise ValueError('Subject result does not match the native binary source request; no approximation is accepted.')
    _source_current(editor, preview)
    # Validate persistence capacity before advertising an inspectable result.
    _origin_settings(preview, result)
    preview.update(result=result, state='predicted')


def _origin_settings(preview, result):
    provenance = json.loads(result.provenance_json)
    if (canonical(provenance.get('clicks')) != canonical(preview['clicks'])
            or provenance.get('mask_sha256') != hashlib.sha256(result.mask_bytes).hexdigest()
            or provenance.get('mask_format') != result.mask_format):
        raise ValueError('Subject result click/mask provenance changed.')
    settings = {'provider_execution': provenance, 'acceptance': 'ordinary Keep; foreground within coarse mask; outside unchanged',
                'conversion': {'operation': 'keep', 'strength': 1., 'range': [preview['frame'], preview['frame'] + 1]},
                'result_identity': result.identity}
    if len(canonical(settings).encode()) > 16 * 1024:
        raise ValueError('Subject provenance exceeds the 16 KiB accepted-guidance settings bound.')
    return settings


def subject_view(editor):
    preview = _current(editor)
    display = preview['source_display']
    result = preview['result']
    overlay = None
    if result is not None:
        native = np.frombuffer(result.mask_bytes, np.uint8).reshape(result.height, result.width)
        overlay = cv2.resize(native, (display.pixels.shape[1], display.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
        preview['inspected'] = True
    return replace(display, pixels=display.pixels.copy()), overlay


def clear_subject(editor):
    preview = _current(editor)
    preview.update(result=None, clicks=[], inspected=False, state='prepared')
    preview['owner'].clear()


def accept_subject(editor):
    preview = _current(editor)
    result = preview['result']
    if result is None or not preview['inspected']:
        raise ValueError('Predict and inspect subject guidance before accepting it as Keep.')
    _source_current(editor, preview)
    for pending in list(editor.tracking_cleanup_pending):
        _cleanup(editor, pending)
    if len(editor.tracking_cleanup_pending) >= MAX_CLEANUP_PENDING:
        raise OSError('Private guidance cleanup remains blocked; inspect warnings before acceptance.')
    ok, encoded = cv2.imencode('.png', np.frombuffer(result.mask_bytes, np.uint8).reshape(result.height, result.width))
    if not ok:
        raise ValueError('Could not encode the exact native coarse subject mask.')
    raw = encoded.tobytes()
    provenance = json.loads(result.provenance_json)
    provider = provenance.get('provider', {})
    weight_hash = provider.get('checkpoint_sha256')
    origin = {'kind': 'subject_clicks', 'resources': [], 'decode': {'source': preview['decode'], 'reference': None},
        'parameters': {'contract': 'corridorkey-guidance-v1',
            'producer': {'id': 'corridorkey.subject_guidance', 'version': '1',
                         'weights_sha256': [weight_hash] if weight_hash else []},
            'settings': _origin_settings(preview, result)},
        'observation': {'source_samples': [{'frame': preview['frame'], 'source_frame': preview['source_frame'],
                                           'rgb_sha256': preview['native_hash']}], 'reference_samples': [],
            'result_asset_sha256': hashlib.sha256(raw).hexdigest(),
            'diagnostics': {'role': 'coarse subject guidance; not alpha matting', 'evidence': provenance.get('evidence'),
                            'quality_verified': False}}}
    path = editor.workspace / ('subject-guidance-' + uuid.uuid4().hex + '.png')
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            stream.write(raw)
        editor._import_raster(path, kind='keep', frame=preview['frame'], strength=1., direction='both',
                              span=[preview['frame'], preview['frame'] + 1], origin=origin)
    finally:
        if created:
            _cleanup(editor, path)
