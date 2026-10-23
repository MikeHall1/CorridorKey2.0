"""One-frame clean-plate inspection on the existing correction editor owner."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .clean_plate import (
    INPUT_CONTRACT,
    KERNEL_VERSION,
    CleanPlateProvenance,
    CleanPlateSettings,
    PlateMapping,
    assist_clean_plate,
)
from .correction_tracking_context import tracking_decode
from .correction_tracking_editor import MAX_CLEANUP_PENDING, _cleanup, _source_hash
from .corrections import canonical
from .desktop_preview import PreviewFrame
from .native_image_admission import admit_image
from .preview_sequence import scan_sequence


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _display(linear, *, count=1, note):
    from .color_utils import linear_to_srgb
    height, width = linear.shape[:2]
    rgb = np.clip(linear_to_srgb(np.array(linear, dtype=np.float32, copy=True)), 0, 1)
    if max(height, width) > 512:
        scale = 512 / max(height, width)
        rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    pixels = np.ascontiguousarray(np.rint(rgb * 255).astype(np.uint8))
    return PreviewFrame(pixels, count, width, height, None, note + '; sRGB thumbnail; signed/HDR values clip only for display')


def discard_plate(editor):
    preview = getattr(editor, 'plate_preview', None)
    editor.plate_preview = None
    if preview is not None:
        _cleanup(editor, preview['directory'])


def plate_summary(editor):
    preview = getattr(editor, 'plate_preview', None)
    if preview is None:
        return None
    result = {key: preview[key] for key in ('frame', 'source_frame', 'reference_frame', 'reference_path', 'request', 'inspected')}
    result.update(usable=preview['candidate'].usable, report=preview['candidate'].report())
    return json.loads(canonical(result))


def prepare_plate(editor, *, frame, reference_path, reference_frame=0, reference_mode: Literal['single', 'sequence']='single',
                  reference_profile='srgb', alignment: Literal['declared', 'estimate_translation']='declared', translation_xy=(0., 0.),
                  compensate_exposure=False, samples_path=None, whole_frame_samples=False,
                  background_threshold=.04, foreground_threshold=.35):
    from .trimap_refinement import load_trimap_asset
    from .video_io import _read_frame_dir_views
    config, _ = editor.job.configurations()
    if not Path(editor.job.input_path).is_dir():
        raise ValueError('Clean-plate assistance currently requires a source image sequence; video timing/decode acceptance is unavailable.')
    if type(frame) is not int or not 0 <= frame < editor.binding['frame_count']:
        raise ValueError('Choose a source frame inside the selected correction timeline.')
    if type(whole_frame_samples) is not bool or (samples_path is not None and whole_frame_samples):
        raise ValueError('Choose a binary background sample mask or explicitly declare the whole image as samples.')
    if samples_path is None and not whole_frame_samples:
        raise ValueError('Select native binary background samples, or explicitly choose the whole-image background assumption.')
    for directory in list(editor.tracking_cleanup_pending):
        _cleanup(editor, directory)
    if len(editor.tracking_cleanup_pending) >= MAX_CLEANUP_PENDING:
        raise OSError('Private preview cleanup is blocked; inspect cleanup warnings before preparing more guidance.')
    settings = CleanPlateSettings(alignment=alignment, translation_xy=translation_xy,
                                  compensate_exposure=compensate_exposure,
                                  background_threshold=background_threshold, foreground_threshold=foreground_threshold)
    settings.validate()
    height, width = editor.meta['native_hw']
    if height * width > settings.max_pixels or height * width * 320 + min(height, settings.proxy_max_side) * min(width, settings.proxy_max_side) * 24 > settings.max_working_bytes:
        raise ValueError('Clean-plate native pixel/estimated working-byte bound exceeded; no implicit scaling is performed.')
    source_index = scan_sequence(editor.job.input_path, check=editor.check)
    source_frame = frame + editor.binding['start_frame']
    path = source_index.path_at(source_frame)
    source_admitted = admit_image(path, expected_hw=(height, width), max_pixels=settings.max_pixels, check=editor.check)
    if _source_hash(source_index, editor.check) != editor.binding['sha256']:
        raise ValueError('Source changed; reopen the editor before clean-plate assistance.')
    _, source_rgb = _read_frame_dir_views(path, exr_linear=config.frame_dir_linear,
                                         alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile,
                                         encoded_bytes=source_admitted.encoded_bytes)
    source_index.check_path(source_frame)
    if tuple(source_rgb.shape[:2]) != (height, width):
        raise ValueError('Source frame geometry changed after editor binding.')
    reference = Path(reference_path).absolute()
    reference = reference.parent.resolve() / reference.name
    if reference_profile not in {'srgb', 'linear_bt709_d65', 'linear_bt2020_d65'}:
        raise ValueError('Choose explicit sRGB or scene-linear reference primaries.')
    reference_config = replace(config, frame_dir_linear=reference_profile != 'srgb', frame_dir_alpha='ignore',
                               source_colour_profile='legacy' if reference_profile == 'srgb' else reference_profile,
                               linear_brightness=1., linear_contrast=1., linear_contrast_pivot=.18)
    reference_decode = tracking_decode(reference_config, reference)
    reference_decode['source_kind'] = 'frames'  # A selected still uses the native image decoder too.
    if reference.is_dir():
        reference_index = scan_sequence(reference, check=editor.check)
        reference_count = len(reference_index.paths)
    else:
        if reference.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.exr'} or not reference.is_file():
            raise ValueError('Select a local PNG/JPEG/EXR plate or an image-sequence directory.')
        reference_index, reference_count = None, 1
    mapping = PlateMapping(reference_mode, source_frame, 1, reference_frame, reference_count)
    mapping.validate()
    actual_reference_frame = mapping.reference_frame(source_frame)
    selected_reference = reference_index.path_at(actual_reference_frame) if reference_index is not None else str(reference)
    reference_admitted = admit_image(selected_reference, expected_hw=(height, width), max_pixels=settings.max_pixels, check=editor.check)
    reference_hash = reference_admitted.sha256
    _, reference_rgb = _read_frame_dir_views(selected_reference, exr_linear=reference_config.frame_dir_linear,
                                            alpha_mode='ignore', source_colour_profile=reference_config.source_colour_profile,
                                            encoded_bytes=reference_admitted.encoded_bytes)
    if admit_image(selected_reference, expected_hw=(height, width), max_pixels=settings.max_pixels, check=editor.check).sha256 != reference_hash:
        raise ValueError('Selected clean plate changed during decoding.')
    if reference_index is not None:
        reference_index.check_path(actual_reference_frame)
    resources = [{'id': 'plate', 'role': 'reference', 'asset': {'path': selected_reference, 'sha256': reference_hash}}]
    if samples_path is not None:
        sample_path = Path(samples_path).absolute()
        sample_path = sample_path.parent.resolve() / sample_path.name
        sample_admitted = admit_image(sample_path, expected_hw=(height, width), max_pixels=settings.max_pixels, check=editor.check)
        if sample_admitted.channels != 1:
            raise ValueError('Background sample masks must be native grayscale binary images: white selected, black excluded.')
        sample_hash = sample_admitted.sha256
        samples = load_trimap_asset(sample_path, expected_hw=(height, width), expected_sha256=sample_hash, check=editor.check).labels
        if np.any(samples == 128):
            raise ValueError('Background samples must be binary: white selected, black excluded; unknown trimap classes are not sampling weights.')
        samples = (samples == 255).astype(np.uint8)
        resources.append({'id': 'samples', 'role': 'source_sample', 'asset': {'path': str(sample_path), 'sha256': sample_hash}})
    else:
        samples = np.ones((height, width), np.uint8)
    decode = {'source': tracking_decode(config, editor.job.input_path), 'reference': reference_decode}
    provenance = CleanPlateProvenance(editor.binding['sha256'], reference_hash, _hash(decode['source']),
                                      _hash(reference_decode), source_frame, actual_reference_frame, mapping, INPUT_CONTRACT)
    candidate = assist_clean_plate(source_rgb, reference_rgb, samples, settings=settings, provenance=provenance, check=editor.check)
    if (_source_hash(source_index, editor.check) != editor.binding['sha256']
            or scan_sequence(editor.job.input_path, check=editor.check).identity != source_index.identity):
        raise ValueError('Source changed during clean-plate analysis; candidate discarded.')
    for resource in resources:
        if admit_image(resource['asset']['path'], expected_hw=(height, width), max_pixels=settings.max_pixels, check=editor.check).sha256 != resource['asset']['sha256']:
            raise ValueError('Plate or sample reference changed during analysis; candidate discarded.')
    reference_display = _display(reference_rgb, count=reference_count,
                                 note=f'Clean plate reference frame {actual_reference_frame}; explicit {reference_profile}')
    directory = editor.workspace / ('plate-preview-' + uuid.uuid4().hex)
    directory.mkdir(exist_ok=False)
    old = getattr(editor, 'plate_preview', None)
    editor.plate_preview = {'frame': frame, 'source_frame': source_frame, 'reference_frame': actual_reference_frame,
                            'reference_path': selected_reference, 'directory': directory, 'candidate': candidate,
                            'document_identity': editor.document.identity, 'resources': resources, 'decode': decode,
                            'reference_display': reference_display, 'inspected': False,
                            'request': json.loads(canonical({'settings': asdict(settings), 'mapping': asdict(mapping),
                                        'reference_profile': reference_profile, 'whole_frame_samples': whole_frame_samples,
                                        'sample_mask_path': str(samples_path) if samples_path is not None else None}))}
    if old is not None:
        _cleanup(editor, old['directory'])


def _current(editor):
    preview = getattr(editor, 'plate_preview', None)
    if preview is None or preview['document_identity'] != editor.document.identity:
        raise ValueError('No current clean-plate preview; prepare it again after authored changes.')
    return preview


def plate_view(editor, mode='guidance'):
    preview = _current(editor)
    candidate = preview['candidate']
    if mode == 'guidance':
        display, _ = editor.view(preview['frame'])
        overlay = cv2.resize(candidate.labels(), (display.pixels.shape[1], display.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
        preview['inspected'] = True
        return display, overlay
    if mode == 'screen_field':
        return _display(candidate.screen_field(), note='Spatial screen field in working linear BT.709; no render pixels changed'), None
    if mode == 'residual':
        ceiling = preview['request']['settings']['foreground_threshold']
        gray = np.clip(candidate.residual() / ceiling, 0, 1)
        # Residual is an explicitly scaled display score, not scene radiance.
        display = _display(np.repeat(gray[..., None], 3, axis=2), note=f'Relative RGB residual, black=0 white>={ceiling:g}; uncalibrated heuristic')
        return display, None
    if mode == 'reference':
        return editor.plate_reference(), None
    raise ValueError('Choose guidance, screen field, residual or reference inspection.')


def accept_plate(editor):
    preview = _current(editor)
    candidate = preview['candidate']
    if not preview['inspected'] or not candidate.usable:
        raise ValueError('Inspect a usable rough-guidance preview before acceptance; unusable results cannot guide a render.')
    if preview.get('staging_cleanup_blocked'):
        raise OSError('Private guidance staging cleanup is blocked; Discard preview before another acceptance attempt.')
    source_index = scan_sequence(editor.job.input_path, check=editor.check)
    if _source_hash(source_index, editor.check) != editor.binding['sha256']:
        raise ValueError('Source changed after clean-plate preview; acceptance refused.')
    report = candidate.report()
    hashes = report['provenance']['input_hashes']
    ok, encoded = cv2.imencode('.png', candidate.labels())
    if not ok:
        raise ValueError('Could not encode exact native rough trimap classes.')
    path = preview['directory'] / ('guidance-' + uuid.uuid4().hex + '.png')
    raw = encoded.tobytes()
    result_hash = hashlib.sha256(raw).hexdigest()
    origin = {'kind': 'clean_plate', 'resources': preview['resources'], 'decode': preview['decode'],
              'parameters': {'contract': 'corridorkey-guidance-v1',
                             'producer': {'id': 'corridorkey.clean_plate', 'version': KERNEL_VERSION, 'weights_sha256': []},
                             'settings': {**{key: value for key, value in preview['request'].items() if key != 'sample_mask_path'},
                                          'candidate_identity': candidate.identity}},
              'observation': {'source_samples': [{'frame': preview['frame'], 'source_frame': preview['source_frame'],
                                                   'rgb_sha256': hashes['source_rgb_f32le']}],
                              'reference_samples': [{'resource_id': 'plate', 'frame': preview['reference_frame'],
                                                     'rgb_sha256': hashes['reference_rgb_f32le']}],
                              'result_asset_sha256': result_hash, 'diagnostics': report['diagnostics']}}
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            stream.write(raw)
        editor._import_raster(path, kind='trimap', frame=preview['frame'], strength=1., direction='both',
                              span=[preview['frame'], preview['frame'] + 1], origin=origin)
    finally:
        try:
            if created:
                path.unlink(missing_ok=True)  # Only the file this call created.
        except OSError as exc:
            # Never replace an import failure or misreport an already committed
            # edit. The preview or pending directory still owns this file.
            preview['staging_cleanup_blocked'] = True
            editor.tracking_cleanup_warnings = [*editor.tracking_cleanup_warnings, str(exc)[:1024]][-8:]
            if preview['directory'] not in editor.tracking_cleanup_pending:
                editor.tracking_cleanup_pending.append(preview['directory'])
