"""Native correction authoring with shared metadata, history and raster validation."""
from __future__ import annotations

import uuid
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .correction_rasters import CorrectionRasters
from .corrections import (
    REGION_TARGETS,
    CorrectionDocument,
    CorrectionHistory,
    load_corrections,
    save_corrections,
)
from .desktop_preview import PreviewFrame, preview
from .identities import file_sha256, source_identity
from .preview_sequence import scan_sequence
from .project_collection import admit_file, assert_pin
from .session import JobRequest

MAX_EDITOR_PIXELS = 64 * 1024 * 1024


class CorrectionEditor:
    """Single-owner editor model. Workspace/assets/saves are immutable publications."""

    def __init__(self, job: JobRequest, parent: Path, *, existing: Path | None = None, check=None):
        from .video_io import probe_frame_dir_meta, probe_video_meta
        self.job, self.check = job, check or (lambda: None)
        config, _ = job.configurations()
        job.validate_source(config)
        source = Path(job.input_path)
        parent = parent.resolve()
        if not parent.is_dir() or parent == source or (source.is_dir() and parent.is_relative_to(source)):
            raise ValueError('Choose an existing correction-project parent outside the source.')
        before = source_identity(source, check=self.check)['sha256']
        if source.is_dir():
            self.meta = probe_frame_dir_meta(str(source), num_frames=config.num_frames, start_frame=config.start_frame, shape_quantum=config.shape_quantum,
                         fps=config.frame_dir_fps, exr_linear=config.frame_dir_linear, model_scale=config.model_scale,
                         alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile)
        else:
            self.meta = probe_video_meta(str(source), num_frames=config.num_frames, start_frame=config.start_frame, shape_quantum=config.shape_quantum,
                         native_resolution=True, start_mode=config.start_mode, seed=config.seed, model_scale=config.model_scale,
                         linear_brightness=config.linear_brightness, linear_contrast=config.linear_contrast,
                         linear_contrast_pivot=config.linear_contrast_pivot)
        after = source_identity(source, check=self.check)['sha256']
        if before != after:
            raise ValueError('Source changed during editor binding.')
        height, width = self.meta['native_hw']
        if height * width > MAX_EDITOR_PIXELS:
            raise ValueError('Correction editor native canvas exceeds the 64Mi-pixel limit; use core raster documents for larger sources.')
        self.binding = {'sha256': before, 'start_frame': self.meta['start_frame'], 'frame_count': self.meta['num_frames'],
                        'fps': self.meta['native_fps'], 'width': width, 'height': height}
        if existing is not None:
            document = load_corrections(existing, source=source, check=self.check)
            if document.to_dict()['source'] != self.binding:
                raise ValueError('Correction document selected timeline/geometry differs from this job.')
            rasters = CorrectionRasters(document, self.meta, check=self.check)
            rasters.close()
        else:
            document = CorrectionDocument.parse({'schema_version': 1, 'revision': 0, 'source': self.binding,
                         'shots': [[0, self.binding['frame_count']]], 'entries': []}, parent)
        self.tracking_preview = None
        self.plate_preview = None
        self.subject_preview = None
        self.subject_cleanup_error = None
        self.tracking_cleanup_pending: list[Path] = []
        self.tracking_cleanup_warnings: list[str] = []
        self.history = CorrectionHistory(document, max_snapshots=128)
        self.revision = document.to_dict()['revision']
        self.invalidated: list[tuple[int, int]] = []
        self.saved_path: Path | None = None
        self.checkpoint_path: Path | None = None
        self.workspace = parent / f'corrections-{uuid.uuid4().hex}'
        self.workspace.mkdir(exist_ok=False)
        (self.workspace / 'assets').mkdir()
        self._checkpoint(document)

    @property
    def document(self) -> CorrectionDocument:
        return self.history.current

    def snapshot(self) -> dict[str, Any]:
        return {'document': self.document.to_dict(), 'identity': self.document.identity,
                'invalidated': self.invalidated.copy(), 'workspace': str(self.workspace),
                'checkpoint_path': str(self.checkpoint_path),
                'saved_path': str(self.saved_path) if self.saved_path else None,
                'tracking_preview': self.tracking_summary(),
                'plate_preview': self.plate_summary(),
                'subject_preview': self.subject_summary(),
                'subject_cleanup_error': self.subject_cleanup_error,
                'tracking_cleanup_pending': [str(path) for path in self.tracking_cleanup_pending],
                'tracking_cleanup_warnings': self.tracking_cleanup_warnings.copy()}

    def _apply(self, data: dict[str, Any], *, preserve_tracks: bool = False) -> None:
        # Authored edits conservatively discard all derivatives; undo retains
        # the exact earlier accepted snapshot and immutable mask assets.
        if data.get('schema_version', 1) >= 3 and not preserve_tracks:
            data['tracks'] = []
        if data.get('schema_version', 1) >= 4:
            present = {entry['id'] for entry in data['entries']}
            data['guidance'] = [origin for origin in data['guidance'] if origin['entry_id'] in present]
        data['revision'] = self.revision + 1
        document = CorrectionDocument.parse(data, self.document.base_dir)
        self._checkpoint(document)
        self.invalidated = self.history.apply(document)
        self.revision += 1
        self.saved_path = None
        if not preserve_tracks:
            self.discard_tracking()
        self.discard_plate()
        self.discard_subject()

    def prepare_tracking(self, **parameters):
        from .correction_tracking_editor import prepare_tracking
        prepare_tracking(self, **parameters)

    def accept_tracking(self):
        from .correction_tracking_editor import accept_tracking
        accept_tracking(self)

    def discard_tracking(self):
        from .correction_tracking_editor import discard_tracking
        discard_tracking(self)

    def tracking_summary(self):
        from .correction_tracking_editor import tracking_summary
        return tracking_summary(self)

    def tracking_view(self, frame):
        from .correction_tracking_editor import tracking_view
        return tracking_view(self, frame)

    def tracking_overlay_view(self, frame, kind, opacity=.55):
        from .correction_tracking_editor import tracking_overlay_view
        return tracking_overlay_view(self, frame, kind, opacity)

    def clear_tracking(self):
        data = self.document.to_dict()
        if not data.get('tracks'):
            raise ValueError('No accepted tracked corrections to clear.')
        data['tracks'] = []
        self._apply(data)

    def prepare_plate(self, **parameters):
        from .clean_plate_editor import prepare_plate
        prepare_plate(self, **parameters)

    def plate_view(self, mode='guidance'):
        from .clean_plate_editor import plate_view
        return plate_view(self, mode)

    def plate_summary(self):
        from .clean_plate_editor import plate_summary
        return plate_summary(self)

    def plate_reference(self):
        if self.plate_preview is None:
            return None
        display = self.plate_preview['reference_display']
        return replace(display, pixels=display.pixels.copy())

    def accept_plate(self):
        from .clean_plate_editor import accept_plate
        accept_plate(self)

    def discard_plate(self):
        from .clean_plate_editor import discard_plate
        discard_plate(self)

    def detach_guidance(self, identifier):
        data = self.document.to_dict()
        entry = next(entry for entry in data['entries'] if entry['id'] == identifier)
        if 'guidance_id' not in entry:
            raise ValueError('Selected entry has no generated guidance origin.')
        origin_id = entry.pop('guidance_id')
        data['guidance'] = [origin for origin in data['guidance'] if origin['id'] != origin_id]
        entry['revision'] = self.revision + 1
        self._apply(data)

    def prepare_subject(self, **parameters):
        from .subject_guidance_editor import prepare_subject
        prepare_subject(self, **parameters)

    def predict_subject(self, **parameters):
        from .subject_guidance_editor import predict_subject
        predict_subject(self, **parameters)

    def subject_summary(self):
        from .subject_guidance_editor import subject_summary
        return subject_summary(self)

    def subject_view(self):
        from .subject_guidance_editor import subject_view
        return subject_view(self)

    def clear_subject(self):
        from .subject_guidance_editor import clear_subject
        clear_subject(self)

    def accept_subject(self):
        from .subject_guidance_editor import accept_subject
        accept_subject(self)

    def discard_subject(self):
        from .subject_guidance_editor import discard_subject
        discard_subject(self)

    def _checkpoint(self, document: CorrectionDocument) -> None:
        self.check()
        path = self.workspace / f'edit-r{document.to_dict()["revision"]}-{uuid.uuid4().hex}.json'
        save_corrections(path, document)
        self.checkpoint_path = path

    def _entry(self, kind: str, frame: int, strength: float, direction: str, span: list[int], asset: Path,
               *, sha256: str | None = None, purpose: str = 'refinement') -> dict[str, Any]:
        return {'id': uuid.uuid4().hex, 'revision': self.revision + 1, 'frame': frame, 'kind': kind,
                'asset': {'path': str(asset.resolve()), 'sha256': sha256 if sha256 is not None else file_sha256(asset, check=self.check)},
                'strength': strength, 'direction': direction, 'range': span,
                **({'purpose': purpose, 'target': REGION_TARGETS.get(purpose)} if kind == 'region' else {})}

    def import_mask(self, path: Path, *, kind: str, frame: int, strength: float, direction: str, span: list[int], purpose: str = 'refinement') -> None:
        if kind not in {'matte', 'keep', 'remove', 'region'}:
            raise ValueError('Coverage import supports matte/keep/remove/region; use explicit trimap import for classes.')
        self._import_raster(path, kind=kind, frame=frame, strength=strength, direction=direction, span=span, purpose=purpose)

    def import_trimap(self, path: Path, *, frame: int, strength: float, direction: str, span: list[int]) -> None:
        self._import_raster(path, kind='trimap', frame=frame, strength=strength, direction=direction, span=span)

    def _import_raster(self, path: Path, *, kind: str, frame: int, strength: float, direction: str, span: list[int], origin=None, purpose: str = 'refinement') -> None:
        from .correction_tracking_editor import MAX_CLEANUP_PENDING, _cleanup
        for pending in list(self.tracking_cleanup_pending):
            _cleanup(self, pending)
        if len(self.tracking_cleanup_pending) >= MAX_CLEANUP_PENDING:
            raise OSError('Private asset cleanup remains blocked; inspect pending cleanup before importing more masks.')
        admitted_asset = None
        if kind in ('trimap', 'region'):
            from .trimap_refinement import load_coverage_asset, load_trimap_asset
            # Admit the original lexical path before resolution or any unbounded hash.
            loader = load_coverage_asset if kind == 'region' else load_trimap_asset
            admitted_asset = loader(path, expected_hw=self.meta['native_hw'], check=self.check)
            entry = self._entry(kind, frame, strength, direction, span, path, sha256=admitted_asset.sha256, purpose=purpose)
        else:
            path = path.resolve()
            entry = self._entry(kind, frame, strength, direction, span, path)
        data = self.document.to_dict()
        data['entries'].append(entry)
        if data['schema_version'] >= 3:
            data['tracks'] = []
        if kind == 'trimap':
            data['schema_version'] = max(2, data['schema_version'])
        if kind == 'region':
            data['schema_version'] = 5
            data.setdefault('tracks', [])
            data.setdefault('guidance', [])
        candidate = CorrectionDocument.parse(data, self.document.base_dir)
        if origin is not None:
            from .corrections import attach_guidance
            candidate = attach_guidance(candidate, entry['id'], guidance_id=uuid.uuid4().hex, **origin)
            candidate.verify_assets(check=self.check)
            data = candidate.to_dict()
            entry = next(item for item in data['entries'] if item['id'] == entry['id'])
        rasters = CorrectionRasters(candidate, self.meta, check=self.check)
        rasters.close()
        # Copy exact verified encoded bytes, preserving authored float/uint16 precision.
        raw = admitted_asset.encoded_bytes if admitted_asset is not None else path.read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != entry['asset']['sha256']:
            raise ValueError('Imported raster changed before publication.')
        target = self.workspace / 'assets' / (uuid.uuid4().hex + path.suffix.lower())
        created, previous_checkpoint = False, self.checkpoint_path
        try:
            with target.open('xb') as stream:
                created = True
                stream.write(raw)
            entry['asset']['path'] = str(target)
            self._apply(data)
        except BaseException:
            # Never remove an accepted asset or one owned by a successfully
            # published recovery checkpoint. Failed checkpoint writes own none.
            if created and self.checkpoint_path == previous_checkpoint and target.resolve() not in self.document.asset_paths():
                _cleanup(self, target)
            raise

    def commit_brush(self, mask: NDArray[np.uint8], *, kind: str, frame: int, strength: float,
                     direction: str, span: list[int], purpose: str = 'refinement') -> None:
        if mask.dtype != np.uint8 or mask.shape != (self.binding['height'], self.binding['width']):
            raise ValueError('Brush must be a native-size uint8 coverage mask.')
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError('This brush authors binary regions; use imported masks for soft coverage.')
        # Binary values are exact in uint16 PNG; no imported mask is quantized.
        encoded_ok, encoded = cv2.imencode('.png', mask.astype(np.uint16) * 257)
        if not encoded_ok:
            raise ValueError('Could not encode correction brush mask.')
        path = self.workspace / 'assets' / (uuid.uuid4().hex + '.png')
        with path.open('xb') as stream:
            stream.write(encoded.tobytes())
        try:
            self.import_mask(path, kind=kind, frame=frame, strength=strength, direction=direction, span=span, purpose=purpose)
        finally:
            path.unlink(missing_ok=True)  # Only this private, newly created staging mask.

    def reset(self, frame: int) -> None:
        data = self.document.to_dict()
        data['entries'].append({'id': uuid.uuid4().hex, 'revision': self.revision + 1, 'kind': 'reset', 'frame': frame})
        self._apply(data)

    def edit(self, identifier: str, changes: dict[str, Any]) -> None:
        if set(changes) - {'frame', 'kind', 'strength', 'direction', 'range'}:
            raise ValueError('Only authored entry fields can be edited; asset replacement is a new entry.')
        data = self.document.to_dict()
        entry = next(e for e in data['entries'] if e['id'] == identifier)
        if (changes.get('kind', entry['kind']) == 'trimap') != (entry['kind'] == 'trimap'):
            raise ValueError('Trimap classes and coverage kinds cannot be reinterpreted; import a new raster explicitly.')
        if (changes.get('kind', entry['kind']) == 'region') != (entry['kind'] == 'region'):
            raise ValueError('Regions and alpha corrections cannot be reinterpreted; import a new raster explicitly.')
        entry.update(changes)
        entry['revision'] = self.revision + 1
        self._apply(data)

    def remove(self, identifier: str) -> None:
        data = self.document.to_dict()
        entries = data['entries']
        data['entries'] = [e for e in entries if e['id'] != identifier]
        if len(entries) == len(data['entries']):
            raise ValueError('No selected correction entry.')
        self._apply(data)

    def move(self, identifier: str, delta: int) -> None:
        data = self.document.to_dict()
        entries = data['entries']
        index = next(i for i, e in enumerate(entries) if e['id'] == identifier)
        destination = index + delta
        if not 0 <= destination < len(entries):
            raise ValueError('Entry is already at the end of this ordering.')
        entries[index], entries[destination] = entries[destination], entries[index]
        self._apply(data)

    def shot(self, frame: int, *, remove: bool = False) -> None:
        data = self.document.to_dict()
        boundaries = {point for shot in data['shots'] for point in shot}
        if not 0 < frame < self.binding['frame_count']:
            raise ValueError('Shot boundary must lie inside the selected timeline.')
        if remove:
            boundaries.remove(frame)
        else:
            boundaries.add(frame)
        points = sorted(boundaries)
        data['shots'] = [list(pair) for pair in pairwise(points)]
        self._apply(data)

    def undo(self) -> None:
        invalidated = self.history.undo()
        try:
            self._checkpoint(self.document)
        except BaseException:
            self.history.redo()
            raise
        self.invalidated = invalidated
        self.saved_path = None
        self.discard_tracking()
        self.discard_plate()
        self.discard_subject()

    def redo(self) -> None:
        invalidated = self.history.redo()
        try:
            self._checkpoint(self.document)
        except BaseException:
            self.history.undo()
            raise
        self.invalidated = invalidated
        self.saved_path = None
        self.discard_tracking()
        self.discard_plate()
        self.discard_subject()

    def save(self) -> Path:
        if source_identity(self.job.input_path, check=self.check)['sha256'] != self.binding['sha256']:
            raise ValueError('Source changed; correction project must be rebound explicitly.')
        self.document.verify_assets(check=self.check)
        path = self.workspace / f'corrections-r{self.document.to_dict()["revision"]}-{uuid.uuid4().hex}.json'
        save_corrections(path, self.document)
        self.saved_path = path
        return path

    def view(self, frame: int, identifier: str | None = None) -> tuple[PreviewFrame, NDArray[np.uint8] | None]:
        if type(frame) is not int or not 0 <= frame < self.binding['frame_count']:
            raise ValueError('Editor frame is outside the selected timeline.')
        config, _ = self.job.configurations()
        data = self.document.to_dict() if identifier is not None else None
        region_track = None
        if data is not None:
            region = next((entry for entry in data['entries'] if entry['id'] == identifier and entry['kind'] == 'region'), None)
            if region is not None:
                region_track = next((track for track in data.get('tracks', []) if identifier in track['anchor_ids']
                                     and track['range'][0] <= frame < track['range'][1]), None)
        # Capture source membership/stat metadata BEFORE native preview decoding.
        source_index = scan_sequence(self.job.input_path, check=self.check) if region_track is not None else None
        if source_index is not None:
            source_index.check_path(frame + self.binding['start_frame'])
        display = preview(self.job.input_path, frame + self.binding['start_frame'],
                          linear=config.frame_dir_linear if Path(self.job.input_path).is_dir() else False,
                          alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile)
        if source_index is not None:
            source_index.check_path(frame + self.binding['start_frame'])
        if (display.height, display.width) != self.meta['native_hw']:
            raise ValueError('Source frame geometry changed.')
        overlay = None
        tracked_region_inputs = None
        if identifier is not None:
            assert data is not None
            entry = next(e for e in data['entries'] if e['id'] == identifier)
            if entry['kind'] == 'trimap' and entry['range'][0] <= frame < entry['range'][1]:
                from .trimap_refinement import load_trimap_asset
                asset = load_trimap_asset(self.document.asset_path(entry), expected_hw=self.meta['native_hw'],
                                          expected_sha256=entry['asset']['sha256'], check=self.check)
                overlay = cv2.resize(asset.labels, (display.pixels.shape[1], display.pixels.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            elif entry['kind'] == 'region':
                track = region_track
                coverage = None
                if track is not None:
                    from .corrections import load_tracked_asset
                    from .trimap_refinement import load_coverage_asset
                    tracked_region_inputs = []
                    for anchor in data['entries']:
                        if anchor['id'] in track['anchor_ids']:
                            path = self.document.base_dir / anchor['asset']['path']
                            tracked_region_inputs.append(admit_file(path))
                            load_coverage_asset(path, expected_hw=self.meta['native_hw'],
                                expected_sha256=anchor['asset']['sha256'], check=self.check)
                    row = track['frames'][frame - track['range'][0]]
                    observation = track['observations'][frame - track['range'][0]]
                    tracked_region_inputs.append(admit_file(self.document.base_dir / row['asset']['path']))
                    coverage = load_tracked_asset(self.document.base_dir / row['asset']['path'],
                        expected_hw=self.meta['native_hw'], expected_sha256=row['asset']['sha256'],
                        expected_mask_sha256=observation['derived_mask_sha256'], check=self.check)
                    display = replace(display, note=display.note + f" Accepted tracked {entry['purpose']}/{entry['target']} region coverage; purpose-specific weights only.")
                elif entry['range'][0] <= frame < entry['range'][1]:
                    from .trimap_refinement import load_coverage_asset
                    coverage = load_coverage_asset(self.document.base_dir / entry['asset']['path'],
                        expected_hw=self.meta['native_hw'], expected_sha256=entry['asset']['sha256'], check=self.check).coverage
                if coverage is not None:
                    overlay = np.rint(cv2.resize(coverage, (display.pixels.shape[1], display.pixels.shape[0]),
                                                 interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
            elif entry['kind'] != 'reset' and entry['range'][0] <= frame < entry['range'][1]:
                # Obtain the actual stored coverage through the shared matte operator.
                isolated = {**entry, 'kind': 'matte', 'strength': 1.0, 'direction': 'both'}
                isolated.pop('guidance_id', None)
                isolated.pop('purpose', None)
                isolated.pop('target', None)
                candidate = CorrectionDocument.parse({**data, 'entries': [isolated],
                    **({'tracks': []} if data['schema_version'] >= 3 else {}),
                    **({'guidance': []} if data['schema_version'] >= 4 else {})}, self.document.base_dir)
                rasters = CorrectionRasters(candidate, self.meta, check=self.check)
                try:
                    mask = rasters.apply(np.zeros(self.meta['native_hw'], dtype=np.float32), entry['frame'])
                finally:
                    rasters.close()
                overlay = np.rint(cv2.resize(mask, (display.pixels.shape[1], display.pixels.shape[0]),
                                             interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        if tracked_region_inputs is not None:
            from .correction_tracking_editor import _source_hash
            assert source_index is not None
            if _source_hash(source_index, self.check) != self.binding['sha256']:
                raise ValueError('Source changed since accepted region tracking; re-track before inspection')
            for pin in tracked_region_inputs:
                self.check()
                assert_pin(pin)
            if scan_sequence(self.job.input_path, check=self.check).identity != source_index.identity:
                raise ValueError('Source sequence changed during accepted region inspection; re-track before inspection')
        return display, overlay
