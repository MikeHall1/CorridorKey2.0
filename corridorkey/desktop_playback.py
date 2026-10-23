"""Bounded sequence playback through the workspace's sole preview decoder."""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .desktop_comparison import common_range
from .playback import FrameRequest, Playback
from .preview_exports import (
    ExportRequest,
    PreparedExportIndexes,
    capture_export_request,
)
from .preview_sequence import PreparedSequence
from .session import JobRequest

if TYPE_CHECKING:
    from .desktop_window import DesktopWindow


class PlaybackControls(QWidget):
    """No new decoder, queue, timer or persistent frame cache.

    The workspace calls tick from its existing 50ms poll. The loader can replace
    queued/completed manual requests: a later serial's terminal result also
    proves the sole sequential decoder retired an earlier playback request.
    """

    def __init__(self, workspace: DesktopWindow):
        super().__init__(workspace)
        self.workspace = workspace
        self.engine: Playback | None = None
        self.pending_serial: int | None = None
        self.pending_request: FrameRequest | None = None
        self.selected_id: str | None = None
        self.displayed_frame: int | None = None
        self.displayed_seconds: float | None = None
        self.binding: tuple[object, ...] | None = None
        self.source_sequence: PreparedSequence | None = None
        self.export_indexes: PreparedExportIndexes | None = None
        self.preparation_serial: int | None = None
        self.preparation_context: tuple[int, JobRequest | ExportRequest] | None = None
        self.navigation_generation = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        play = QPushButton('Play sequence')
        play.clicked.connect(self.play)
        row.addWidget(play)
        pause = QPushButton('Pause')
        pause.clicked.connect(self.pause)
        row.addWidget(pause)
        refresh = QPushButton('Refresh sequence indexes')
        refresh.clicked.connect(self.refresh_source)
        row.addWidget(refresh)
        self.loop = QCheckBox('Loop source range')
        self.loop.toggled.connect(self.change_loop)
        row.addWidget(self.loop)
        self.in_frame, self.out_frame = QLineEdit(), QLineEdit()
        self.in_frame.setPlaceholderText('Source In')
        self.out_frame.setPlaceholderText('Out exclusive')
        for field in (self.in_frame, self.out_frame):
            field.setMaximumWidth(110)
            field.textChanged.connect(self.navigate)
            row.addWidget(field)
        self.speed = QComboBox()
        for label, value in [('¼×', .25), ('½×', .5), ('1×', 1.), ('2×', 2.), ('4×', 4.)]:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self.change_speed)
        row.addWidget(self.speed)
        layout.addLayout(row)
        self.status = QLabel('Playback: Source can play a local image sequence directly; export/A-B views require completed renders. Declared sequence timing only; video/VFR unsupported.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def play(self):
        owner = self.workspace
        try:
            if owner.closing:
                raise ValueError('Workspace is closing.')
            if owner.view.currentText() == 'Source' and owner.moving_reference is None:
                self._play_source()
                return
            view = owner.selected_render_view()
            request_id = owner.selected()
            if request_id is None:
                raise ValueError('Select a completed sequence render.')
            state = owner.jobs[request_id]
            config, _ = state['job'].configurations()
            if not Path(view.source).is_dir() or config.output_format != 'exr':
                raise ValueError('Playback currently requires a completed source image sequence with EXR exports. Video/VFR or unknown timing is unsupported.')
            if not math.isclose(view.fps, config.frame_dir_fps, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError('Recorded sequence timing differs from its declared frame_dir_fps.')
            reference = owner.moving_reference
            if reference is not None and (not reference.linear or not Path(reference.source).is_dir()):
                raise ValueError('Synchronized playback A also requires recorded sequence/EXR timing.')
            mode = owner.view.currentText()
            background_image = getattr(owner, 'background_image', None)
            if mode == 'Source' and owner.source.text().strip() != view.source:
                raise ValueError('Source playback must match the selected completed render source.')
            for current, current_mode in ((view, mode), (reference, owner.reference_mode or mode)):
                if current is None:
                    continue
                options = current.options(current_mode, current.start, owner.background, background_image=background_image)
                if any(not Path(options[key]).is_dir() for key in ('path', 'alpha_path') if key in options):
                    raise ValueError('Playback supports image-sequence directories only; video/VFR exports are unsupported.')
            start, end = common_range(view, reference) if reference is not None else (view.start, view.end)
            lower = int(self.in_frame.text()) if self.in_frame.text().strip() else start
            upper = int(self.out_frame.text()) if self.out_frame.text().strip() else end
            binding = (request_id, state['job'], view, reference, mode, owner.reference_mode,
                       owner.background, None if background_image is None else background_image.identity, lower, upper, owner.frame.value(), owner.source.text(),
                       owner.config.toPlainText())
            if self.engine is not None and self.binding == binding:
                if self.engine.playing:
                    return  # Repeated Play must not reset a running timeline.
            else:
                self.displayed_frame = self.displayed_seconds = None
                timing_evidence = f'Completed source metadata and job frame_dir_fps={config.frame_dir_fps}; declared constant sequence spacing, not video PTS.'
                if self.engine is None:
                    self.engine = Playback(view, source_identity=view.source, timebase='sequence_declared',
                                           timing_evidence=timing_evidence, reference=reference, mode=mode,
                                           reference_mode=owner.reference_mode or mode, background=owner.background,
                                           background_image=background_image)
                else:
                    self.engine.bind(view, source_identity=view.source, timebase='sequence_declared',
                                     timing_evidence=timing_evidence, reference=reference, mode=mode,
                                     reference_mode=owner.reference_mode or mode, background=owner.background,
                                     background_image=background_image)
                self.engine.set_range(lower, upper)
                self.engine.set_speed(self.speed.currentData())
                self.engine.set_loop(self.loop.isChecked())
                # Still navigation is local to an export, absolute for Source
                # and synchronized A/B. Explicit navigation replaces the cursor.
                index = owner.frame.value()
                if reference is None and mode != 'Source':
                    index += view.start
                self.engine.seek(min(max(index, lower), upper - 1))
                self.binding = binding
            self.selected_id = owner.selected()
            request = capture_export_request(view, mode=mode, reference=reference, reference_mode=owner.reference_mode or mode)
            if self.export_indexes is None or self.export_indexes.request.identity != request.identity:
                context = (self.navigation_generation, request)
                if self.preparing and self.preparation_context == context:
                    return
                self.pause()
                self.export_indexes = None
                self.preparation_context = context
                self.preparation_serial = owner.loader.submit(prepare_exports=request)
                owner.preview_serial = self.preparation_serial
                self.status.setText('Preparing bounded source/foreground/alpha sequence indexes off-thread; no pixels cached or render settings changed.')
                return
            self.engine.bind_export_indexes(self.export_indexes)
            owner.preview_serial = -1
            self.engine.play()
            self.tick()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.pause()
            self.status.setText(str(exc))

    @property
    def preparing(self):
        return self.preparation_serial is not None and self.preparation_context is not None

    def _play_source(self):
        owner = self.workspace
        job = owner.job(preview_only=True)
        if not Path(job.input_path).is_dir():
            raise ValueError('Source playback requires an image-sequence directory; video/VFR or unknown timing is unsupported.')
        if self.source_sequence is not None and self.source_sequence.job == job:
            self._start_source(self.source_sequence)
            return
        context = (self.navigation_generation, job)
        if self.preparing and self.preparation_context == context:
            return
        self.pause()
        self.selected_id = owner.selected()
        self.preparation_context = context
        self.preparation_serial = owner.loader.submit(prepare_sequence=job)
        owner.preview_serial = self.preparation_serial
        owner.pixmap = None
        owner.image.setText('Preparing bounded local source sequence…')
        owner.preview_note.setText('Indexing local image paths and decoding the selected first frame off-thread; no model loading or inference.')
        self.status.setText('Preparing local source playback: at most 100,000 directory entries/frames and 32 MiB of indexed path bytes. Native image decode memory is unchanged; no pixel cache.')

    def accept_preparation(self, serial, frame, error):
        metadata = isinstance(frame, (PreparedSequence, PreparedExportIndexes))
        if self.preparation_serial is None or serial < self.preparation_serial:
            return metadata
        expected_serial = self.preparation_serial
        context = self.preparation_context
        self.preparation_serial, self.preparation_context = None, None
        if serial != expected_serial:
            if context is not None:
                self.status.setText('Sequence preparation superseded; choose Play explicitly to prepare again.')
            return metadata  # Unrelated photo/still terminals must continue through normal routing.
        if context is None or context[0] != self.navigation_generation or self.workspace.closing:
            return True
        if error is not None:
            self.status.setText('Sequence preparation failed: ' + error)
            return True
        try:
            if isinstance(context[1], ExportRequest):
                if not isinstance(frame, PreparedExportIndexes) or frame.request.identity != context[1].identity:
                    raise ValueError('Export preparation returned a mismatched snapshot; no playback started.')
                if self.engine is None:
                    raise ValueError('Playback binding retired during export preparation.')
                self.engine.bind_export_indexes(frame)
                self.export_indexes = frame
                self.workspace.preview_serial = -1
                self.engine.play()
                self.tick()
            else:
                if not isinstance(frame, PreparedSequence) or frame.job != context[1]:
                    raise ValueError('Source preparation returned a mismatched snapshot; no playback started.')
                self.source_sequence = frame
                self._start_source(frame)
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.pause()
            self.status.setText(str(exc))
        return True

    def _start_source(self, prepared: PreparedSequence):
        owner, view = self.workspace, prepared.view
        lower = int(self.in_frame.text()) if self.in_frame.text().strip() else view.start
        upper = int(self.out_frame.text()) if self.out_frame.text().strip() else view.end
        binding = ('source', prepared.job, prepared.index.identity, view, lower, upper,
                   owner.frame.value(), owner.selected())
        if self.engine is not None and self.binding == binding:
            if self.engine.playing:
                return
        else:
            self.displayed_frame = self.displayed_seconds = None
            timing_evidence = f'Local source membership prepared off-thread; explicit frame_dir_fps={view.fps:g}, constant sequence spacing; not video PTS.'
            if self.engine is None:
                self.engine = Playback(view, source_identity=view.source, timebase='sequence_declared',
                                       timing_evidence=timing_evidence, mode='Source')
            else:
                self.engine.bind(view, source_identity=view.source, timebase='sequence_declared',
                                 timing_evidence=timing_evidence, mode='Source')
            self.engine.set_range(lower, upper)
            self.engine.set_speed(self.speed.currentData())
            self.engine.set_loop(self.loop.isChecked())
            self.engine.seek(min(max(owner.frame.value(), lower), upper - 1))
            self.binding = binding
        self.selected_id = owner.selected()
        owner.preview_serial = -1
        self.engine.play()
        self.tick()

    def pause(self, *_):
        if self.preparing:
            assert self.preparation_serial is not None
            self.workspace.loader.cancel(self.preparation_serial)
            self.preparation_context = None
            self.workspace.preview_serial = -1
            self.status.setText('Sequence preparation cancelled; any native read must finish before the owned decoder exits.')
        if self.engine is None:
            return
        self.engine.pause()
        if self.displayed_frame is not None and self.engine.start <= self.displayed_frame < self.engine.end:
            self.engine.seek(self.displayed_frame)  # Freeze visible pixels, not a buffered clock target.
        # A native decode may still run. Keep its ownership until a same/later
        # terminal loader serial proves retirement; fence pixels immediately.
        if self.pending_serial is not None:
            self.workspace.preview_serial = -1
        self._status('Paused')

    def navigate(self, *_):
        """An explicit still preview/seek/context change supersedes resume."""
        self.navigation_generation += 1
        self.binding = None
        self.source_sequence = None
        self.displayed_frame = self.displayed_seconds = None
        self.pause()

    def refresh_source(self):
        self.navigate()
        self.export_indexes = None
        if self.engine is not None:
            self.engine.export_indexes = None
        self.status.setText('Frozen source/export indexes cleared. Play explicitly to inspect current sequence membership again; no render settings changed.')

    def selection_changed(self):
        if not self.workspace._updating_diagnostic_rows and self.workspace.selected() != self.selected_id:
            self.navigate()

    def change_speed(self, *_):
        if self.engine is not None:
            self.engine.set_speed(self.speed.currentData())
            if self.pending_serial is not None:
                self.workspace.preview_serial = -1
            self._status('Playing' if self.engine.playing else 'Paused')

    def change_loop(self, *_):
        if self.engine is not None:
            self.engine.set_loop(self.loop.isChecked())
            if self.pending_serial is not None:
                self.workspace.preview_serial = -1

    def _status(self, state):
        if self.engine is None:
            return
        position = (f'Displayed source frame {self.displayed_frame} · {self.displayed_seconds:.6f}s'
                    if self.displayed_frame is not None and self.displayed_seconds is not None else 'No playback frame displayed')
        self.status.setText(f'{state} · {position} · declared {self.engine.view.fps:g} fps · '
                            f'buffered opportunities {self.engine.buffered_opportunities} · '
                            f'dropped opportunities {self.engine.dropped_opportunities} · '
                            f'stale completions {self.engine.stale_completions}. '
                            '50ms UI poll; no real-time guarantee. Source path/range binding, not content authentication.')

    def tick(self):
        if self.engine is None or not self.engine.playing or self.workspace.closing:
            return
        request = self.engine.tick()
        if request is None:
            self._status('Buffering' if self.pending_serial is not None else 'Playing')
            return
        try:
            options = dict(request.options)
            if self.source_sequence is not None and self.engine.reference is None and self.engine.mode == 'Source':
                options['sequence'] = self.source_sequence.index
            serial = self.workspace.loader.submit(**options)
        except (ValueError, RuntimeError) as exc:
            self.engine.complete(request, succeeded=False)
            self.pause()
            self.status.setText(str(exc))
            return
        self.pending_serial, self.pending_request = serial, request
        self.workspace.preview_serial = serial
        photo_identity = None if self.engine.background_image is None else self.engine.background_image.identity[:16]
        photo_b = f' · photo {photo_identity}' if self.engine.mode == 'Photo composite' and photo_identity is not None else ''
        self.workspace.preview_description = f'{self.engine.mode}{photo_b} · source frame {request.source_frame} · {request.source_seconds:.6f}s · {self.engine.view.source}'
        if self.engine.reference is not None:
            photo_a = f' · photo {photo_identity}' if self.engine.reference_mode == 'Photo composite' and photo_identity is not None else ''
            self.workspace.reference_description = f'{self.engine.reference_mode}{photo_a} · source frame {request.source_frame} · {request.source_seconds:.6f}s · synchronized A'
        self._status('Buffering')

    def complete_loader(self, serial: int, error: str | None) -> bool | None:
        """Return None for unrelated stills, False for fenced playback pixels."""
        if self.pending_serial is None or serial < self.pending_serial:
            return None
        assert self.engine is not None and self.pending_request is not None
        request, playback_serial = self.pending_request, self.pending_serial
        self.pending_request, self.pending_serial = None, None
        current = (request.generation == self.engine.generation
                   and request.source_identity == self.engine.source_identity)
        valid = self.engine.complete(request, succeeded=error is None and serial == playback_serial)
        if serial != playback_serial:
            return None
        if valid:
            self.displayed_frame, self.displayed_seconds = request.source_frame, request.source_seconds
        if error is not None and current:
            self.pause()
            if self.source_sequence is not None:
                self.source_sequence = None
                self.binding = None
            self.status.setText('Playback decode failed: ' + error + ' Refresh sequence indexes explicitly after checking changed or missing files; no automatic retry.')
        else:
            self._status('Playing' if self.engine.playing else 'Paused / range end')
        return valid
