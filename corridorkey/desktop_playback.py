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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        play = QPushButton('Play sequence')
        play.clicked.connect(self.play)
        row.addWidget(play)
        pause = QPushButton('Pause')
        pause.clicked.connect(self.pause)
        row.addWidget(pause)
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
        self.status = QLabel('Playback: select a completed image-sequence render. Declared sequence timing only; video/VFR unsupported.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def play(self):
        owner = self.workspace
        try:
            if owner.closing:
                raise ValueError('Workspace is closing.')
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
            if mode == 'Source' and owner.source.text().strip() != view.source:
                raise ValueError('Source playback must match the selected completed render source.')
            for current, current_mode in ((view, mode), (reference, owner.reference_mode or mode)):
                if current is None:
                    continue
                options = current.options(current_mode, current.start, owner.background)
                if any(not Path(options[key]).is_dir() for key in ('path', 'alpha_path') if key in options):
                    raise ValueError('Playback supports image-sequence directories only; video/VFR exports are unsupported.')
            start, end = common_range(view, reference) if reference is not None else (view.start, view.end)
            lower = int(self.in_frame.text()) if self.in_frame.text().strip() else start
            upper = int(self.out_frame.text()) if self.out_frame.text().strip() else end
            binding = (request_id, state['job'], view, reference, mode, owner.reference_mode,
                       owner.background, lower, upper, owner.frame.value(), owner.source.text(),
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
                                           reference_mode=owner.reference_mode or mode, background=owner.background)
                else:
                    self.engine.bind(view, source_identity=view.source, timebase='sequence_declared',
                                     timing_evidence=timing_evidence, reference=reference, mode=mode,
                                     reference_mode=owner.reference_mode or mode, background=owner.background)
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
            owner.preview_serial = -1
            self.engine.play()
            self.tick()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.pause()
            self.status.setText(str(exc))

    def pause(self, *_):
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
        self.binding = None
        self.displayed_frame = self.displayed_seconds = None
        self.pause()

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
            serial = self.workspace.loader.submit(**request.options)
        except (ValueError, RuntimeError) as exc:
            self.engine.complete(request, succeeded=False)
            self.pause()
            self.status.setText(str(exc))
            return
        self.pending_serial, self.pending_request = serial, request
        self.workspace.preview_serial = serial
        self.workspace.preview_description = f'{self.engine.mode} · source frame {request.source_frame} · {request.source_seconds:.6f}s · {self.engine.view.source}'
        if self.engine.reference is not None:
            self.workspace.reference_description = f'{self.engine.reference_mode} · source frame {request.source_frame} · {request.source_seconds:.6f}s · synchronized A'
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
            self.status.setText('Playback decode failed: ' + error)
        else:
            self._status('Playing' if self.engine.playing else 'Paused / range end')
        return valid
