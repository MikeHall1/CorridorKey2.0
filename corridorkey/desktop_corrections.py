"""Native correction editor dialog on the existing history/raster pipeline."""
from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .correction_canvas import CorrectionCanvas
from .correction_editor_service import EditorService
from .session import JobRequest


class CorrectionDialog(QDialog):
    saved = Signal(object, str)

    def __init__(self, job: JobRequest, parent_dir: Path, parent=None, *, existing: Path | None = None):
        super().__init__(parent)
        self.job = job
        self.setWindowTitle('CorridorKey · Authored corrections')
        self.resize(1100, 920)
        self.service = EditorService(job, parent_dir, existing=existing)
        self.snapshot: dict[str, Any] | None = None
        self.pending: int | None = None
        self.displayed_frame: int | None = None
        self.displayed_role = 'authored'
        self.view_requested_frame = 0
        self.entries: list[dict[str, Any]] = []
        self.closing = False
        self.presented_entry: str | None = None
        self.presented_strength: float | None = None
        self.tracking_anchors: list[str] = []
        root = QVBoxLayout(self)
        self.help = QLabel('Trimaps are imported native classes: foreground green, background red, unknown transparent. Unknown is not 50% alpha. Fixed native 1-pixel RGB-guided refinement adjusts UNKNOWN; strength blends once, known classes are exact at strength 1. Trimaps affect final alpha only, not neural hints, carried mattes or flow; later entries can override. Global alpha refinement is separate. Quick/spatial draft remains approximate. Paint binary native regions (blue draft; green stored mask). Commit mask before saving. Matte defines full-frame coverage, including zero outside paint; keep/remove protect only painted regions. Directional ranges repeat a static raster, not motion tracking. Undo/redo operates on committed edits. This overlay is authored coverage, not a neural key.')
        self.help.setWordWrap(True)
        root.addWidget(self.help)
        controls = QHBoxLayout()
        self.frame, self.start, self.end, self.radius = QSpinBox(), QSpinBox(), QSpinBox(), QSpinBox()
        self.frame.setRange(0, 1000000000)
        self.start.setRange(0, 1000000000)
        self.end.setRange(1, 1000000000)
        self.end.setValue(1)
        self.radius.setRange(1, 2048)
        self.radius.setValue(12)
        self.kind, self.direction = QComboBox(), QComboBox()
        self.kind.addItems(['keep', 'remove', 'matte'])
        self.direction.addItems(['both', 'forward', 'backward'])
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0, 1)
        self.strength.setDecimals(3)
        self.strength.setValue(1)
        self.eraser = QCheckBox('Erase draft')
        for label, widget in [('Frame', self.frame), ('Kind', self.kind), ('Strength', self.strength),
                              ('Direction', self.direction), ('Start', self.start), ('End exclusive', self.end),
                              ('Radius px', self.radius)]:
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(self.eraser)
        root.addLayout(controls)
        self.canvas = CorrectionCanvas()
        self.radius.valueChanged.connect(lambda value: setattr(self.canvas, 'radius', value))
        self.eraser.toggled.connect(lambda value: setattr(self.canvas, 'value', 0 if value else 255))
        root.addWidget(self.canvas, 3)
        self.frame_note = QLabel('Binding selected source/timeline in background…')
        root.addWidget(self.frame_note)
        first = QHBoxLayout()
        for text, callback in [('Load frame', self.load_frame), ('Commit brush', self.commit), ('Clear draft', self.clear_draft),
                               ('Import mask…', self.import_mask), ('Import trimap…', self.import_trimap), ('New mask', self.new_mask),
                               ('Save snapshot & attach', self.save)]:
            self._button(text, callback, first)
        root.addLayout(first)
        second = QHBoxLayout()
        self.selection_buttons: list[QPushButton] = []
        for text, callback in [('Undo committed', lambda: self.request('undo')), ('Redo committed', lambda: self.request('redo')),
                               ('Add reset', lambda: self.request('reset', frame=self.frame.value())),
                               ('Split shot', lambda: self.request('shot', frame=self.frame.value())),
                               ('Merge at frame', lambda: self.request('shot', frame=self.frame.value(), remove=True)),
                               ('Update selected', self.edit), ('Delete selected', self.remove),
                               ('Move up', lambda: self.move_entry(-1)), ('Move down', lambda: self.move_entry(1))]:
            button = self._button(text, callback, second)
            if text in {'Update selected', 'Delete selected', 'Move up', 'Move down'}:
                self.selection_buttons.append(button)
                button.setEnabled(False)
        root.addLayout(second)
        tracking = QHBoxLayout()
        self.tracking_policy = QComboBox()
        self.tracking_policy.addItems(['request_anchor', 'stop', 'retain_approximation'])
        self.tracking_policy.setToolTip('Flow/photometric reliability is heuristic, not calibrated confidence. Retain approximation explicitly accepts uncertain draft propagation.')
        tracking.addWidget(QLabel('Tracking reliability'))
        tracking.addWidget(self.tracking_policy)
        for index, (text, callback) in enumerate([('Add selected anchor', self.add_tracking_anchor),
                               ('Clear anchor list', self.clear_tracking_anchors),
                               ('Preview track', self.preview_tracking),
                               ('Inspect tracked frame', self.inspect_tracking),
                               ('Accept inspected track', lambda: self.request('track_accept')),
                               ('Discard preview', lambda: self.request('track_discard')),
                               ('Clear accepted tracks', lambda: self.request('track_clear')),
                               ('Cancel operation', self.cancel_operation)]):
            if index == 3:
                root.addLayout(tracking)
                tracking = QHBoxLayout()
            self._button(text, callback, tracking)
        root.addLayout(tracking)
        self.tracking_note = QLabel('Sequence tracking: choose same-kind/strength contiguous authored anchors; Start/End and Direction set tracking range. Preview changes no history. Inspect a retained frame before explicit acceptance. Cuts/resets stop the range. Video/trimap tracking unavailable.')
        self.tracking_note.setWordWrap(True)
        root.addWidget(self.tracking_note)
        self._build_plate_controls(root)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Kind', 'Frame', 'Range [start,end)', 'Direction', 'Strength'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.select)
        root.addWidget(self.table, 1)
        self.timeline, self.status = QLabel(), QLabel()
        self.status.setWordWrap(True)
        self.timeline.setWordWrap(True)
        root.addWidget(self.timeline)
        root.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)

    def _build_plate_controls(self, root):
        self.plate_toggle = QPushButton('Show clean-plate assistance ▸')
        self.plate_toggle.setCheckable(True)
        root.addWidget(self.plate_toggle)
        self.plate_panel = QWidget()
        root.addWidget(self.plate_panel)
        root = QVBoxLayout(self.plate_panel)
        self.plate_toggle.toggled.connect(self._toggle_plate)
        self.plate_panel.hide()
        row = QHBoxLayout()
        self.plate_reference_path = QLineEdit()
        self.plate_reference_path.setPlaceholderText('Local empty-plate image or sequence directory')
        row.addWidget(self.plate_reference_path)
        self._button('Plate image…', lambda: self.choose_plate(False), row)
        self._button('Plate sequence…', lambda: self.choose_plate(True), row)
        self.plate_reference_frame = QSpinBox()
        self.plate_reference_frame.setRange(0, 99999)
        row.addWidget(QLabel('Reference absolute index'))
        row.addWidget(self.plate_reference_frame)
        self.plate_profile = QComboBox()
        for text, value in [('sRGB / BT.709', 'srgb'), ('Linear BT.709 / D65', 'linear_bt709_d65'),
                            ('Linear BT.2020 / D65', 'linear_bt2020_d65')]:
            self.plate_profile.addItem(text, value)
        self.plate_profile.setPlaceholderText('Choose reference colour…')
        self.plate_profile.setCurrentIndex(-1)
        row.addWidget(QLabel('Declare reference colour'))
        row.addWidget(self.plate_profile)
        root.addLayout(row)
        row = QHBoxLayout()
        self.plate_samples_path = QLineEdit()
        self.plate_samples_path.setPlaceholderText('Required native binary known-background mask (0 / maximum)')
        row.addWidget(self.plate_samples_path)
        self._button('Background samples…', self.choose_plate_samples, row)
        self.plate_whole_frame = QCheckBox('Assume entire source frame is known background')
        self.plate_whole_frame.toggled.connect(lambda checked: self.plate_samples_path.setEnabled(not checked))
        row.addWidget(self.plate_whole_frame)
        root.addLayout(row)
        row = QHBoxLayout()
        self.plate_alignment = QComboBox()
        self.plate_alignment.addItem('Declare translation', 'declared')
        self.plate_alignment.addItem('Estimate translation (heuristic)', 'estimate_translation')
        row.addWidget(self.plate_alignment)
        self.plate_dx, self.plate_dy = QDoubleSpinBox(), QDoubleSpinBox()
        for label, widget in [('Reference → source X px', self.plate_dx), ('Y px', self.plate_dy)]:
            widget.setRange(-32, 32)
            widget.setDecimals(2)
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.plate_alignment.currentIndexChanged.connect(self._plate_alignment_changed)
        self.plate_exposure = QCheckBox('Fit scalar exposure (gain 0.25–4)')
        row.addWidget(self.plate_exposure)
        self.plate_bg, self.plate_fg = QDoubleSpinBox(), QDoubleSpinBox()
        for label, widget, value in [('BG residual ≤', self.plate_bg, .04), ('FG residual ≥', self.plate_fg, .35)]:
            widget.setRange(.001, 100)
            widget.setDecimals(3)
            widget.setValue(value)
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        root.addLayout(row)
        row = QHBoxLayout()
        self._button('Preview clean plate', self.preview_plate, row)
        self.plate_mode = QComboBox()
        for text, mode in [('Rough guidance', 'guidance'), ('Spatial screen field', 'screen_field'),
                           ('Relative residual', 'residual'), ('Reference', 'reference')]:
            self.plate_mode.addItem(text, mode)
        row.addWidget(self.plate_mode)
        self.plate_inspect = self._button('Inspect plate view', self.inspect_plate, row)
        self.plate_accept = self._button('Accept inspected guidance', lambda: self.request('plate_accept'), row)
        self.plate_discard = self._button('Discard plate preview', lambda: self.request('plate_discard'), row)
        self.detach_guidance_button = self._button('Detach selected guidance origin', self.detach_guidance, row)
        self.detach_guidance_button.setEnabled(False)
        self.detach_guidance_button.setToolTip('Deliberately keep the existing mask as authored coverage/classes and remove its generated origin. Reference resources are no longer required; this does not recalculate or repaint the mask.')
        for button in (self.plate_inspect, self.plate_accept, self.plate_discard):
            button.setEnabled(False)
        root.addLayout(row)
        row = QHBoxLayout()
        self.plate_thumbnail = QLabel('Reference thumbnail appears after explicit preview')
        self.plate_thumbnail.setMaximumSize(192, 96)
        row.addWidget(self.plate_thumbnail)
        self.plate_note = QLabel('Optional single-frame assistance for image-sequence sources only. Select an explicit reference index and colour declaration. Samples must be known background; the whole-frame assumption can erase foreground. Translation only (32 px per axis), native geometry, bounded pixels/memory. Similar-coloured foreground and shadows are uncertain. Inspect rough guidance before deliberate acceptance as a final-only trimap. No model runs or neural channels; quality benefit unverified.')
        self.plate_note.setWordWrap(True)
        self.plate_note.setAlignment(Qt.AlignmentFlag.AlignTop)
        note_scroll = QScrollArea()
        note_scroll.setWidgetResizable(True)
        note_scroll.setMinimumHeight(56)
        note_scroll.setMaximumHeight(96)
        note_scroll.setWidget(self.plate_note)
        row.addWidget(note_scroll, 1)
        root.addLayout(row)

    def _toggle_plate(self, visible):
        self.plate_panel.setVisible(visible)
        self.plate_toggle.setText('Hide clean-plate assistance ▾' if visible else 'Show clean-plate assistance ▸')

    def _plate_alignment_changed(self):
        declared = self.plate_alignment.currentData() == 'declared'
        self.plate_dx.setEnabled(declared)
        self.plate_dy.setEnabled(declared)

    def choose_plate(self, directory):
        if self.pending is not None or self.canvas.dirty:
            self.status.setText('Finish current work and Commit/Clear draft before choosing a reference.')
            return
        path = (QFileDialog.getExistingDirectory(self, 'Choose local empty-plate sequence') if directory else
                QFileDialog.getOpenFileName(self, 'Choose local empty-plate image', '', 'Images (*.png *.exr *.jpg *.jpeg)')[0])
        if path:
            self.plate_reference_path.setText(path)
            self.plate_reference_frame.setValue(0)

    def choose_plate_samples(self):
        if self.pending is not None or self.canvas.dirty:
            self.status.setText('Finish current work and Commit/Clear draft before choosing samples.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Known-background samples: native binary mask', '', 'Masks (*.png *.exr)')
        if path:
            self.plate_samples_path.setText(path)
            self.plate_whole_frame.setChecked(False)

    def preview_plate(self):
        reference = self.plate_reference_path.text().strip()
        samples = self.plate_samples_path.text().strip()
        if not reference:
            self.status.setText('Choose a local reference image or sequence first.')
            return
        if self.plate_profile.currentData() is None:
            self.status.setText('Declare reference colour explicitly before previewing.')
            return
        if not samples and not self.plate_whole_frame.isChecked():
            self.status.setText('Choose native known-background samples or explicitly declare the whole-frame background assumption.')
            return
        if self.plate_bg.value() >= self.plate_fg.value():
            self.status.setText('Background threshold must be below foreground threshold, leaving an unknown band.')
            return
        self.request('plate_preview', frame=self.frame.value(), reference_path=Path(reference),
                     reference_frame=self.plate_reference_frame.value(), reference_mode='sequence' if Path(reference).is_dir() else 'single',
                     reference_profile=self.plate_profile.currentData(), alignment=self.plate_alignment.currentData(),
                     translation_xy=(self.plate_dx.value(), self.plate_dy.value()) if self.plate_alignment.currentData() == 'declared' else (0., 0.),
                     compensate_exposure=self.plate_exposure.isChecked(), samples_path=Path(samples) if samples and not self.plate_whole_frame.isChecked() else None,
                     whole_frame_samples=self.plate_whole_frame.isChecked(), background_threshold=self.plate_bg.value(), foreground_threshold=self.plate_fg.value())

    def detach_guidance(self):
        entry = self.selected()
        if entry is None or 'guidance_id' not in entry:
            self.status.setText('Select a generated guidance entry to detach its origin; the existing mask is retained.')
            return
        self.request('detach_guidance', identifier=entry['id'])

    def inspect_plate(self):
        self.request('plate_view', mode=self.plate_mode.currentData())

    def _present_plate(self, snapshot):
        preview = snapshot.get('plate_preview')
        idle = self.pending is None and not self.closing
        self.plate_inspect.setEnabled(preview is not None and idle)
        self.plate_discard.setEnabled(preview is not None and idle)
        self.plate_accept.setEnabled(preview is not None and preview.get('usable', False) and preview.get('inspected', False) and idle)
        if preview is None:
            self.plate_thumbnail.clear()
            self.plate_thumbnail.setText('No retained plate preview')
            return
        diagnostics = preview.get('report', {}).get('diagnostics', {})
        alignment, exposure = diagnostics.get('alignment', {}), diagnostics.get('exposure', {})
        counts = diagnostics.get('class_counts', {})
        request = preview.get('request', {})
        settings = request.get('settings', request)
        sample_scope = ('ENTIRE source frame assumed background; foreground may be erased' if request.get('whole_frame_samples') else 'artist binary background samples')
        self.plate_note.setText(
            f"Captured source frame {preview.get('source_frame')} (selection {preview.get('frame')}) → reference index {preview.get('reference_frame')} · {preview.get('reference_path')}\n"
            f"Reference colour {request.get('reference_profile')} · {sample_scope}\n"
            f"Usable: {preview.get('usable')} · rough guidance inspected: {preview.get('inspected')} · alignment {alignment.get('requested')}, shift {alignment.get('translation_reference_to_source_xy')}, phase response {alignment.get('phase_response')} · exposure gain {exposure.get('gain')}\n"
            f"Background {counts.get('background', 0)} / unknown {counts.get('unknown', 0)} / foreground {counts.get('foreground', 0)} pixels. "
            f"{diagnostics.get('reason') or 'Conservative candidate; inspect before acceptance.'}\n"
            f"Captured thresholds BG {settings.get('background_threshold')} / FG {settings.get('foreground_threshold')}; current controls prepare a new preview. One native frame; translation ≤32 px per axis; pixel/estimated-memory budgets enforced. Translation/photometric heuristics only; similar-coloured foreground may become background. No inference or verified quality benefit.")

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def selected(self) -> dict[str, Any] | None:
        selection = self.table.selectionModel()
        selected = selection.selectedRows() if selection is not None else []
        row = selected[0].row() if len(selected) == 1 else -1
        return self.entries[row] if 0 <= row < len(self.entries) else None

    def add_tracking_anchor(self):
        entry = self.selected()
        if self.pending is not None or entry is None or entry['kind'] not in {'keep', 'remove', 'matte'}:
            self.status.setText('Select a committed coverage entry after the current operation finishes.')
            return
        if entry['id'] not in self.tracking_anchors:
            self.tracking_anchors.append(entry['id'])
        self.tracking_note.setText('Artist anchor frames: ' + str([e['frame'] for e in self.entries if e['id'] in self.tracking_anchors]) + '. Same kind/strength and contiguous entry order required. Use Start/End/Direction, then Preview track; inspect before accepting.')

    def cancel_operation(self):
        if self.pending is not None:
            self.service.cancel_current.set()
            self.status.setText('Cancellation requested; waiting for the owned operation. Native flow/analysis/decoding may finish first.')

    def clear_tracking_anchors(self):
        if self.pending is None:
            self.tracking_anchors.clear()
            self.tracking_note.setText('Anchor list cleared; Preview track uses the selected authored entry. Existing accepted tracks remain until cleared or an authored edit invalidates them.')

    def preview_tracking(self):
        entry = self.selected()
        anchors = self.tracking_anchors.copy() or ([entry['id']] if entry is not None else [])
        self.request('track_preview', anchor_ids=anchors, span=[self.start.value(), self.end.value()],
                     direction=self.direction.currentText(), unreliable_policy=self.tracking_policy.currentText())

    def inspect_tracking(self):
        if self.request('track_view', frame=self.frame.value()):
            self.view_requested_frame = self.frame.value()

    def parameters(self):
        return {'kind': self.kind.currentText(), 'frame': self.frame.value(), 'strength': self.strength.value(),
                'direction': self.direction.currentText(), 'span': [self.start.value(), self.end.value()]}

    def request(self, action, *, allow_draft=False, **payload):
        if self.closing:
            self.status.setText('Editor is closing; no new operations accepted.')
            return False
        if self.snapshot is None or self.pending is not None or not self.service.ready.is_set() or self.service.done.is_set():
            self.status.setText('Wait for the current editor operation to finish.')
            return False
        if self.canvas.dirty and not allow_draft:
            self.status.setText('Commit or Clear draft before changing frame/history or saving.')
            return False
        try:
            self.pending = self.service.request(action, **payload)
            self.canvas.painting = False
            self.status.setText('Working: ' + action)
            return True
        except (RuntimeError, queue.Full) as exc:
            self.status.setText(str(exc))
            return False

    def load_frame(self):
        entry = self.selected()
        if self.request('view', frame=self.frame.value(), identifier=entry['id'] if entry else None):
            self.view_requested_frame = self.frame.value()

    def commit(self):
        if self.displayed_role != 'authored':
            self.status.setText('Derived preview is read-only. Load an authored frame or choose New mask before painting/committing.')
            return
        entry = self.selected()
        if entry is not None and entry['kind'] == 'trimap':
            self.status.setText('Imported trimap classes cannot be binary painted. Choose New mask for a separate coverage entry.')
            return
        if self.canvas.draft is None or self.displayed_frame != self.frame.value():
            self.status.setText('Load this frame before painting/committing.')
            return
        self.request('brush', allow_draft=True, mask=self.canvas.draft.copy(), **self.parameters())

    def clear_draft(self):
        if self.pending is None:
            self.canvas.clear_draft()
            self.status.setText('Draft cleared. Committed edits remain.')

    def import_mask(self):
        if self.pending is not None or self.canvas.dirty:
            self.status.setText('Finish current work and Commit/Clear draft before importing.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Import native grayscale/RGBA coverage', '', 'Masks (*.png *.exr *.jpg *.jpeg)')
        if path:
            self.request('import', path=Path(path), **self.parameters())

    def import_trimap(self):
        if self.pending is not None or self.canvas.dirty:
            self.status.setText('Finish current work and Commit/Clear draft before importing.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Import native trimap: BG 0 / UNKNOWN midpoint / FG maximum', '', 'Trimaps (*.png *.exr)')
        if path:
            parameters = self.parameters()
            parameters.pop('kind')
            self.request('import_trimap', path=Path(path), **parameters)

    def new_mask(self):
        if self.pending is not None or self.canvas.dirty:
            self.status.setText('Finish current work and Commit/Clear draft first.')
            return
        self.table.setCurrentCell(-1, -1)
        self.table.clearSelection()
        for widget in (self.kind, self.strength, self.direction, self.start, self.end):
            widget.setEnabled(True)
        if self.kind.currentIndex() < 0:
            self.kind.setCurrentIndex(0)
        self.canvas.overlay_kind = 'coverage'
        self.canvas.saved_overlay = None
        self.canvas.update()
        self.start.setValue(self.frame.value())
        self.end.setValue(self.frame.value() + 1)
        if self.displayed_role == 'plate' or self.displayed_frame != self.frame.value():
            # Plate inspection may show reference/diagnostic pixels. Painting
            # becomes available only after the actual requested source arrives.
            self.canvas.painting = False
            if self.request('view', frame=self.frame.value(), identifier=None):
                self.view_requested_frame = self.frame.value()
            return
        self.displayed_role = 'authored'
        self.canvas.painting = not self.closing

    def save(self):
        self.request('save')

    def edit(self):
        entry = self.selected()
        if entry is None:
            self.status.setText('Select an entry first.')
            return
        changes: dict[str, Any] = {'frame': self.frame.value()}
        if entry['kind'] != 'reset':
            parameters = self.parameters()
            changes.update({key: value for key, value in parameters.items() if key != 'span'})
            if self.presented_entry == entry['id'] and self.strength.value() == self.presented_strength:
                changes.pop('strength', None)
            changes['range'] = parameters['span']
            if entry['kind'] == 'trimap':
                changes['kind'] = 'trimap'
        self.request('edit', identifier=entry['id'], changes=changes)

    def remove(self):
        entry = self.selected()
        if entry is not None:
            self.request('remove', identifier=entry['id'])

    def move_entry(self, delta):
        entry = self.selected()
        if entry is not None:
            self.request('move', identifier=entry['id'], delta=delta)

    def select(self):
        entry = self.selected()
        for button in self.selection_buttons:
            button.setEnabled(entry is not None and self.pending is None and not self.closing)
        self.detach_guidance_button.setEnabled(entry is not None and 'guidance_id' in entry and self.pending is None and not self.closing)
        self.presented_entry = entry['id'] if entry is not None else None
        self.presented_strength = None
        reset = entry is not None and entry['kind'] == 'reset'
        trimap = entry is not None and entry['kind'] == 'trimap'
        for widget in (self.kind, self.strength, self.direction, self.start, self.end):
            widget.setEnabled(not reset)
        self.kind.setEnabled(not reset and not trimap)
        self.kind.setPlaceholderText('Imported trimap')
        self.canvas.painting = self.displayed_role == 'authored' and not trimap and self.pending is None and not self.closing
        if entry is not None:
            self.frame.setValue(entry['frame'])
            if not reset:
                if trimap:
                    self.kind.setCurrentIndex(-1)
                else:
                    self.kind.setCurrentText(entry['kind'])
                self.strength.setValue(entry['strength'])
                self.presented_strength = self.strength.value()
                self.direction.setCurrentText(entry['direction'])
                self.start.setValue(entry['range'][0])
                self.end.setValue(entry['range'][1])

    def populate(self, snapshot, action):
        old = self.selected()
        selected = old['id'] if old is not None else None
        self.snapshot = snapshot
        data = snapshot['document']
        self.entries = data['entries']
        self.tracking_anchors = [key for key in self.tracking_anchors if any(e['id'] == key for e in self.entries)]
        if action in ('brush', 'import', 'import_trimap', 'reset', 'plate_accept') and self.entries:
            selected = self.entries[-1]['id']
        count = data['source']['frame_count']
        self.frame.setMaximum(count - 1)
        self.start.setMaximum(count - 1)
        self.end.setMaximum(count)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.entries))
        self.table.setCurrentCell(-1, -1)
        for row, entry in enumerate(self.entries):
            for col, value in enumerate([entry['kind'], entry['frame'], entry.get('range', ''),
                                         entry.get('direction', ''), entry.get('strength', '')]):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
            if entry['id'] == selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self.select()
        self.timeline.setText(f"Shots: {data['shots']} · Last edit invalidated whole shots: {snapshot['invalidated']}\nCommitted recovery snapshot: {snapshot['checkpoint_path']} · Accepted tracks: {len(data.get('tracks', []))}")
        preview = snapshot.get('tracking_preview')
        if preview is not None:
            events = ', '.join(f"{row['frame']}:{row['status']}" for row in preview['frames'])
            self.tracking_note.setText(f"Captured tracking request: {preview['request']} · Accept uses this inspected snapshot.\nTracking preview · complete={preview['complete']} · inspected={preview['inspected']} · charged bytes={preview['bytes_charged']}\n{events}\nReliability/occlusion are heuristic; inspect selected frame before explicit acceptance. Missing masks require another anchor or a shorter range.")
        self._present_plate(snapshot)
        if snapshot.get('tracking_cleanup_pending'):
            self.timeline.setText(self.timeline.text() + '\nPrivate preview cleanup pending: ' + str(snapshot['tracking_cleanup_pending']))
            self.timeline.setToolTip('\n'.join(snapshot.get('tracking_cleanup_warnings', [])))

    def poll(self):
        while True:
            try:
                event = self.service.events.get_nowait()
            except queue.Empty:
                break
            self.pending = None
            self.canvas.painting = self.displayed_role == 'authored' and not self.closing
            if event['kind'] == 'error':
                if event.get('snapshot') is not None:
                    self.populate(event['snapshot'], 'error')
                self.select()
                extra = (' · Private preview cleanup pending; see timeline warnings.'
                         if event.get('snapshot', {}).get('tracking_cleanup_pending') else '')
                if event.get('snapshot_error'):
                    extra += ' · Failure snapshot unavailable: ' + event['snapshot_error']
                self.status.setText(event['error'] + extra)
                continue
            action = event.get('action', 'ready')
            if action in ('view', 'track_view', 'plate_view'):
                self.snapshot = event['snapshot']
            else:
                self.populate(event['snapshot'], action)
            if action in ('view', 'track_view', 'plate_view'):
                display, overlay = event['view']
                self.displayed_role = 'plate' if action == 'plate_view' else ('tracking' if action == 'track_view' else 'authored')
                self.canvas.set_frame(display.pixels, display.width, display.height, overlay,
                                      overlay_kind=event.get('overlay_kind', 'coverage'))
                self.canvas.painting = action == 'view' and event.get('overlay_kind') != 'trimap' and not self.closing
                self.displayed_frame = (event['snapshot'].get('plate_preview') or {}).get('frame') if action == 'plate_view' else self.view_requested_frame
                role = 'derived clean-plate preview; heuristic guidance, read-only' if action == 'plate_view' else ('derived tracking preview; heuristic guidance, not neural alpha' if action == 'track_view' else 'authored overlay only')
                self.frame_note.setText(f"Displayed selected-frame {self.displayed_frame} (source start {event['snapshot']['document']['source']['start_frame']}) · {display.width}×{display.height} · {role}")
            elif action in ('brush', 'import', 'import_trimap'):
                self.canvas.clear_draft()
            self._present_plate(event['snapshot'])
            if event.get('plate_reference') is not None:
                thumbnail = event['plate_reference']
                pixels = thumbnail.pixels
                image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
                self.plate_thumbnail.setPixmap(QPixmap.fromImage(image).scaled(192, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            if action == 'save':
                self.saved.emit(self.job, event['snapshot']['saved_path'])
                self.status.setText('Immutable snapshot saved: ' + event['snapshot']['saved_path'])
            else:
                self.status.setText('Ready · ' + action)
            if action in ('ready', 'brush', 'import', 'import_trimap', 'edit', 'undo', 'redo', 'remove', 'move', 'plate_accept'):
                self.load_frame()
        if self.closing and self.service.done.is_set():
            self.close()

    def closeEvent(self, event: QCloseEvent):
        if self.canvas.dirty:
            self.status.setText('Commit or Clear draft before closing; committed edits have on-disk recovery snapshots.')
            event.ignore()
            return
        self.closing = True
        self.canvas.painting = False
        self.service.stop()
        if self.service.done.is_set():
            self.timer.stop()
            event.accept()
        else:
            event.ignore()
