"""Native correction editor dialog on the existing history/raster pipeline."""
from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
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
        for text, callback in [('Add selected anchor', self.add_tracking_anchor),
                               ('Clear anchor list', self.clear_tracking_anchors),
                               ('Preview track', self.preview_tracking),
                               ('Inspect tracked frame', self.inspect_tracking),
                               ('Accept inspected track', lambda: self.request('track_accept')),
                               ('Discard preview', lambda: self.request('track_discard')),
                               ('Clear accepted tracks', lambda: self.request('track_clear')),
                               ('Cancel operation', self.cancel_operation)]:
            self._button(text, callback, tracking)
        root.addLayout(tracking)
        self.tracking_note = QLabel('Sequence tracking: choose same-kind/strength contiguous authored anchors; Start/End and Direction set tracking range. Preview changes no history. Inspect a retained frame before explicit acceptance. Cuts/resets stop the range. Video/trimap tracking unavailable.')
        self.tracking_note.setWordWrap(True)
        root.addWidget(self.tracking_note)
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
            self.status.setText('Cancellation requested; waiting for the owned operation. Native flow/decoding may finish first.')

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
        if self.displayed_role == 'tracking':
            self.status.setText('Tracked preview is read-only. Load an authored frame or choose New mask before painting/committing.')
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
        self.displayed_role = 'authored'
        self.canvas.painting = not self.closing
        self.canvas.update()
        self.start.setValue(self.frame.value())
        self.end.setValue(self.frame.value() + 1)

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
        self.presented_entry = entry['id'] if entry is not None else None
        self.presented_strength = None
        reset = entry is not None and entry['kind'] == 'reset'
        trimap = entry is not None and entry['kind'] == 'trimap'
        for widget in (self.kind, self.strength, self.direction, self.start, self.end):
            widget.setEnabled(not reset)
        self.kind.setEnabled(not reset and not trimap)
        self.kind.setPlaceholderText('Imported trimap')
        self.canvas.painting = self.displayed_role != 'tracking' and not trimap and self.pending is None and not self.closing
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
        if action in ('brush', 'import', 'import_trimap', 'reset') and self.entries:
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
            self.canvas.painting = self.displayed_role != 'tracking' and not self.closing
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
            if action in ('view', 'track_view'):
                self.snapshot = event['snapshot']
            else:
                self.populate(event['snapshot'], action)
            if action in ('view', 'track_view'):
                display, overlay = event['view']
                self.displayed_role = 'tracking' if action == 'track_view' else 'authored'
                self.canvas.set_frame(display.pixels, display.width, display.height, overlay,
                                      overlay_kind=event.get('overlay_kind', 'coverage'))
                self.canvas.painting = action != 'track_view' and event.get('overlay_kind') != 'trimap' and not self.closing
                self.displayed_frame = self.view_requested_frame
                role = 'derived tracking preview; heuristic guidance, not neural alpha' if action == 'track_view' else 'authored overlay only'
                self.frame_note.setText(f"Displayed selected-frame {self.displayed_frame} (source start {event['snapshot']['document']['source']['start_frame']}) · {display.width}×{display.height} · {role}")
            elif action in ('brush', 'import', 'import_trimap'):
                self.canvas.clear_draft()
            if action == 'save':
                self.saved.emit(self.job, event['snapshot']['saved_path'])
                self.status.setText('Immutable snapshot saved: ' + event['snapshot']['saved_path'])
            else:
                self.status.setText('Ready · ' + action)
            if action in ('ready', 'brush', 'import', 'import_trimap', 'edit', 'undo', 'redo', 'remove', 'move'):
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
