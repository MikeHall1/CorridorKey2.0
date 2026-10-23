"""Explicit completed-render cache preparation with owned background I/O."""
from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from .ofx_cache_task import CachePreparation
from .ofx_record import cache_spec
from .worker_protocol import decode_frame, encode_frame

STAGES = {
    'ofx_save_record': 'Saving selected render settings',
    'ofx_load_image_adapters': 'Loading image readers',
    'ofx_validate': 'Checking completed outputs',
    'ofx_hash_exports': 'Checking export files',
    'ofx_convert_frames': 'Writing cache frames',
    'ofx_verify_exports': 'Verifying original export files',
    'ofx_publish': 'Finishing cache',
}


class OfxCachePanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Prepare cached effect')
        self.resize(820, 400)
        self.record: dict[str, Any] | None = None
        self.task: CachePreparation | None = None
        self.running = False
        self.close_when_done = False
        layout = QVBoxLayout(self)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.selection = QLabel('Select a completed render in the queue, then open this panel.')
        self.selection.setWordWrap(True)
        layout.addWidget(self.selection)
        destination = QHBoxLayout()
        self.parent_path = QLineEdit()
        self.parent_path.setPlaceholderText('Existing destination folder')
        destination.addWidget(self.parent_path)
        self.destination_button = QPushButton('Destination…')
        self.destination_button.clicked.connect(self.choose_destination)
        destination.addWidget(self.destination_button)
        layout.addLayout(destination)
        actions = QHBoxLayout()
        self.prepare_button = QPushButton('Prepare new cache')
        self.prepare_button.clicked.connect(self.prepare)
        self.cancel_button = QPushButton('Cancel preparation')
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.prepare_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        self.cache_path = QLineEdit()
        self.cache_path.setReadOnly(True)
        self.cache_path.setPlaceholderText('Completed cache path appears here')
        layout.addWidget(self.cache_path)
        self.copy_button = QPushButton('Copy cache path')
        self.copy_button.clicked.connect(self.copy_path)
        layout.addWidget(self.copy_button)
        self.status = QLabel('No cache preparation started. Opening this panel submits no render.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._binding_note()
        self._controls()

    def _source_bound(self) -> bool:
        settings = self.record.get('requested_settings') if self.record is not None else None
        return isinstance(settings, dict) and settings.get('source_witness') is True

    def _binding_note(self):
        if self._source_bound():
            self.note.setText('Source-bound preparation: the experimental effect will verify exact opaque scene-linear source pixels across the full selected range before rendering. Match the host frame offset and colour pipeline. This strict check can be slow; it does not validate managed colour or Resolve compatibility.')
        else:
            self.note.setText('Prepare completed EXR foreground and alpha for the experimental cached effect. Match the source, host frame offset and scene-linear colour assumptions in Resolve. This render has no producer source witness; automatic host-source validation is unavailable.')

    @property
    def io_busy(self) -> bool:
        return self.task is not None and self.task.busy

    def _controls(self):
        self.prepare_button.setEnabled(self.record is not None and not self.io_busy and not self.close_when_done)
        self.cancel_button.setEnabled(self.running)
        self.copy_button.setEnabled(bool(self.cache_path.text()) and not self.running)
        self.parent_path.setEnabled(not self.io_busy and not self.close_when_done)
        self.destination_button.setEnabled(not self.io_busy and not self.close_when_done)

    def load_record(self, record: dict[str, Any]):
        if self.io_busy or self.close_when_done:
            raise ValueError('Wait for current cache preparation before replacing its selection.')
        self.reject_selection('Checking selected completed render…')
        snapshot = decode_frame(encode_frame(record))
        spec = cache_spec(snapshot)
        self.record = snapshot
        self._binding_note()
        source_rgb = snapshot['diagnostics']['output_contracts']['fg'].get('foreground_mode') == 'source_rgb_draft'
        mode = ('Source-RGB draft' if source_rgb else 'Approximate spatial draft') if spec.draft else 'Recorded foreground'
        execution = snapshot['diagnostics'].get('execution')
        if isinstance(execution, dict) and execution.get('neural_inference') is False:
            mode = f'Test-only foreground · {mode}' if spec.draft else 'Test-only foreground'
        output = str(snapshot.get('output_dir', ''))[:4096]
        source_label = Path(spec.source).name[:120] or spec.source[:120]
        render_label = Path(output).name[:80]
        self.selection.setText(f'{source_label} · Render {render_label}\n{spec.width}×{spec.height} · {spec.fps:g} fps · source range [{spec.start}, {spec.start+spec.count})\n{mode}. Cache frame0 corresponds to source frame {spec.start}; choose its host frame explicitly.')
        self.selection.setToolTip(f'Source: {spec.source}\nRender: {output}')
        self.cache_path.clear()
        self.status.setText('Completed render selected. Choose a destination and Prepare; queued settings remain unchanged.')
        self._controls()

    def reject_selection(self, message: str):
        if not self.io_busy:
            self.record = None
            self.cache_path.clear()
            self.selection.setText('No completed render selected.')
            self._binding_note()
        self.status.setText(message[:4096])
        self._controls()

    def choose_destination(self):
        if self.io_busy:
            return
        path = QFileDialog.getExistingDirectory(self, 'Choose cache destination parent')
        if path:
            self.parent_path.setText(path)

    def prepare(self):
        if self.record is None or self.io_busy or self.close_when_done:
            self.status.setText('Select a completed render and wait for current preparation.')
            return
        parent = self.parent()
        if parent is not None and getattr(parent, 'closing', False):
            self.status.setText('Workspace is closing; wait for owned I/O cleanup.')
            return
        if not self.parent_path.text().strip():
            self.status.setText('Choose an existing destination folder first.')
            return
        try:
            self.task = CachePreparation(self.record, Path(self.parent_path.text()))
            self.running = True
            self.cache_path.clear()
            self.status.setText('Preparing cache off the interface thread…')
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def cancel(self):
        if self.task is not None and self.io_busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for file/native I/O cleanup…')

    def copy_path(self):
        if self.cache_path.text() and not self.running:
            QApplication.clipboard().setText(self.cache_path.text())

    def poll(self):
        if self.task is not None:
            if self.running:
                event = self.task.progress()
                if event is not None and not self.task.cancellation.cancelled:
                    phase = STAGES.get(event.stage, 'Preparing cache')
                    self.status.setText(f'{phase} · {event.completed}/{event.total}' if event.total is not None else phase + '…')
            try:
                result = self.task.results.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                self.running = False
                if result.status == 'complete' and result.cache is not None:
                    self.cache_path.setText(str(result.cache))
                    binding = ' The effect checks the full witnessed source range.' if self._source_bound() else ''
                    self.status.setText('Cache prepared. Copy its path into the cached effect and align the host frame/colour settings. No render submitted.' + binding)
                else:
                    location = f'\nPreparation folder: {result.workspace}' if result.workspace is not None else ''
                    self.status.setText(f'{result.status}: {result.error}{location}')
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
