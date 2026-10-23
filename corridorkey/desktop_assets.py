"""Deliberate local asset selection; no automatic download or model load."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from .asset_inventory import AssetSelection, inspect_assets, inspect_backend
from .desktop_asset_audit import AssetAuditPanel
from .desktop_asset_download import AssetDownloadPanel


class AssetsPanel(QDialog):
    use_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Local model assets')
        self.resize(960, 700)
        self.io_thread: threading.Thread | None = None
        self.results: queue.Queue[tuple[AssetSelection, dict[str, Any]]] = queue.Queue(maxsize=1)
        self.checked: AssetSelection | None = None
        self.ready = False
        self.close_when_done = False
        self.audit_panel = AssetAuditPanel(self)
        self.download_panel = AssetDownloadPanel(self)
        self.download_panel.selection_ready.connect(self.select_downloaded)
        layout = QVBoxLayout(self)
        note = QLabel('Choose a local asset tree containing weights/ and vfms/ in the existing layout, and optionally override the runtime checkpoint. Check reads metadata and probes backend availability without deserializing models. Stop the worker and wait before Use; selections affect future normal/recovery work in this app session. Existing batch weights override the selected runtime default. Projects do not save asset selection. No automatic download or CPU/MPS fallback.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.root = QLineEdit(str(Path(__file__).resolve().parents[1]))
        self.weights = QLineEdit()
        self.weights.setPlaceholderText('Blank: preferred safetensors, then legacy .pt in selected tree')
        form = QFormLayout()
        form.addRow('Local asset tree', self.root)
        form.addRow('Runtime checkpoint override', self.weights)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        for text, callback in [('Asset tree…', self.choose_root), ('Checkpoint…', self.choose_weights),
                               ('Check local assets/backend', self.inspect), ('Use for future jobs', self.use)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.audit_button = QPushButton('Audit selected files…')
        self.audit_button.clicked.connect(self.open_audit)
        layout.addWidget(self.audit_button)
        self.download_button = QPushButton('Download model assets…')
        self.download_button.clicked.connect(self.open_download)
        layout.addWidget(self.download_button)
        self.status = QLabel('No local asset check performed.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.root.textChanged.connect(self.sync_audit_selection)
        self.weights.textChanged.connect(self.sync_audit_selection)

    @property
    def presence_busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    @property
    def io_busy(self):
        return self.presence_busy or self.audit_busy or self.download_busy

    @property
    def audit_busy(self):
        panel = getattr(self, 'audit_panel', None)
        return panel is not None and panel.io_busy

    @property
    def download_busy(self):
        panel = getattr(self, 'download_panel', None)
        return panel is not None and panel.io_busy

    def sync_audit_selection(self):
        # Lexical paths only. Resolution, hashing and structure checks are owned
        # by the child task rather than the Qt event thread.
        self.audit_panel.source_root.setText(self.root.text())
        self.audit_panel.weights_path.setText(self.weights.text())
        self.audit_panel.invalidate_report()
        self.download_panel.invalidate_result()

    def open_download(self):
        if self.io_busy or self.close_when_done or getattr(self.parent(), 'closing', False):
            self.status.setText('Wait for the current asset operation or closing work.')
            return
        self.download_panel.show()
        self.download_panel.raise_()
        self.download_panel.load_catalog()

    def select_downloaded(self, selection: AssetSelection):
        if self.io_busy or self.close_when_done or getattr(self.parent(), 'closing', False):
            return
        self.checked, self.ready = None, False
        self.root.setText(str(selection.root))
        self.weights.setText(str(selection.weights) if selection.weights else '')
        self.status.setText('Downloaded paths filled. Check local assets/backend, stop worker, then Use explicitly. No worker activated or job submitted.')

    def open_audit(self):
        if self.io_busy or self.close_when_done or getattr(self.parent(), 'closing', False):
            self.status.setText('Wait for the current asset operation or closing work.')
            return
        self.sync_audit_selection()
        self.audit_panel.show()
        self.audit_panel.raise_()

    def selection(self) -> AssetSelection:
        if not self.root.text().strip():
            raise ValueError('Choose a local asset tree.')
        return AssetSelection(Path(self.root.text()).resolve(), Path(self.weights.text()).resolve() if self.weights.text().strip() else None)

    def choose_root(self):
        path = QFileDialog.getExistingDirectory(self, 'Select existing local asset tree')
        if path:
            self.root.setText(path)

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select local runtime checkpoint', '', 'Checkpoints (*.safetensors *.pt);;All files (*)')
        if path:
            self.weights.setText(path)

    def inspect(self):
        try:
            if self.io_busy or self.close_when_done or getattr(self.parent(), 'closing', False):
                raise ValueError('Wait for the current local check.')
            selection = self.selection()
            self.checked, self.ready = None, False
            self.io_thread = threading.Thread(target=self._inspect, args=(selection,), daemon=True)
            self.io_thread.start()
            self.status.setText('Checking local metadata and backend; no model loading or download…')
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def _inspect(self, selection: AssetSelection):
        try:
            data = inspect_assets(selection)
            data['backend'] = inspect_backend()
        except BaseException as exc:
            logging.getLogger(__name__).exception('Local asset inspection failed')
            data = {'assets_present': False, 'error': f'{type(exc).__name__}: {exc}'[:4096]}
        self.results.put((selection, data))

    def poll(self):
        self.audit_panel.poll()
        self.download_panel.poll()
        self._poll_presence()
        self.audit_button.setEnabled(not self.io_busy and not self.close_when_done)
        self.download_button.setEnabled(not self.io_busy and not self.close_when_done)
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def _poll_presence(self):
        try:
            selection, data = self.results.get_nowait()
        except queue.Empty:
            return
        self.checked, self.ready = selection, data['assets_present']
        ordered = {'backend': data.get('backend'), **data}
        self.details.setPlainText(json.dumps(ordered, indent=2))
        summary = 'Local check complete. Presence does not prove model validity; stop worker before Use.' if self.ready else 'Required assets missing/invalid. Nothing selected or downloaded.'
        backend = data.get('backend', {}).get('status', 'Backend NOT VERIFIED')
        self.status.setText(summary + ' ' + backend)

    def use(self):
        try:
            if self.io_busy or not self.ready or self.checked != self.selection():
                raise ValueError('Check the current local paths before Use.')
            self.use_requested.emit(self.checked)
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def closeEvent(self, event: QCloseEvent):
        self.audit_panel.close()
        self.download_panel.close()
        if self.io_busy:
            self.close_when_done = True
            event.ignore()
        else:
            event.accept()
