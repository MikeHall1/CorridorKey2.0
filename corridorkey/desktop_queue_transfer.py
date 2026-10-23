"""Explicit archived queue export/import with owned background I/O."""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
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

from .control import CancellationToken, CancelledError

MAX_REPORT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class TransferResult:
    status: str
    raw: bytes | None
    error: str | None


class QueueTransferTask:
    def __init__(self, operation: str, source: str, parent: str):
        if operation not in {'export', 'import'}:
            raise ValueError('Choose export or import.')
        self.operation = operation
        self.source = source
        self.parent = parent
        self.cancellation = CancellationToken()
        self.results: queue.Queue[TransferResult] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name='queue-history-transfer', daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Queue transfer cancelled by user')

    def _run(self):
        result = TransferResult('failed', None, 'Unexpected queue transfer worker termination')
        try:
            from . import queue_transfer

            check = self.cancellation.check
            check()
            # Preserve lexical paths for the core's symlink/admission policy;
            # canonicalization and all filesystem checks belong to this worker.
            source = Path(self.source)
            parent = Path(self.parent)
            check()
            function = queue_transfer.export_history if self.operation == 'export' else queue_transfer.import_history
            report = function(source, parent, check=check)
            # The core reports any committed or partial destination. Cancellation
            # after publication cannot replace that authoritative result.
            raw = json.dumps(report, allow_nan=False, separators=(',', ':')).encode('utf-8')
            if len(raw) > MAX_REPORT_BYTES:
                raise ValueError('Queue transfer report exceeds the 4 MiB interface limit.')
            result = TransferResult('complete', raw, None)
        except CancelledError as exc:
            result = TransferResult('cancelled', None, str(exc)[:4096])
        except BaseException as exc:
            logging.getLogger(__name__).exception('Queue transfer failed')
            result = TransferResult('failed', None, f'{type(exc).__name__}: {exc}'[:4096])
        finally:
            self.results.put(result)
            self.done.set()


class QueueTransferPanel(QDialog):
    history_ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Queue history export / import')
        self.resize(850, 540)
        self.task: QueueTransferTask | None = None
        self.running = False
        self.close_when_done = False
        self.imported_path: str | None = None
        layout = QVBoxLayout(self)
        note = QLabel('Export already archived queue history, or import an exported container into fresh archived history. Choose an existing destination parent; each operation creates a unique child without overwriting existing history. Managed batch snapshots travel with history; external footage, models, corrections, recovery caches and rendered outputs stay at their original paths. Import does not prove output validity or another machine’s path compatibility. Nothing activates, renders or resumes automatically.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.operation = QComboBox()
        self.operation.addItems(['Export archived history', 'Import exported history'])
        form.addRow('Operation', self.operation)
        self.source_path = QLineEdit()
        self.source_button = QPushButton('Source…')
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_path)
        source_row.addWidget(self.source_button)
        self.source_label = QLabel('Existing archived queue history')
        form.addRow(self.source_label, source_row)
        self.destination_parent = QLineEdit()
        self.parent_button = QPushButton('Destination parent…')
        parent_row = QHBoxLayout()
        parent_row.addWidget(self.destination_parent)
        parent_row.addWidget(self.parent_button)
        form.addRow('Existing destination parent', parent_row)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.run_button = QPushButton('Export to new container')
        self.cancel_button = QPushButton('Cancel operation')
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        self.result_path = QLineEdit()
        self.result_path.setReadOnly(True)
        self.result_path.setPlaceholderText('Actual new or partial destination appears here')
        layout.addWidget(self.result_path)
        self.status = QLabel('Choose source and destination. No history transfer started.')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.use_button = QPushButton('Use imported history')
        layout.addWidget(self.use_button)
        self.operation.currentIndexChanged.connect(self.mode_changed)
        self.source_button.clicked.connect(self.choose_source)
        self.parent_button.clicked.connect(self.choose_parent)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.use_button.clicked.connect(self.use_imported)
        self.source_path.textChanged.connect(self.invalidate_imported)
        self.destination_parent.textChanged.connect(self.invalidate_imported)
        self._controls()

    @property
    def io_busy(self) -> bool:
        return self.task is not None and self.task.busy

    def _controls(self):
        parent = self.parent()
        parent_running = parent is not None and getattr(parent, 'history_busy', False)
        idle = not self.io_busy and not self.close_when_done and not parent_running
        for widget in (self.operation, self.source_path, self.source_button, self.destination_parent,
                       self.parent_button, self.run_button):
            widget.setEnabled(idle)
        self.cancel_button.setEnabled(self.running)
        self.use_button.setEnabled(idle and self.imported_path is not None)

    def invalidate_imported(self):
        self.imported_path = None
        self._controls()

    def mode_changed(self):
        self.invalidate_imported()
        is_import = self.operation.currentIndex() == 1
        self.source_label.setText('Existing exported history' if is_import else 'Existing archived queue history')
        self.run_button.setText('Import to fresh history' if is_import else 'Export to new container')
        self.source_path.clear()

    def set_history(self, path: str):
        if not self.io_busy and not self.close_when_done and self.operation.currentIndex() == 0:
            self.source_path.setText(path)

    def choose_source(self):
        if not self.io_busy and not self.close_when_done:
            path = QFileDialog.getExistingDirectory(self, 'Choose existing exported history' if self.operation.currentIndex() == 1 else 'Choose existing archived history')
            if path:
                self.source_path.setText(path)

    def choose_parent(self):
        if not self.io_busy and not self.close_when_done:
            path = QFileDialog.getExistingDirectory(self, 'Choose existing destination parent')
            if path:
                self.destination_parent.setText(path)

    def start(self):
        if self.io_busy or self.close_when_done:
            return
        parent = self.parent()
        if parent is not None and (getattr(parent, 'history_busy', False) or getattr(parent, 'close_when_done', False)):
            self.status.setText('Wait for history I/O and closing work before starting export or import.')
            return
        if parent is not None:
            owner = parent.parent()
            if owner is not None and getattr(owner, 'closing', False):
                self.status.setText('Workspace is closing; wait for owned I/O cleanup.')
                return
        source, destination = self.source_path.text().strip(), self.destination_parent.text().strip()
        if not source or not destination:
            self.status.setText('Choose an existing source folder and destination parent.')
            return
        operation = 'import' if self.operation.currentIndex() == 1 else 'export'
        self.imported_path = None
        self.result_path.clear()
        self.details.clear()
        try:
            self.task = QueueTransferTask(operation, source, destination)
            self.running = True
            self.status.setText('Importing into fresh archived history…' if operation == 'import' else 'Exporting and verifying archived history…')
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def cancel(self):
        if self.task is not None and self.io_busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual copy and verification cleanup. Any partial destination will be reported.')

    def use_imported(self):
        if self.imported_path is not None and not self.io_busy and not self.close_when_done:
            self.history_ready.emit(self.imported_path)
            self.status.setText('Imported archived history selected for reading. No activation, recovery or rendering submitted.')

    def poll(self):
        if self.task is not None:
            try:
                result = self.task.results.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                self.running = False
                if result.raw is not None:
                    report = json.loads(result.raw)
                    self._show_report(report)
                else:
                    self.status.setText(f'{result.status}: {result.error}'[:4096])
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def _show_report(self, report: dict[str, Any]):
        status = str(report.get('status', 'failed'))
        destination = report.get('destination')
        self.result_path.setText(destination if isinstance(destination, str) else '')
        usable = report.get('usable') is True

        def count(key: str) -> str:
            value = report.get(key)
            return str(value) if type(value) is int and value >= 0 else 'Unknown'

        lines = [f'Result: {status}',
                 f'Destination ready to use: {"Yes" if usable else "No"}',
                 f'Copied files: {count("copied_files")}',
                 f'Copied bytes: {count("copied_bytes")}',
                 f'Preserved requests: {count("request_count")}',
                 'External footage, assets and outputs remain at their original paths.',
                 'Imported history stays archived. Loading settings is separate from activation or execution.']
        if not usable and destination:
            lines.append('Partial destination retained. It is not a verified export or usable imported history.')
        for key, label in (('warning', 'Warning'), ('error', 'Error')):
            if report.get(key):
                lines.append(f'{label}: {str(report[key])[:4096]}')
        self.details.setPlainText('\n'.join(lines))
        is_import = self.task is not None and self.task.operation == 'import'
        unchanged = self.task is not None and self.task.source == self.source_path.text().strip() and self.task.parent == self.destination_parent.text().strip() and is_import == (self.operation.currentIndex() == 1)
        if usable and status == 'complete' and isinstance(destination, str) and destination and is_import and unchanged:
            self.imported_path = destination
        location = f' Destination: {destination}' if destination else ''
        title = 'Import' if is_import else 'Export'
        self.status.setText((f'{title} {status}. ' + ('Verified destination ready.' if usable else 'Destination is not ready to use.') + location + (' No activation, rendering or recovery submitted.' if usable else ' See actual copy counts and error below.'))[:4096])

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
