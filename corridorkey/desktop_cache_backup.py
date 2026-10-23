"""Explicit recovery-cache backup/restore with owned background I/O."""
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
class BackupResult:
    status: str
    raw: bytes | None
    error: str | None


class BackupTask:
    def __init__(self, operation: str, source: str, parent: str):
        if operation not in {'backup', 'restore'}:
            raise ValueError('Choose backup or restore.')
        self.operation = operation
        self.source = source
        self.parent = parent
        self.cancellation = CancellationToken()
        self.results: queue.Queue[BackupResult] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name='cache-backup-restore', daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Backup or restore cancelled by user')

    def _run(self):
        result = BackupResult('failed', None, 'Unexpected backup worker termination')
        try:
            from . import cache_backup

            check = self.cancellation.check
            check()
            # Preserve lexical paths for the core's symlink/admission policy;
            # canonicalization and all filesystem checks belong to this worker.
            source = Path(self.source)
            parent = Path(self.parent)
            check()
            function = cache_backup.backup_workspace if self.operation == 'backup' else cache_backup.restore_backup
            report = function(source, parent, check=check)
            # The core reports any committed or partial destination. Cancellation
            # after publication cannot replace that authoritative result.
            raw = json.dumps(report, allow_nan=False, separators=(',', ':')).encode('utf-8')
            if len(raw) > MAX_REPORT_BYTES:
                raise ValueError('Backup report exceeds the 4 MiB interface limit.')
            result = BackupResult('complete', raw, None)
        except CancelledError as exc:
            result = BackupResult('cancelled', None, str(exc)[:4096])
        except BaseException as exc:
            logging.getLogger(__name__).exception('Backup or restore failed')
            result = BackupResult('failed', None, f'{type(exc).__name__}: {exc}'[:4096])
        finally:
            self.results.put(result)
            self.done.set()


class CacheBackupPanel(QDialog):
    restored_workspace_ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Recovery cache backup / restore')
        self.resize(850, 540)
        self.task: BackupTask | None = None
        self.running = False
        self.close_when_done = False
        self.restored_path: str | None = None
        layout = QVBoxLayout(self)
        note = QLabel('Back up an existing recovery cache, or restore a verified backup into a fresh workspace. Choose an existing destination parent; each operation creates a new unique child and never overwrites another workspace. Source footage and exports are not copied. External paths remain provenance; copying does not guarantee cache reuse on another machine. Queue history portability is separate. Nothing renders or resumes automatically.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.operation = QComboBox()
        self.operation.addItems(['Create backup', 'Restore backup'])
        form.addRow('Operation', self.operation)
        self.source_path = QLineEdit()
        self.source_button = QPushButton('Source…')
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_path)
        source_row.addWidget(self.source_button)
        self.source_label = QLabel('Existing recovery workspace')
        form.addRow(self.source_label, source_row)
        self.destination_parent = QLineEdit()
        self.parent_button = QPushButton('Destination parent…')
        parent_row = QHBoxLayout()
        parent_row.addWidget(self.destination_parent)
        parent_row.addWidget(self.parent_button)
        form.addRow('Existing destination parent', parent_row)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.run_button = QPushButton('Create new backup')
        self.cancel_button = QPushButton('Cancel operation')
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        self.result_path = QLineEdit()
        self.result_path.setReadOnly(True)
        self.result_path.setPlaceholderText('Actual new or partial destination appears here')
        layout.addWidget(self.result_path)
        self.status = QLabel('Choose source and destination. No backup or restore started.')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.use_button = QPushButton('Use restored workspace for inventory')
        layout.addWidget(self.use_button)
        self.operation.currentIndexChanged.connect(self.mode_changed)
        self.source_button.clicked.connect(self.choose_source)
        self.parent_button.clicked.connect(self.choose_parent)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.use_button.clicked.connect(self.use_restored)
        self.source_path.textChanged.connect(self.invalidate_restored)
        self.destination_parent.textChanged.connect(self.invalidate_restored)
        self._controls()

    @property
    def io_busy(self) -> bool:
        return self.task is not None and self.task.busy

    def _controls(self):
        parent = self.parent()
        parent_running = parent is not None and getattr(parent, 'maintenance_busy', False)
        idle = not self.io_busy and not self.close_when_done and not parent_running
        for widget in (self.operation, self.source_path, self.source_button, self.destination_parent,
                       self.parent_button, self.run_button):
            widget.setEnabled(idle)
        self.cancel_button.setEnabled(self.running)
        self.use_button.setEnabled(idle and self.restored_path is not None)

    def invalidate_restored(self):
        self.restored_path = None
        self._controls()

    def mode_changed(self):
        self.invalidate_restored()
        restore = self.operation.currentIndex() == 1
        self.source_label.setText('Existing verified backup' if restore else 'Existing recovery workspace')
        self.run_button.setText('Restore to new workspace' if restore else 'Create new backup')
        self.source_path.clear()

    def set_workspace(self, path: str):
        if not self.io_busy and not self.close_when_done and self.operation.currentIndex() == 0:
            self.source_path.setText(path)

    def choose_source(self):
        if not self.io_busy and not self.close_when_done:
            path = QFileDialog.getExistingDirectory(self, 'Choose existing backup' if self.operation.currentIndex() == 1 else 'Choose existing recovery workspace')
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
        if parent is not None and (getattr(parent, 'maintenance_busy', False) or getattr(parent, 'close_when_done', False)):
            self.status.setText('Wait for cache maintenance and closing work before starting backup or restore.')
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
        operation = 'restore' if self.operation.currentIndex() == 1 else 'backup'
        self.restored_path = None
        self.result_path.clear()
        self.details.clear()
        try:
            self.task = BackupTask(operation, source, destination)
            self.running = True
            self.status.setText('Restoring into a new workspace…' if operation == 'restore' else 'Creating and verifying a new backup…')
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def cancel(self):
        if self.task is not None and self.io_busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual copy and verification cleanup. Any partial destination will be reported.')

    def use_restored(self):
        if self.restored_path is not None and not self.io_busy and not self.close_when_done:
            self.restored_workspace_ready.emit(self.restored_path)
            self.status.setText('Restored workspace selected for inventory. No recovery or rendering submitted.')

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
                 f'Sealed bundles: {count("entry_count")}',
                 f'Preserved journals: {count("journal_count")}',
                 f'Preserved reports: {count("report_count")}',
                 'External footage and exports remain at their original paths.',
                 'Recovery and rendering require a separate explicit request.']
        if not usable and destination:
            lines.append('Partial destination retained. It is not a completed backup or usable restored workspace.')
        for key, label in (('warning', 'Warning'), ('error', 'Error')):
            if report.get(key):
                lines.append(f'{label}: {str(report[key])[:4096]}')
        self.details.setPlainText('\n'.join(lines))
        restore = self.task is not None and self.task.operation == 'restore'
        unchanged = self.task is not None and self.task.source == self.source_path.text().strip() and self.task.parent == self.destination_parent.text().strip() and restore == (self.operation.currentIndex() == 1)
        if usable and status == 'complete' and isinstance(destination, str) and destination and restore and unchanged:
            self.restored_path = destination
        location = f' Destination: {destination}' if destination else ''
        title = 'Restore' if restore else 'Backup'
        self.status.setText((f'{title} {status}. ' + ('Verified destination ready.' if usable else 'Destination is not ready to use.') + location + (' No rendering or recovery submitted.' if usable else ' See actual copy counts and error below.'))[:4096])

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
