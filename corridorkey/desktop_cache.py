"""Explicit managed-cache maintenance with one owned background operation."""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .control import CancellationToken, CancelledError
from .desktop_cache_backup import CacheBackupPanel

MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_TARGET_BYTES = 2**63 - 1


def _encode(value: dict[str, Any]) -> bytes:
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError('Cache maintenance report exceeds the 16 MiB interface limit.')
    return raw


def _number(value: Any) -> str:
    return str(value) if type(value) is int and value >= 0 else 'Unknown'


def report_summary(operation: str, report: dict[str, Any]) -> str:
    """User-facing facts only; internal ownership and snapshot hashes stay hidden."""
    if operation == 'inventory':
        lines = [f'Sealed bundles: {_number(report.get("entry_count"))}',
                 f'Sealed logical bytes: {_number(report.get("sealed_bytes"))}',
                 f'Preserved staging: {_number(report.get("staging_entries"))} entries · {_number(report.get("staging_bytes"))} bytes',
                 f'Preserved journals: {_number(report.get("journal_count"))} files · {_number(report.get("journal_bytes"))} bytes',
                 f'Preserved reports: {_number(report.get("report_count"))} files · {_number(report.get("report_bytes"))} bytes',
                 'Inventory reads metadata; payload integrity is not verified. No files changed.']
    elif operation == 'preview':
        lines = [f'Retain at most: {_number(report.get("target_bytes"))} logical bytes',
                 f'Before trim: {_number(report.get("before_bytes"))} logical bytes',
                 f'Proposed removal: {_number(report.get("remove_count"))} bundles · {_number(report.get("remove_bytes"))} bytes',
                 f'After trim: {_number(report.get("after_bytes"))} logical bytes',
                 'Staging, journals, reports, source files and exports are preserved.',
                 'Preview changes no files. Review all candidate pages before Apply preview.']
    else:
        removed = report.get('removed')
        removed_count = len(removed) if isinstance(removed, list) else None
        lines = [f'Result: {report.get("status", "Unknown")}',
                 f'Committed removals: {_number(removed_count)} bundles · {_number(report.get("removed_bytes"))} bytes',
                 f'Retained bundles: {_number(report.get("remaining_entry_count"))}',
                 f'Retained sealed bytes: {_number(report.get("remaining_sealed_bytes"))}',
                 f'Retained file bytes, including partial bundles: {_number(report.get("remaining_entry_bytes"))}',
                 'Staging, journals, reports, source files and exports are preserved.']
        if report.get('partial_entry') is not None:
            lines.append(f'Partially removed bundle: {str(report["partial_entry"])[:4096]}')
            lines.append('Already removed files remain removed; inspect this bundle before reuse.')
        if report.get('error'):
            lines.append(f'Error: {str(report["error"])[:4096]}')
    return '\n'.join(lines)


@dataclass(frozen=True)
class MaintenanceResult:
    operation: str
    status: str
    raw: bytes | None
    error: str | None


class MaintenanceTask:
    """No Qt access or filesystem access on the initiating event thread."""

    def __init__(self, operation: str, root: str, target: int, plan: bytes | None = None):
        if operation not in {'inventory', 'preview', 'apply'}:
            raise ValueError('Unknown cache maintenance operation.')
        self.operation = operation
        self.root = root
        self.target = target
        self.plan = plan
        self.cancellation = CancellationToken()
        self.results: queue.Queue[MaintenanceResult] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name='cache-maintenance', daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Cache maintenance cancelled by user')

    def _run(self):
        result = MaintenanceResult(self.operation, 'failed', None, 'Unexpected maintenance worker termination')
        try:
            from . import cache_maintenance

            check = self.cancellation.check
            check()
            # Preserve lexical input so the core can reject a symlink root
            # before it resolves and inspects the workspace on this worker.
            root = Path(self.root)
            check()
            if self.operation == 'inventory':
                report = cache_maintenance.inventory(root, check=check)
            elif self.operation == 'preview':
                report = cache_maintenance.plan_trim(root, self.target, check=check)
            else:
                if self.plan is None:
                    raise ValueError('Preview a trim before applying it.')
                report = cache_maintenance.apply_trim(root, json.loads(self.plan), check=check)
            # Publication may already have occurred. A late cancellation must not
            # discard the authoritative report of actual removals.
            result = MaintenanceResult(self.operation, 'complete', _encode(report), None)
        except CancelledError as exc:
            result = MaintenanceResult(self.operation, 'cancelled', None, str(exc)[:4096])
        except BaseException as exc:
            logging.getLogger(__name__).exception('Cache maintenance failed')
            result = MaintenanceResult(self.operation, 'failed', None, f'{type(exc).__name__}: {exc}'[:4096])
        finally:
            self.results.put(result)
            self.done.set()


class CacheMaintenancePanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Managed cache maintenance')
        self.resize(900, 620)
        self.task: MaintenanceTask | None = None
        self.plan: bytes | None = None
        self.running = False
        self.close_when_done = False
        self.rows: tuple[tuple[str, str, str], ...] = ()
        self.page = 0
        self.request_fields: tuple[str, str] | None = None
        self.backup_panel = CacheBackupPanel(self)
        self.backup_panel.restored_workspace_ready.connect(self.set_workspace)
        layout = QVBoxLayout(self)
        note = QLabel('Inspect an existing recovery cache workspace without creating or pruning it. Preview lists least-recently-used managed bundles to remove for an exact byte target; Apply preview is the explicit deletion action. This is a one-time trim, not a future recovery budget or a disk reservation. Original sources, exports, journals and staging are preserved. Active workspaces and stale previews refuse. Logical file lengths exclude filesystem allocation, total disk, RAM and VRAM. Inventory does not certify payload integrity.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        root_row = QHBoxLayout()
        self.root_path = QLineEdit()
        self.root_path.setPlaceholderText('Existing managed recovery cache workspace')
        self.choose_button = QPushButton('Workspace…')
        self.choose_button.clicked.connect(self.choose_workspace)
        root_row.addWidget(self.root_path)
        root_row.addWidget(self.choose_button)
        form.addRow('Cache workspace', root_row)
        self.target = QLineEdit('1073741824')
        form.addRow('Retain at most bytes (0–9223372036854775807)', self.target)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.inventory_button = QPushButton('Inspect inventory')
        self.preview_button = QPushButton('Preview trim')
        self.apply_button = QPushButton('Apply preview')
        self.cancel_button = QPushButton('Cancel operation')
        self.inventory_button.clicked.connect(lambda: self.start('inventory'))
        self.preview_button.clicked.connect(lambda: self.start('preview'))
        self.apply_button.clicked.connect(lambda: self.start('apply'))
        self.cancel_button.clicked.connect(self.cancel)
        for button in (self.inventory_button, self.preview_button, self.apply_button, self.cancel_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.backup_button = QPushButton('Backup / restore…')
        self.backup_button.clicked.connect(self.open_backup)
        layout.addWidget(self.backup_button)
        self.status = QLabel('No maintenance started. Opening this panel changes no cache files.')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(160)
        layout.addWidget(self.details)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Managed bundle key', 'Logical bytes', 'Modified time (ns)'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 530)
        layout.addWidget(self.table)
        paging = QHBoxLayout()
        self.previous_button = QPushButton('Previous 100')
        self.next_button = QPushButton('Next 100')
        self.page_label = QLabel('No entries')
        self.previous_button.clicked.connect(lambda: self.show_page(self.page - 1))
        self.next_button.clicked.connect(lambda: self.show_page(self.page + 1))
        for widget in (self.previous_button, self.page_label, self.next_button):
            paging.addWidget(widget)
        layout.addLayout(paging)
        self.root_path.textChanged.connect(self.invalidate_plan)
        self.target.textChanged.connect(self.invalidate_plan)
        self._controls()

    @property
    def maintenance_busy(self) -> bool:
        return self.task is not None and self.task.busy

    @property
    def io_busy(self) -> bool:
        return self.maintenance_busy or self.backup_panel.io_busy

    def _controls(self):
        idle = not self.io_busy and not self.close_when_done
        for widget in (self.root_path, self.target, self.choose_button, self.inventory_button, self.preview_button):
            widget.setEnabled(idle)
        self.apply_button.setEnabled(idle and self.plan is not None)
        self.cancel_button.setEnabled(self.running)
        self.backup_button.setEnabled(idle)
        self.previous_button.setEnabled(self.page > 0)
        self.next_button.setEnabled((self.page + 1) * 100 < len(self.rows))

    def set_workspace(self, path: str):
        if not self.io_busy and not self.close_when_done:
            self.root_path.setText(path)

    def open_backup(self):
        if self.io_busy or self.close_when_done:
            return
        self.backup_panel.set_workspace(self.root_path.text())
        self.backup_panel.show()
        self.backup_panel.raise_()

    def show_page(self, page: int):
        self.page = max(0, min(page, max(0, (len(self.rows) - 1) // 100)))
        start = self.page * 100
        rows = self.rows[start:start + 100]
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.page_label.setText(f'Entries {start + 1}–{start + len(rows)} of {len(self.rows)}' if rows else 'No entries')
        self._controls()

    def invalidate_plan(self):
        self.plan = None
        self._controls()

    def choose_workspace(self):
        if self.io_busy or self.close_when_done:
            return
        path = QFileDialog.getExistingDirectory(self, 'Choose existing managed recovery cache workspace')
        if path:
            self.root_path.setText(path)

    def start(self, operation: str):
        if self.io_busy or self.close_when_done:
            return
        parent = self.parent()
        if parent is not None and getattr(parent, 'closing', False):
            self.status.setText('Workspace is closing; wait for owned I/O cleanup.')
            return
        try:
            if operation not in {'inventory', 'preview', 'apply'}:
                raise ValueError('Unknown cache maintenance operation.')
            root = self.root_path.text().strip()
            if not root:
                raise ValueError('Choose an existing managed cache workspace.')
            text = self.target.text().strip()
            if operation != 'inventory' and (not text.isascii() or not text.isdecimal() or len(text) > 19):
                raise ValueError('Retained bytes must be a decimal integer from 0 to 9223372036854775807.')
            target = int(text) if operation != 'inventory' else 0
            if target > MAX_TARGET_BYTES:
                raise ValueError('Retained byte target is too large.')
            if operation == 'apply' and self.plan is None:
                raise ValueError('Preview the exact trim before applying it.')
            plan = self.plan if operation == 'apply' else None
            self.plan = None
            self.details.clear()
            self.rows = ()
            self.show_page(0)
            self.request_fields = (self.root_path.text(), self.target.text())
            self.task = MaintenanceTask(operation, root, target, plan)
            self.running = True
            self.status.setText({'inventory': 'Reading bounded inventory…', 'preview': 'Checking exact trim candidates…', 'apply': 'Applying reviewed trim; waiting for actual results…'}[operation])
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def cancel(self):
        if self.backup_panel.io_busy:
            self.backup_panel.cancel()
        if self.task is not None and self.task.busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual file and database cleanup. Already removed bundles remain removed.')

    def poll(self):
        self.backup_panel.poll()
        if self.task is not None:
            try:
                result = self.task.results.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                self.running = False
                if result.raw is not None:
                    report = json.loads(result.raw)
                    entries = report.get('entries', report.get('remove', []))
                    self.rows = tuple((str(item['key']), str(item['bytes']), str(item['mtime_ns'])) for item in entries)
                    if 'removed' in report:
                        self.rows = tuple((str(key), 'Removed', '') for key in report['removed'])
                    self.details.setPlainText(report_summary(result.operation, report))
                    self.show_page(0)
                    unchanged = self.request_fields == (self.root_path.text(), self.target.text())
                    if result.operation == 'preview' and result.status == 'complete' and unchanged:
                        self.plan = result.raw
                    actual_status = report.get('status', result.status)
                    if result.operation == 'apply':
                        remaining = _number(report.get('remaining_sealed_bytes'))
                        retained = _number(report.get('remaining_entry_count'))
                        removed_bytes = _number(report.get('removed_bytes'))
                        message = f'Apply {actual_status}: {len(report.get("removed", []))} bundles removed, {removed_bytes} bytes removed; {remaining} logical bytes remain. Retained bundle count: {retained}. '
                        self.status.setText((message + str(report.get('error') or 'Original exports, journals and reports preserved.'))[:4096])
                    elif result.operation == 'preview':
                        self.status.setText('Review exact removal candidates below, then Apply preview to delete them.' if unchanged else 'Preview invalidated because the workspace or target changed; preview again.')
                    else:
                        self.status.setText(f'Inventory complete: {report.get("entry_count", 0)} sealed bundles, {report.get("sealed_bytes", 0)} logical bytes. No files changed.')
                else:
                    self.status.setText(f'{result.status}: {result.error}')
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def closeEvent(self, event: QCloseEvent):
        self.backup_panel.close()
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
