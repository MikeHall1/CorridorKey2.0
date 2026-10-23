"""Optional structural asset audit and explicit report export, without loading models."""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

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

from .asset_inventory import AssetSelection
from .control import CancellationToken, CancelledError, ProgressEvent

MAX_REPORT_BYTES = 16 * 1024 * 1024


def entry_summary(entry: dict) -> str:
    """Show the decision and limits before optional container statistics."""
    details = entry.get('details')
    if not isinstance(details, dict):
        return str(details or entry.get('format') or 'No structural details available')
    parts = []
    reason = details.get('error') or details.get('reason')
    missing = details.get('missing_vectors')
    if reason:
        parts.append(str(reason))
    elif isinstance(missing, list) and missing:
        parts.append(f'Missing {len(missing)} normalization vectors; values NOT VERIFIED')
    elif details.get('runtime_format'):
        parts.append('Runtime checkpoint header recognized; model compatibility NOT VERIFIED')
    elif entry.get('format_status') == 'UNVERIFIED':
        parts.append(str(details.get('weights') or 'Container inspected; model-dependent values NOT VERIFIED'))
    else:
        parts.append(f'{entry.get("format", "Container")} structure checked')
    for key, noun in (('tensor_count', 'tensors'), ('file_count', 'files'), ('shard_count', 'shards')):
        value = details.get(key)
        if type(value) is int:
            parts.append(f'{value} {noun}')
    return ' · '.join(parts)


@dataclass(frozen=True)
class AuditResult:
    operation: str
    status: str
    raw: bytes | None
    error: str | None


class AuditTask:
    def __init__(self, operation: str, selection: AssetSelection, report: bytes | None = None, destination: str | None = None):
        self.operation, self.selection = operation, selection
        self.report, self.destination = report, destination
        self.cancellation = CancellationToken()
        self.results: queue.Queue[AuditResult] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest: ProgressEvent | None = None
        self.thread = threading.Thread(target=self._run, name='local-asset-audit', daemon=True)
        self.thread.start()

    @property
    def busy(self):
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Asset audit or report export cancelled by user')

    def progress(self):
        with self.lock:
            return self.latest

    def _progress(self, event: ProgressEvent):
        with self.lock:
            self.latest = event

    def _run(self):
        result = AuditResult(self.operation, 'failed', None, 'Unexpected asset audit worker termination')
        try:
            from . import asset_audit

            self.cancellation.check()
            if self.operation == 'audit':
                data = asset_audit.audit_assets(self.selection, check=self.cancellation.check, progress=self._progress)
            elif self.operation == 'export' and self.report is not None and self.destination is not None:
                data = asset_audit.export_audit(json.loads(self.report), Path(self.destination), check=self.cancellation.check)
            else:
                raise ValueError('Choose a completed audit before exporting its report.')
            raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8')
            if len(raw) > MAX_REPORT_BYTES:
                raise ValueError('Asset audit report exceeds the 16 MiB interface limit.')
            # Export may have committed already. Preserve its authoritative result.
            result = AuditResult(self.operation, 'complete', raw, None)
        except CancelledError as exc:
            result = AuditResult(self.operation, 'cancelled', None, str(exc)[:4096])
        except BaseException as exc:
            logging.getLogger(__name__).exception('Asset audit/export failed')
            result = AuditResult(self.operation, 'failed', None, f'{type(exc).__name__}: {exc}'[:4096])
        finally:
            self.results.put(result)
            self.done.set()


class AssetAuditPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Audit local model files')
        self.resize(1000, 650)
        self.task: AuditTask | None = None
        self.report: bytes | None = None
        self.running = False
        self.close_when_done = False
        layout = QVBoxLayout(self)
        note = QLabel('This optional audit hashes local files and checks supported container structure without deserializing models. Structural PASS does not prove tensor compatibility, publisher authenticity, GPU readiness or successful inference. Legacy .pt remains supported by the runtime; its structure is UNVERIFIED here. Audit results do not change the selected assets. Save report rechecks source identity and writes a fresh file without overwriting existing files.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.source_root, self.weights_path = QLineEdit(), QLineEdit()
        self.source_root.setReadOnly(True)
        self.weights_path.setReadOnly(True)
        self.weights_path.setPlaceholderText('Default checkpoint selection in this tree')
        form.addRow('Selected local asset tree', self.source_root)
        form.addRow('Selected checkpoint override', self.weights_path)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.audit_button = QPushButton('Audit selected files')
        self.save_button = QPushButton('Save report…')
        self.cancel_button = QPushButton('Cancel operation')
        for button in (self.audit_button, self.save_button, self.cancel_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.audited_selection = QLabel('No audited selection.')
        self.audited_selection.setWordWrap(True)
        self.audited_selection.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.audited_selection)
        self.status = QLabel('Audit is optional. No model loaded or report saved.')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Asset', 'Presence', 'Structure', 'Local SHA-256', 'Format / details'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((140, 85, 100, 160)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(140)
        layout.addWidget(self.details)
        self.export_path = QLineEdit()
        self.export_path.setReadOnly(True)
        self.export_path.setPlaceholderText('Actual saved or partial report path appears here')
        layout.addWidget(self.export_path)
        self.audit_button.clicked.connect(self.start)
        self.save_button.clicked.connect(self.choose_save)
        self.cancel_button.clicked.connect(self.cancel)
        self.source_root.textChanged.connect(self.invalidate_report)
        self.weights_path.textChanged.connect(self.invalidate_report)
        self._controls()

    @property
    def io_busy(self):
        return self.task is not None and self.task.busy

    def selection(self):
        root, weights = self.source_root.text().strip(), self.weights_path.text().strip()
        if not root:
            raise ValueError('Choose a local asset tree in Local assets first.')
        return AssetSelection(Path(root), Path(weights) if weights else None)

    def _controls(self):
        parent = self.parent()
        idle = not self.io_busy and not self.close_when_done and not getattr(parent, 'presence_busy', False) and not getattr(parent, 'download_busy', False)
        self.audit_button.setEnabled(idle and bool(self.source_root.text().strip()))
        self.save_button.setEnabled(idle and self.report is not None)
        self.cancel_button.setEnabled(self.running)

    def invalidate_report(self):
        self.report = None
        self._controls()

    def load_selection(self, selection: AssetSelection):
        self.source_root.setText(str(selection.root))
        self.weights_path.setText(str(selection.weights) if selection.weights is not None else '')
        self.invalidate_report()

    def _can_start(self):
        parent = self.parent()
        owner = parent.parent() if parent is not None else None
        return not (self.io_busy or self.close_when_done or getattr(parent, 'presence_busy', False)
                    or getattr(parent, 'download_busy', False)
                    or getattr(parent, 'close_when_done', False) or getattr(owner, 'closing', False))

    def start(self):
        if not self._can_start():
            self.status.setText('Wait for current asset check, audit or closing work.')
            return
        try:
            selection = self.selection()
            self.report = None
            self.export_path.clear()
            self.table.setRowCount(0)
            self.task = AuditTask('audit', selection)
            self.running = True
            self.audited_selection.setText(f'Auditing: {selection.root}\nCheckpoint: {selection.weights or "default selection"}')
            self.status.setText('Checking structure and hashing selected local files…')
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def choose_save(self):
        if not self._can_start() or self.report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save audit to a new report file', '', 'JSON (*.json)',
                                            options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.save_report(Path(path))

    def save_report(self, path: Path):
        if not self._can_start() or self.report is None:
            self.status.setText('Audit the current selection before saving a report.')
            return
        try:
            self.task = AuditTask('export', self.selection(), self.report, str(path))
            self.running = True
            self.export_path.clear()
            self.status.setText('Rechecking audited sources before saving a fresh report…')
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def cancel(self):
        if self.task is not None and self.io_busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual file I/O cleanup…')

    def poll(self):
        if self.task is not None:
            event = self.task.progress()
            if self.running and event is not None and not self.task.cancellation.cancelled:
                unit = 'bytes' if event.stage == 'asset_hash' else 'assets'
                count = f'{event.completed}/{event.total}' if event.total is not None else str(event.completed)
                self.status.setText(f'{"Hashing local files" if unit == "bytes" else "Checking file structure"} · {count} {unit}')
            try:
                result = self.task.results.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                self.running = False
                if result.raw is None:
                    self.status.setText(f'{result.status}: {result.error}'[:4096])
                elif result.operation == 'audit':
                    self._show_audit(result.raw)
                else:
                    self._show_export(json.loads(result.raw))
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def _show_audit(self, raw: bytes):
        data = json.loads(raw)
        entries = data.get('entries', [])
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = [entry['asset'], 'Present' if entry.get('present') else 'Missing', entry.get('format_status', 'UNVERIFIED'),
                      entry.get('sha256') or 'Unavailable', entry_summary(entry)]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                tooltip = json.dumps(entry.get('details'), indent=2, ensure_ascii=False) if column == 4 else str(value)
                item.setToolTip(str(entry.get('path', '')) if column == 0 else tooltip)
                self.table.setItem(row, column, item)
        assert self.task is not None
        try:
            current = self.task.selection == self.selection()
        except ValueError:
            current = False
        captured = self.task.selection
        self.audited_selection.setText(f'Audited selection: {captured.root}\nCheckpoint: {captured.weights or "default selection"}')
        complete = data.get('status') == 'complete'
        self.report = raw if complete and current else None
        self.details.setPlainText('Model compatibility: NOT VERIFIED\nPublisher authenticity: NOT VERIFIED\nLocal hashes identify these files; they do not authenticate their publisher.\n' + str(data.get('error') or 'Structural checks do not establish successful inference, CUDA readiness or visual quality.'))
        self.status.setText(('Audit scan complete; inspect individual PASS / FAIL / UNVERIFIED results. No assets selected or models loaded.' if complete else f'Audit {data.get("status", "failed")}: {data.get("error")}')[:4096] if current else 'Selection changed during audit. Results describe the captured paths above; audit the current selection before saving.')

    def _show_export(self, data):
        usable = data.get('usable') is True
        path = data.get('destination') if usable else (data.get('partial_path') or data.get('destination'))
        self.export_path.setText(str(path) if path is not None else '')
        status = data.get('status', 'failed')
        self.status.setText((f'Report export {status}. ' + ('Fresh report saved.' if usable else 'No completed report saved.') + f' {data.get("error") or data.get("warning") or ""}')[:4096])
        if not usable:
            self.report = None
        if data.get('partial_path'):
            label = 'Private report alias retained; published report is complete: ' if usable else 'Partial report retained (not complete): '
            self.details.appendPlainText(label + str(data['partial_path']))

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
