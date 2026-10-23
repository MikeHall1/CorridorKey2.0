"""Deliberate publisher plan review and owned model downloads; never autoactivate."""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
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

MAX_PLAN_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class DownloadResult:
    operation: str
    status: str
    raw: bytes | None
    error: str | None


class DownloadTask:
    def __init__(self, operation: str, *, plan: bytes | None = None, parent: str = '',
                 budget: int = 0, protected_paths: tuple[Path, ...] = ()):
        self.operation, self.plan = operation, plan
        self.parent, self.budget, self.protected_paths = parent, budget, protected_paths
        self.cancellation = CancellationToken()
        self.results: queue.Queue[DownloadResult] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest: ProgressEvent | None = None
        self.thread = threading.Thread(target=self._run, name='model-asset-download', daemon=True)
        self.thread.start()

    @property
    def busy(self):
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Model metadata resolution or download cancelled by user')

    def progress(self):
        with self.lock:
            return self.latest

    def _progress(self, event: ProgressEvent):
        with self.lock:
            self.latest = event

    def _run(self):
        result = DownloadResult(self.operation, 'failed', None, 'Unexpected model download worker termination')
        try:
            from . import asset_download

            check = self.cancellation.check
            check()
            if self.operation == 'catalog':
                data = {'status': 'complete', 'catalog': asset_download.catalog()}
            elif self.operation == 'resolve':
                data = asset_download.resolve_plan(check=check, progress=self._progress)
            elif self.operation == 'download' and self.plan is not None:
                data = asset_download.download_plan(json.loads(self.plan), Path(self.parent), self.budget,
                                                    protected_paths=self.protected_paths, check=check, progress=self._progress)
            else:
                raise ValueError('Resolve an online model plan before downloading.')
            raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8')
            if len(raw) > MAX_PLAN_BYTES:
                raise ValueError('Model download report exceeds the 16 MiB interface limit.')
            # A published complete tree remains complete after late cancellation.
            result = DownloadResult(self.operation, 'complete', raw, None)
        except CancelledError as exc:
            result = DownloadResult(self.operation, 'cancelled', None, str(exc)[:4096])
        except BaseException as exc:
            logging.getLogger(__name__).exception('Model asset download operation failed')
            result = DownloadResult(self.operation, 'failed', None, f'{type(exc).__name__}: {exc}'[:4096])
        finally:
            self.results.put(result)
            self.done.set()


class AssetDownloadPanel(QDialog):
    selection_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Download model assets')
        self.resize(1050, 760)
        self.task: DownloadTask | None = None
        self.plan: bytes | None = None
        self.catalog_loaded = False
        self.running = False
        self.close_when_done = False
        self.downloaded: AssetSelection | None = None
        layout = QVBoxLayout(self)
        note = QLabel('Opening this window reads an offline catalog only. Resolve online plan contacts the listed publishers for bounded metadata; it does not download model weights. Review licences, publishers, pinned revisions and exact payload bytes before Download. The asset tree includes local Python model code. Downloading does not prove model compatibility or GPU readiness. Select downloaded paths only fills Local assets; Check, Stop worker and Use remain separate explicit actions.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.resolve_button = QPushButton('Resolve online plan')
        self.download_button = QPushButton('Download reviewed plan')
        self.cancel_button = QPushButton('Cancel operation')
        for button in (self.resolve_button, self.download_button, self.cancel_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.catalog_details = QPlainTextEdit()
        self.catalog_details.setReadOnly(True)
        self.catalog_details.setMaximumHeight(170)
        layout.addWidget(self.catalog_details)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Local file', 'Publisher / pinned revision', 'Payload bytes', 'Expected checksum'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((300, 330, 120)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table)
        form = QFormLayout()
        row = QHBoxLayout()
        self.destination_parent = QLineEdit()
        self.destination_parent.setPlaceholderText('Existing parent separate from current model assets and source files')
        self.destination_button = QPushButton('Destination parent…')
        row.addWidget(self.destination_parent)
        row.addWidget(self.destination_button)
        form.addRow('Existing destination parent', row)
        self.budget = QLineEdit('0')
        form.addRow('Payload byte budget (0–9223372036854775807)', self.budget)
        layout.addLayout(form)
        self.review = QLabel('Payload size unknown until an online plan resolves. No weights downloaded.')
        self.review.setWordWrap(True)
        self.review.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.review)
        self.status = QLabel('No online metadata requested.')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.partial_files = QLabel()
        self.partial_files.setTextFormat(Qt.TextFormat.PlainText)
        self.partial_files.setWordWrap(True)
        layout.addWidget(self.partial_files)
        self.result_path = QLineEdit()
        self.result_path.setReadOnly(True)
        self.result_path.setPlaceholderText('Actual complete or partial asset destination appears here')
        layout.addWidget(self.result_path)
        self.select_button = QPushButton('Select downloaded paths in Local assets')
        layout.addWidget(self.select_button)
        self.resolve_button.clicked.connect(self.resolve_plan)
        self.download_button.clicked.connect(self.download)
        self.cancel_button.clicked.connect(self.cancel)
        self.destination_button.clicked.connect(self.choose_destination)
        self.select_button.clicked.connect(self.select_downloaded)
        self.destination_parent.textChanged.connect(self.invalidate_result)
        self.budget.textChanged.connect(self.invalidate_result)
        self._controls()

    @property
    def io_busy(self):
        return self.task is not None and self.task.busy

    def protected_paths(self) -> tuple[Path, ...]:
        parent = self.parent()
        if parent is None:
            return ()
        root, weights = getattr(parent, 'root', None), getattr(parent, 'weights', None)
        if not isinstance(root, QLineEdit) or not isinstance(weights, QLineEdit):
            raise TypeError('Download requires the current Local assets path selection.')
        return tuple(Path(value) for value in (root.text().strip(), weights.text().strip()) if value)

    def _budget(self):
        raw = self.budget.text().strip()
        if not raw.isascii() or not raw.isdecimal() or len(raw) > 19 or int(raw) > 2**63 - 1:
            raise ValueError('Choose an exact payload byte budget from 0 to 9223372036854775807.')
        return int(raw)

    def _can_start(self):
        parent = self.parent()
        owner = parent.parent() if parent is not None else None
        return not (self.io_busy or self.close_when_done or getattr(parent, 'presence_busy', False)
                    or getattr(parent, 'audit_busy', False) or getattr(parent, 'close_when_done', False)
                    or getattr(owner, 'closing', False))

    def _controls(self):
        idle = self._can_start()
        self.resolve_button.setEnabled(idle and self.catalog_loaded)
        self.destination_parent.setEnabled(idle)
        self.destination_button.setEnabled(idle)
        self.budget.setEnabled(idle)
        valid = False
        if self.plan is not None:
            total = json.loads(self.plan)['total_bytes']
            try:
                budget = self._budget()
                valid = bool(self.destination_parent.text().strip()) and budget >= total
                protected = '; '.join(str(path) for path in self.protected_paths()) or 'none selected'
                self.review.setText(f'Reviewed payload: {total} bytes. Budget: {budget} bytes. A new unique child will be created under: {self.destination_parent.text() or "choose a parent"}. Protected current asset paths: {protected}. Filesystem overhead requires additional space.')
            except ValueError as exc:
                self.review.setText(str(exc))
        self.download_button.setEnabled(idle and valid)
        self.cancel_button.setEnabled(self.running)
        self.select_button.setEnabled(idle and self.downloaded is not None)

    def invalidate_result(self):
        self.downloaded = None
        self._controls()

    def _start(self, operation: str, **kwargs):
        if not self._can_start():
            self.status.setText('Wait for the current asset operation or closing work.')
            return
        try:
            self.task = DownloadTask(operation, **kwargs)
            self.running = True
            self.status.setText({'catalog': 'Reading offline publisher catalog; no network request…',
                                 'resolve': 'Resolving publisher metadata only; no weight download…',
                                 'download': 'Downloading the reviewed immutable plan into a new asset tree…'}[operation])
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def load_catalog(self):
        if not self.catalog_loaded:
            self._start('catalog')

    def resolve_plan(self):
        if not self._can_start() or not self.catalog_loaded:
            return
        self.plan = None
        self.downloaded = None
        self.table.setRowCount(0)
        self.review.setText('Resolving exact payload size and revisions. No weights downloaded.')
        self._start('resolve')

    def choose_destination(self):
        if self._can_start():
            path = QFileDialog.getExistingDirectory(self, 'Choose an existing separate destination parent')
            if path:
                self.destination_parent.setText(path)

    def download(self):
        if not self._can_start() or self.plan is None:
            self.status.setText('Resolve and review an online plan before downloading.')
            return
        try:
            budget = self._budget()
            if not self.destination_parent.text().strip() or budget < json.loads(self.plan)['total_bytes']:
                raise ValueError('Choose a destination parent and a budget covering the reviewed payload bytes.')
            self.downloaded = None
            self.result_path.clear()
            self.partial_files.clear()
            self._start('download', plan=self.plan, parent=self.destination_parent.text().strip(), budget=budget,
                        protected_paths=self.protected_paths())
        except ValueError as exc:
            self.status.setText(str(exc)[:4096])

    def select_downloaded(self):
        if self._can_start() and self.downloaded is not None:
            self.selection_ready.emit(self.downloaded)
            self.status.setText('Downloaded paths filled in Local assets. Check assets/backend, stop the worker, then Use explicitly. No worker activated or job submitted.')

    def cancel(self):
        if self.task is not None and self.io_busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual network and file cleanup. Any partial destination will be shown.')

    def poll(self):
        if self.task is not None:
            event = self.task.progress()
            if self.running and event is not None and not self.task.cancellation.cancelled:
                phase = {'download_metadata': 'Resolving publisher metadata', 'asset_download_bytes': 'Downloading bytes',
                         'asset_download_files': 'Verifying files'}.get(event.stage, 'Preparing assets')
                self.status.setText(f'{phase} · {event.completed}/{event.total}' if event.total is not None else f'{phase} · {event.completed}')
            try:
                result = self.task.results.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                self.running = False
                if result.raw is None:
                    self.status.setText(f'{result.status}: {result.error}'[:4096])
                else:
                    self._show_result(result.operation, json.loads(result.raw))
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def _publishers(self, publishers):
        lines = []
        for item in publishers:
            lines.append(f'{item["asset"]}: {item["publisher"]} · {item["repository"]}\nLicence: {item["licence"]} · {item["licence_url"]}\nRequested revision: {item["requested_revision"]} · Pinned revision: {item.get("resolved_revision") or "Unknown until online resolution"}')
        self.catalog_details.setPlainText('\n\n'.join(lines))

    def _show_result(self, operation, data):
        if operation == 'catalog':
            self._publishers(data['catalog']['publishers'])
            self.catalog_loaded = True
            self.status.setText('Offline catalog loaded. Payload sizes unknown. Resolve online plan explicitly to request publisher metadata.')
        elif operation == 'resolve':
            if data.get('status') == 'complete' and isinstance(data.get('plan'), dict):
                plan = data['plan']
                if type(plan.get('total_bytes')) is not int or plan['total_bytes'] <= 0:
                    self.status.setText('Plan payload size is unknown or invalid; downloading is disabled.')
                    return
                self.plan = json.dumps(plan, allow_nan=False, separators=(',', ':')).encode('utf-8')
                self._publishers(plan['publishers'])
                self.table.setRowCount(len(plan['files']))
                for row, item in enumerate(plan['files']):
                    checksum = item['checksum']
                    values = [item['path'], f'{item["repository"]} @ {item["revision"]}', item['bytes'],
                              f'{checksum["algorithm"]}: {checksum["value"]}']
                    for column, value in enumerate(values):
                        cell = QTableWidgetItem(str(value))
                        cell.setToolTip(str(value))
                        self.table.setItem(row, column, cell)
                self.status.setText('Online plan resolved; no model payload downloaded. Review publisher licences, pinned revisions, hashes, destination and budget, then Download explicitly.')
            else:
                self.status.setText(f'Plan {data.get("status", "failed")}: {data.get("error")}'[:4096])
        else:
            usable = data.get('usable') is True
            destination = data.get('destination')
            self.result_path.setText(str(destination) if destination else '')
            partial = data.get('partial_files') or []
            descriptions = [f'{item.get("path", item.get("name", "Unknown path"))}: {item.get("bytes", "Unknown")} bytes'
                            if isinstance(item, dict) else str(item) for item in partial]
            self.partial_files.setText(f'{"Retained private files" if usable else "Partial files retained"}: {len(partial)}. {descriptions[0]}' if partial else '')
            self.partial_files.setToolTip('\n'.join(descriptions))
            assert self.task is not None
            try:
                unchanged = (self.task.plan == self.plan and self.task.parent == self.destination_parent.text().strip()
                             and self.task.budget == self._budget() and self.task.protected_paths == self.protected_paths())
            except ValueError:
                unchanged = False
            relative = Path(data.get('weights_relative', ''))
            safe = bool(str(relative)) and not relative.is_absolute() and '..' not in relative.parts and relative != Path('.')
            if usable and data.get('status') == 'complete' and destination and unchanged and safe:
                self.downloaded = AssetSelection(Path(destination), Path(destination) / relative)
            count = data.get('copied_files')
            size = data.get('copied_bytes')
            self.status.setText((f'Download {data.get("status", "failed")}: {count if count is not None else "Unknown"} files, {size if size is not None else "Unknown"} bytes copied. ' + ('Complete asset tree ready for explicit path selection; model/GPU compatibility NOT VERIFIED. ' if usable else 'Destination is partial or unavailable and cannot be selected. ') + str(data.get('error') or data.get('warning') or ''))[:4096])

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
