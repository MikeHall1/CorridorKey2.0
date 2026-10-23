"""Durable queue workspace and read-only observation history; no automatic replay."""
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
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .desktop_queue_transfer import QueueTransferPanel
from .queue_store import TERMINAL, QueueStore
from .worker_protocol import job_from_wire


class QueuePanel(QDialog):
    activate_requested = Signal(object)
    load_requested = Signal(object)
    recovery_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Durable queue history')
        self.resize(1020, 720)
        self.root: Path | None = None
        self.archived = False
        self.last_archive: Path | None = None
        self.rows: list[dict[str, Any]] = []
        self.cursors: list[int | None] = [None]
        self.next_before: int | None = None
        self.io_thread: threading.Thread | None = None
        self.results: queue.Queue[tuple[str, Path, Any, str | None]] = queue.Queue(maxsize=1)
        self.close_when_done = False
        self.transfer_panel = QueueTransferPanel(self)
        self.transfer_panel.history_ready.connect(self.open_imported_history)
        layout = QVBoxLayout(self)
        note = QLabel('Create a private queue workspace for future jobs, or inspect an existing one. Saved results are observations: outputs and old PIDs are NOT VERIFIED. Nonterminal history is unresolved. Opening/loading settings never submits work. Use explicit recoverable resume to validate cached/native outputs.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.active = QLabel('Active queue: session only')
        self.location = QLabel('No history selected')
        self.location.setWordWrap(True)
        layout.addWidget(self.active)
        layout.addWidget(self.location)
        buttons = QHBoxLayout()
        for label, callback in [('Create queue…', self.choose_create), ('Open history…', self.choose_open),
                                ('Use for new jobs', self.activate), ('Refresh', self.refresh),
                                ('Inspect selected', self.inspect_selected), ('Load render settings', self.load_selected),
                                ('Load recovery settings', self.load_recovery)]:
            button = QPushButton(label)
            if label == 'Use for new jobs':
                self.use_button = button
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        archive_bar = QHBoxLayout()
        self.archive_button = QPushButton('Archive stopped history')
        self.archive_button.clicked.connect(self.seal_history)
        archive_bar.addWidget(self.archive_button)
        self.fresh_button = QPushButton('Archive + fresh queue')
        self.fresh_button.clicked.connect(self.rotate_history)
        archive_bar.addWidget(self.fresh_button)
        self.last_archive_button = QPushButton('Open last archive')
        self.last_archive_button.clicked.connect(self.open_last_archive)
        archive_bar.addWidget(self.last_archive_button)
        self.archive_state = QLabel('Stop worker and wait before archiving. No records or outputs are deleted.')
        self.archive_state.setWordWrap(True)
        archive_bar.addWidget(self.archive_state, 1)
        layout.addLayout(archive_bar)
        self.transfer_button = QPushButton('Export / import history…')
        self.transfer_button.clicked.connect(self.open_transfer)
        layout.addWidget(self.transfer_button)
        self.last_archive_label = QLabel('No archive recorded in this app session')
        self.last_archive_label.setWordWrap(True)
        layout.addWidget(self.last_archive_label)
        navigation = QHBoxLayout()
        self.newer_button, self.older_button = QPushButton('Newer page'), QPushButton('Older page')
        self.newer_button.clicked.connect(self.newer)
        self.older_button.clicked.connect(self.older)
        self.page_label = QLabel('Page 1 · up to128 saved observations per page')
        navigation.addWidget(self.newer_button)
        navigation.addWidget(self.older_button)
        navigation.addWidget(self.page_label, 1)
        layout.addLayout(navigation)
        self.update_navigation()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Request', 'Action', 'Saved observation', 'Historical PID', 'Generation'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        for index, width in enumerate((130, 95, 280, 110)):
            self.table.setColumnWidth(index, width)
        layout.addWidget(self.table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 2)
        self.status = QLabel('No work is automatically retried.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @property
    def history_busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    @property
    def io_busy(self):
        return self.history_busy or self.transfer_panel.io_busy

    def open_transfer(self):
        if self.io_busy or self.close_when_done:
            return
        if self.root is not None:
            self.transfer_panel.set_history(str(self.root))
        self.transfer_panel.show()
        self.transfer_panel.raise_()

    def open_imported_history(self, root: str):
        self.run_io('open', Path(root))

    def run_io(self, action: str, root: Path, request_id: str | None = None, *, before: int | None = None):
        if self.io_busy or self.close_when_done or getattr(self.parent(), 'closing', False):
            self.status.setText('Wait for current history I/O.')
            return
        self.io_thread = threading.Thread(target=self._io, args=(action, root, request_id, before), daemon=True)
        self.io_thread.start()
        self.newer_button.setEnabled(False)
        self.older_button.setEnabled(False)
        self.status.setText('Reading/writing queue metadata…')

    def _io(self, action, root, request_id, before):
        original = root
        archived_root: Path | None = None
        try:
            if action == 'create':
                root = QueueStore.create(root)
            elif action == 'archive':
                QueueStore.seal(root)
                archived_root = root
            elif action == 'rotate':
                if QueueStore.inspect(root)['archive'] is None:
                    QueueStore.seal(root)
                archived_root = root
                root = QueueStore.create(root.parent)
            data = QueueStore.inspect(root, request_id=request_id, before=before)
            if archived_root is not None:
                data['archived_root'] = str(archived_root)
            error = None
        except BaseException as exc:
            logging.getLogger(__name__).exception('Queue workspace I/O failed')
            data, error = None, f'{type(exc).__name__}: {exc}'[:4096]
            if action in {'archive', 'rotate'}:
                # An exception after publication does not undo a seal. Re-read
                # actual history before reporting the partial transition.
                try:
                    observed = QueueStore.inspect(original)
                    if observed['archive'] is not None:
                        data = {'archived_root': str(original), 'fresh_root': str(root) if root != original else None}
                except BaseException:
                    logging.getLogger(__name__).exception('Archive state after failure is unverified')
                root = original
        self.results.put((action, root, data, error))

    def choose_create(self):
        parent = QFileDialog.getExistingDirectory(self, 'Create private queue under this parent')
        if parent:
            self.run_io('create', Path(parent))

    def choose_open(self):
        path = QFileDialog.getExistingDirectory(self, 'Open owned queue workspace (history only)')
        if path:
            self.run_io('open', Path(path))

    def activate(self):
        if self.archived:
            self.status.setText('Archived history cannot accept new jobs. Create a new queue workspace.')
            return
        if self.root is not None and not self.io_busy:
            self.activate_requested.emit(self.root)

    def refresh(self):
        if self.root is not None:
            self.run_io('open', self.root)

    def update_navigation(self):
        idle = not self.io_busy and not self.close_when_done
        self.use_button.setEnabled(idle and self.root is not None and not self.archived)
        self.archive_button.setEnabled(idle and self.root is not None and not self.archived)
        self.fresh_button.setEnabled(idle and self.root is not None)
        self.fresh_button.setText('Fresh queue beside archive' if self.archived else 'Archive + fresh queue')
        self.last_archive_button.setEnabled(idle and self.last_archive is not None)
        self.transfer_button.setEnabled(idle)
        self.archive_state.setText('Archived history · managed writes disabled; no records or outputs deleted.' if self.archived
                                   else 'Stop worker and wait before archiving. No records or outputs are deleted.')
        self.newer_button.setEnabled(idle and len(self.cursors) > 1)
        self.older_button.setEnabled(idle and self.next_before is not None)
        self.page_label.setText(f'Page {len(self.cursors)} · up to128 saved observations per page; Refresh returns to latest')

    def seal_history(self):
        if self.root is not None and not self.archived:
            self.run_io('archive', self.root)

    def rotate_history(self):
        if self.root is not None:
            self.run_io('rotate', self.root)

    def open_last_archive(self):
        if self.last_archive is not None:
            self.run_io('open', self.last_archive)

    def older(self):
        if self.root is not None and self.next_before is not None and not self.io_busy:
            self.run_io('older', self.root, before=self.next_before)

    def newer(self):
        if self.root is not None and len(self.cursors) > 1 and not self.io_busy:
            self.run_io('newer', self.root, before=self.cursors[-2])

    def selected(self):
        selection = self.table.selectionModel()
        selected = selection.selectedRows() if selection is not None else []
        row = selected[0].row() if len(selected) == 1 else -1
        return self.rows[row]['id'] if 0 <= row < len(self.rows) else None

    def inspect_selected(self):
        request_id = self.selected()
        if self.root is not None and request_id is not None:
            self.run_io('inspect', self.root, request_id)

    def load_selected(self):
        request_id = self.selected()
        if self.root is not None and request_id is not None:
            self.run_io('load', self.root, request_id)

    def load_recovery(self):
        request_id = self.selected()
        if self.root is not None and request_id is not None:
            self.run_io('load_recovery', self.root, request_id)

    def poll(self):
        self.transfer_panel.poll()
        self._poll_history()
        self.update_navigation()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def _poll_history(self):
        try:
            action, root, data, error = self.results.get_nowait()
        except queue.Empty:
            return
        if error is not None:
            if data is not None and data.get('archived_root'):
                self.archived = True
                self.last_archive = Path(data['archived_root'])
                self.last_archive_label.setText('Preserved archive: ' + str(self.last_archive))
                error += ' · History is archived. No new queue activated. Candidate fresh workspace: ' + str(data.get('fresh_root'))
            self.status.setText(error + (' · Refresh to inspect archive state. Stop the owner and wait before archiving active history.' if action in {'archive', 'rotate'} else ''))
            self.update_navigation()
            return
        self.root = root
        self.archived = data['archive'] is not None
        if data.get('archived_root'):
            self.last_archive = Path(data['archived_root'])
            self.last_archive_label.setText('Preserved archive: ' + str(self.last_archive))
        self.location.setText(str(root))
        if action in {'create', 'open', 'older', 'newer', 'archive', 'rotate'}:
            if action == 'older':
                self.cursors.append(data['before'])
            elif action == 'newer':
                self.cursors.pop()
            else:
                self.cursors = [None]
            self.next_before = data['next_before']
            self.update_navigation()
            self.details.clear()
            self.rows = data['rows']
            self.table.setRowCount(len(self.rows))
            self.table.clearSelection()
            for index, row in enumerate(self.rows):
                observed = row['status'] if row['status'] in TERMINAL else 'unresolved: ' + row['status']
                if row['status'] == 'result':
                    observed = 'observed result; outputs unverified'
                for col, value in enumerate([row['id'][:12], row['action'], observed, row['worker_pid'], row['generation']]):
                    self.table.setItem(index, col, QTableWidgetItem(str(value)))
            self.status.setText(data['scope'])
            if action in {'create', 'rotate'}:
                self.activate_requested.emit(root)
        else:
            self.update_navigation()
            text = json.dumps(data, indent=2)
            self.details.setPlainText(text[:2 * 1024 * 1024] + ('\n[View truncated; database record preserved.]' if len(text) > 2 * 1024 * 1024 else ''))
            self.status.setText(data['scope'])
            if action == 'load_recovery':
                if data['action'] == 'recover':
                    self.recovery_requested.emit(data.get('load_payload', data['request']['payload']))
                else:
                    self.status.setText('Select a recovery request first.')
            elif action == 'load':
                if data['action'] != 'render':
                    self.status.setText('Recovery entries are inspected/resumed through recovery controls, not the single-job form.')
                else:
                    self.load_requested.emit(job_from_wire(data['request']['payload']['job']))

    def closeEvent(self, event: QCloseEvent):
        self.transfer_panel.close()
        if self.io_busy:
            self.close_when_done = True
            event.ignore()
        else:
            event.accept()
