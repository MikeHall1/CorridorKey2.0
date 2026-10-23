"""Explicit worker and filesystem snapshots; no recurring polling or model load."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from .resource_snapshot import disk_snapshot


class ResourcesPanel(QDialog):
    worker_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Resource snapshots')
        self.resize(960, 690)
        self.io_thread: threading.Thread | None = None
        self.results: queue.Queue[tuple[dict[str, Any] | None, str | None]] = queue.Queue(maxsize=1)
        layout = QVBoxLayout(self)
        note = QLabel('Snapshots are explicit and may change immediately. Worker status starts its lazy connection if needed, without a model/render. Peak process RSS is not current RAM; CUDA values cover the initialized PyTorch allocator only, not total GPU usage. Filesystem free space is not a job-size estimate, cache budget, reservation or guarantee. No background refresh or memory limits are applied.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        for label, callback in [('Sample worker', self.worker_requested.emit), ('Filesystem directory…', self.choose_path),
                                ('Sample filesystem', self.sample_disk)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.path = QLineEdit(str(Path.cwd()))
        self.path.setPlaceholderText('Output/cache directory or a not-yet-created child path')
        layout.addWidget(self.path)
        self.status = QLabel('No snapshots sampled.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        self.worker_view, self.disk_view = QPlainTextEdit(), QPlainTextEdit()
        for view in (self.worker_view, self.disk_view):
            view.setReadOnly(True)
        self.tabs.addTab(self.worker_view, 'Worker process / CUDA allocator')
        self.tabs.addTab(self.disk_view, 'Containing filesystem')
        layout.addWidget(self.tabs)

    @property
    def io_busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    def choose_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Select filesystem directory to sample')
        if path:
            self.path.setText(path)

    def sample_disk(self):
        if self.io_busy:
            self.status.setText('Wait for current filesystem I/O.')
            return
        if not self.path.text().strip():
            self.status.setText('Enter a directory or future output path first.')
            return
        self.io_thread = threading.Thread(target=self._disk, args=(Path(self.path.text()),), daemon=True)
        self.io_thread.start()
        self.status.setText('Sampling filesystem space off the UI thread…')

    def _disk(self, path: Path):
        try:
            self.results.put((disk_snapshot(path), None))
        except Exception as exc:
            logging.getLogger(__name__).exception('Filesystem snapshot failed')
            self.results.put((None, f'{type(exc).__name__}: {exc}'))

    @staticmethod
    def text(data: dict[str, Any]) -> str:
        raw = json.dumps(data, indent=2, allow_nan=False)
        return raw[:65536] + ('\n[View truncated; snapshot scope unchanged.]' if len(raw) > 65536 else '')

    def observe(self, event: dict[str, Any]):
        self.worker_view.setPlainText(self.text(event))
        self.tabs.setCurrentIndex(0)
        if event['kind'] == 'status' and 'resources' in event['payload']:
            resources = event['payload']['resources']
            peak = resources['process']['peak_rss_bytes']
            rss = f'{peak / 2**20:.1f} MiB' if peak is not None else 'NOT VERIFIED'
            gpu = resources['gpu']
            cuda = (f"allocated {gpu['allocated_bytes'] / 2**20:.1f} MiB, reserved {gpu['reserved_bytes'] / 2**20:.1f} MiB"
                    if gpu['state'] == 'sampled' else 'unavailable: ' + gpu['state'])
            self.status.setText(f'Process lifetime peak RSS {rss} (not current RAM). CUDA allocator {cuda}. Sampled {resources["sampled_utc"]}; no render submitted.')
        else:
            self.status.setText(('Worker snapshot unavailable: ' + str(event.get('payload', {})))[:4096])

    def poll(self):
        try:
            data, error = self.results.get_nowait()
        except queue.Empty:
            return
        if error is not None:
            self.status.setText(error)
        elif data is not None:
            self.disk_view.setPlainText(self.text(data))
            self.tabs.setCurrentIndex(1)
            self.status.setText(f'Filesystem free {data["free_bytes"] / 2**30:.2f} GiB at {data["sampled_utc"]}. No directory created or space reserved; job fit is not guaranteed.')
