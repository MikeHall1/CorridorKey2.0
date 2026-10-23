"""Explicit readiness checks and immutable short-render preparation for Qt."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .control import CancellationToken
from .readiness import check_readiness, export_report, prepare_short_render
from .session import JobRequest
from .worker_protocol import job_to_wire

if TYPE_CHECKING:
    from .desktop_window import DesktopWindow


class ReadinessPanel(QDialog):
    run_requested = Signal(object)

    def __init__(self, parent: DesktopWindow):
        super().__init__(parent)
        self.workspace = parent
        self.setWindowTitle('CorridorKey · Readiness and guided render')
        self.resize(880, 680)
        self.io_thread: threading.Thread | None = None
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self.cancellation = CancellationToken()
        self.close_when_done = False
        self.report: dict | None = None
        self.prepared: JobRequest | None = None
        self.submitted_id: str | None = None
        self.prepare_revision = 0
        self.prepared_selection = None
        self.render_observation: dict | None = None
        self.render_protected: list[str] = []
        layout = QVBoxLayout(self)
        note = QLabel('Lightweight checks run only when requested. They do not load models, run inference, install or download assets. Prepare snapshots the current job; Run submits that snapshot through the existing render worker. Model loading and actual inference are only tested by that explicit render. Decode/display acceptance requires inspecting its completed outputs in the workspace.')
        note.setWordWrap(True)
        layout.addWidget(note)
        actions = QHBoxLayout()
        self.check_button = QPushButton('Check readiness')
        self.check_button.clicked.connect(self.inspect)
        actions.addWidget(self.check_button)
        self.cancel_button = QPushButton('Cancel checks / render')
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.cancel_button)
        self.export_button = QPushButton('Export local report…')
        self.export_button.clicked.connect(self.choose_export)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        form = QFormLayout()
        self.start = QSpinBox()
        self.start.setRange(0, 1000000000)
        self.count = QSpinBox()
        self.count.setRange(1, 120)
        self.count.setValue(8)
        form.addRow('Absolute source start frame', self.start)
        form.addRow('Frame count (maximum 120)', self.count)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.prepare_button = QPushButton('Prepare short render')
        self.prepare_button.clicked.connect(self.prepare)
        actions.addWidget(self.prepare_button)
        self.run_button = QPushButton('Run displayed snapshot')
        self.run_button.clicked.connect(self.run)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)
        self.status = QLabel('No checks or render performed. Choose local footage and an output parent in the workspace.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.start.valueChanged.connect(self.invalidate_prepared)
        self.count.valueChanged.connect(self.invalidate_prepared)
        self._buttons()

    @property
    def io_busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    def _buttons(self):
        idle = not self.io_busy and not self.close_when_done and not self.workspace.closing
        self.check_button.setEnabled(idle)
        self.prepare_button.setEnabled(idle and self.submitted_id is None)
        self.run_button.setEnabled(idle and self.prepared is not None and self.submitted_id is None)
        self.export_button.setEnabled(idle and self.report is not None)

    def invalidate_prepared(self):
        self.prepare_revision += 1
        self.prepared = None
        self._buttons()

    def _begin(self, kind, function):
        if self.io_busy or self.close_when_done or self.workspace.closing:
            self.status.setText('Wait for the current operation or shutdown.')
            return
        self.cancellation = CancellationToken()

        def work():
            try:
                result = function()
                error = None
            except Exception as exc:
                logging.getLogger(__name__).exception('Readiness operation failed')
                result, error = None, f'{type(exc).__name__}: {exc}'[:4096]
            self.results.put((kind, result, error))

        self.io_thread = threading.Thread(target=work, name='corridorkey-readiness', daemon=True)
        self.io_thread.start()
        self.status.setText('Working locally: ' + kind + '…')
        self._buttons()

    def inspect(self):
        if self.io_busy or self.close_when_done or self.workspace.closing:
            return
        self.invalidate_prepared()
        parent = self.workspace
        try:
            job = parent.job()
            note = ''
        except (ValueError, OSError) as exc:
            job = None
            note = f' Current form unavailable: {exc}. Running application checks only.'
        selection = parent.asset_selection
        self._begin('checks', lambda: check_readiness(job, selection, cancellation=self.cancellation))
        if note:
            self.status.setText(self.status.text() + note)

    def prepare(self):
        if self.io_busy or self.close_when_done or self.workspace.closing:
            return
        if self.submitted_id is not None:
            self.status.setText('Wait for the submitted render to finish.')
            return
        try:
            job = self.workspace.job()
            start, count = self.start.value(), self.count.value()
            self.prepared = None
            revision = self.prepare_revision
            selection = self.workspace.asset_selection
            self._begin('preparation', lambda: (revision, selection, prepare_short_render(job, start, count)))
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def run(self):
        if self.io_busy or self.prepared is None or self.submitted_id is not None or self.close_when_done:
            return
        self.run_requested.emit(self.prepared)

    def mark_submitted(self, request_id):
        assert self.prepared is not None
        self.render_observation = {
            'request_id': request_id,
            'requested_job': job_to_wire(self.prepared),
            'evidence_kind': 'Worker protocol observations; provider authenticity and model execution are not independently verified by this UI.',
            'actual_inference': 'NOT VERIFIED', 'output_decode': 'NOT VERIFIED', 'display': 'NOT VERIFIED',
            'events': [],
        }
        if self.report is None:
            self.report = {'schema_version': 1, 'type': 'corridorkey_readiness', 'stages': [], 'protected_paths': []}
        protected = list(self.report.get('protected_paths', []))
        config, _ = self.prepared.configurations()
        protected.extend(path for path in (self.prepared.input_path, self.prepared.initial_hint_path,
                                          self.prepared.hint_video_path, config.corrections_path) if path)
        if self.prepared_selection is not None:
            protected.append(str(self.prepared_selection.root))
            if self.prepared_selection.weights is not None:
                protected.append(str(self.prepared_selection.weights))
        else:
            protected.append(str(Path(__file__).resolve().parents[1]))
        self.render_protected = list(dict.fromkeys(protected))
        self.report['render_observation'] = self.render_observation
        self.submitted_id = request_id
        self.prepared = None  # a displayed output reservation is single use
        self.status.setText('Submitted explicit real-render request. Loading/inference are not yet verified; follow the job row.')
        self._buttons()

    def cancel(self):
        self.cancellation.cancel('Readiness operation cancelled by user')
        if self.submitted_id is not None:
            try:
                self.workspace.bridge.send('cancel', {'target': self.submitted_id, 'reason': 'Cancelled from readiness'})
            except (ValueError, OSError, RuntimeError, queue.Full) as exc:
                self.status.setText(str(exc))
                return
        self.status.setText('Cancellation requested; waiting for owned work. Native operations may finish before cancellation is observed.')

    def observe(self, event):
        if self.submitted_id is None or event.get('request_id') != self.submitted_id:
            return
        kind = event['kind']
        payload = event.get('payload', {})
        if self.render_observation is not None:
            events = self.render_observation['events']
            events.append({'kind': kind, 'stage': str(payload.get('stage', ''))[:1024],
                           'error': str(payload.get('error', ''))[:4096]})
            del events[:-32]
        self.status.setText(f'Real-render worker: {kind} · {payload.get("stage", payload.get("error", ""))}')
        if kind in {'result', 'failed', 'cancelled', 'rejected', 'lost'}:
            self.submitted_id = None
            self._buttons()

    def choose_export(self):
        if self.report is None or self.io_busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save fresh readiness report', '', 'JSON (*.json)')
        if path:
            report = json.loads(json.dumps(self.report))
            if self.render_observation is not None:
                report['render_observation'] = json.loads(json.dumps(self.render_observation))
                report['protected_paths'] = list(dict.fromkeys(
                    [*report.get('protected_paths', []), *self.render_protected]))
            self._begin('export', lambda: export_report(report, Path(path)))

    def poll(self):
        try:
            kind, result, error = self.results.get_nowait()
        except queue.Empty:
            self._buttons()
            if self.close_when_done and not self.io_busy:
                self.close_when_done = False
                self.close()
            return
        if error:
            self.status.setText(error)
        elif self.cancellation.cancelled and kind != 'export':
            if kind == 'checks' and isinstance(result, dict) and result.get('status') == 'cancelled':
                self.report = result
                self.details.setPlainText(json.dumps(result, indent=2))
            self.status.setText('Readiness operation cancelled; any partial cancelled check report can be exported.')
        elif kind == 'checks':
            self.report = result
            stages = result.get('stages', [])
            summary = '\n'.join(f'{stage.get("stage", stage.get("id", "Stage"))}: {stage.get("status", "NOT VERIFIED")} · {stage.get("details", stage.get("message", ""))}' for stage in stages) if isinstance(stages, list) else ''
            self.details.setPlainText(summary + '\n\nLocal report:\n' + json.dumps(result, indent=2))
            self.status.setText('Lightweight checks complete. Model loading, actual inference and output decoding/display require explicit real-render evidence.')
        elif kind == 'preparation':
            revision, selection, job = result
            if revision != self.prepare_revision:
                self.status.setText('Range changed during preparation. Prepare the new range explicitly.')
                self._buttons()
                return
            self.prepared = job
            self.prepared_selection = selection
            snapshot = job_to_wire(job)
            config = json.loads(snapshot.pop('config_json'))
            snapshot['requested_settings'] = config
            snapshot['selected_assets'] = ({'root': str(selection.root), 'weights': str(selection.weights) if selection.weights else None} if selection is not None else 'Default application asset tree / worker defaults')
            snapshot['writer_settings'] = json.loads(snapshot.pop('writer_json'))
            snapshot['source_range'] = {'start': config['start_frame'], 'end_exclusive': config['start_frame'] + config['num_frames']}
            self.details.setPlainText('Prepared immutable snapshot. Run explicitly to load selected models, decode this source range, perform requested inference and write the fresh destination. No render has started. Subranges restart recurrent state.\n\n' + json.dumps(snapshot, indent=2))
            self.status.setText('Review source, range, settings and fresh output below. Current workspace settings remain unchanged.')
        else:
            self.status.setText('Local report exported.')
        self._buttons()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def closeEvent(self, event: QCloseEvent):
        self.cancellation.cancel('Readiness dialog closed')
        if self.io_busy:
            self.close_when_done = True
            event.ignore()
        else:
            event.accept()
