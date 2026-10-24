"""Explicit shared-controller recovery requests and actual report inspection."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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

from .local_json import read_object
from .session import JobRequest


def stage2_summary(report: dict[str, Any]) -> str:
    """Summarize current-attempt counters; restored origin records stay historical."""
    labels = {
        'stage2_calls': 'Actual Stage 2 model calls',
        'stage2_frame_evaluations': 'Frames evaluated by Stage 2',
        'completed_prediction_frames': 'Predictions completed (computed + replayed)',
        'batch_hits': 'Verified batches reused',
        'stage2_frame_instances_reused': 'Prediction frame instances reused',
        'stored_batches': 'Batches stored',
        'corrupt_entries': 'Corrupt cache entries found',
        'budget_rejected_batches': 'Batches skipped by cache/staging limits',
        'bytes_loaded': 'Verified prediction bytes loaded',
    }
    totals = dict.fromkeys(labels, 0)
    reports = dict.fromkeys(labels, 0)
    current, historical = [], 0
    for job in report.get('jobs', []):
        if not isinstance(job, dict):
            continue
        if job.get('mode') == 'render' and job.get('new_session_run') is True:
            current.append(job.get('origin_record'))
        elif job.get('mode') in ('verified_existing', 'verified_cache_restore'):
            historical += 1
    if isinstance(report.get('failed_attempt'), dict):
        current.append(report['failed_attempt'])
    for record in current:
        if not isinstance(record, dict):
            continue
        diagnostics = record.get('diagnostics', {})
        if not isinstance(diagnostics, dict):
            continue
        execution = diagnostics.get('execution')
        replay = diagnostics.get('durable_stage2')
        for source in (execution, replay):
            if not isinstance(source, dict):
                continue
            for key in labels:
                value = source.get(key)
                if type(value) is int and value >= 0:
                    totals[key] += value
                    reports[key] += 1
    lines = ['Stage 2: current attempt only; replay is not new neural throughput.']
    for key, label in labels.items():
        total, reported = totals[key], reports[key]
        if reported:
            suffix = ' (some attempt counters unavailable)' if reported < len(current) else ''
            lines.append(f'{label}: {total}{suffix}')
    if not any(reports.values()):
        lines.append('No current-attempt Stage 2 counters recorded.')
    if historical:
        lines.append(f'Completed output jobs verified/restored: {historical}; their origin counters are historical.')
    return '\n'.join(lines)


def writer_prefix_summary(report: dict[str, Any]) -> str:
    """Current-attempt copies/native writes are separate from historical origins."""
    records = [job.get('origin_record') for job in report.get('jobs', [])
               if isinstance(job, dict) and job.get('mode') == 'render' and job.get('new_session_run') is True]
    if isinstance(report.get('failed_attempt'), dict):
        records.append(report['failed_attempt'])
    snapshots = []
    for record in records:
        if isinstance(record, dict) and isinstance(record.get('diagnostics'), dict):
            snapshot = record['diagnostics'].get('durable_writer_prefix')
            if isinstance(snapshot, dict):
                snapshots.append(snapshot)
    if not snapshots:
        return 'No current-attempt EXR writer-prefix counters recorded.'

    def total(key):
        values = [entry.get(key) for entry in snapshots]
        known = [value for value in values if type(value) is int and value >= 0]
        if not known:
            return 'unavailable'
        return str(sum(known)) + (' (partial reporting)' if len(known) != len(values) else '')

    lines = ['EXR writer prefix: current attempt; copied bytes are not new native writes.']
    errors = [str(entry['last_error'])[:500] for entry in snapshots if entry.get('last_error')]
    if errors:
        lines.append('Checkpoint note: ' + '; '.join(errors[:2]))
    lines.append(f"Verified prefix restored: {total('restored_prefix_frames')} frames; {total('restored_files')} files; {total('restored_bytes')} bytes.")
    lines.append(f"New queue submissions: {total('newly_submitted_frames')} frames. Confirmed new native writes: {total('native_new_written_frames')} frames / {total('native_new_written_files')} files.")
    statuses = [entry.get('native_completion_status', 'not_verified') for entry in snapshots]
    completion = 'complete' if all(value == 'complete' for value in statuses) else 'partial' if 'partial' in statuses else 'not verified'
    lines.append('Native completion: ' + completion + '; partial/unverified counts are observed lower bounds.')
    lines.append(f"Sealed during attempt: {total('stored_frames')} frames / {total('stored_files')} files / {total('stored_bytes')} bytes; global cache retention is not guaranteed.")
    lines.append(f"Cache-budget skips: {total('budget_rejected_frames')} frames. Corrupt entries found: {total('corrupt_entries')}.")
    if any(entry.get('checkpoint_disabled') is True for entry in snapshots):
        lines.append('Prefix checkpointing was disabled during this attempt; no inference/render skipping is implied.')
    return '\n'.join(lines)


class RecoveryPanel(QDialog):
    run_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Verified EXR recovery')
        self.resize(1050, 730)
        self.current_request: str | None = None
        self.io_thread: threading.Thread | None = None
        self.results: queue.Queue[tuple[Any, str | None]] = queue.Queue(maxsize=1)
        layout = QVBoxLayout(self)
        note = QLabel('Uses the existing recoverable EXR controller through the owned worker. Select EXR jobs with explicit scheduling. Resume re-verifies journal/outputs and may restore bytes or recompute; replay is not new neural FPS. Repeat the same cache options with explicit Verify / resume. Verified Stage 2 predictions avoid Stage 2 model calls only. Durable EXR prefixes can copy verified completed frames into fresh output only on explicit Verify / resume; native writes cover the remaining suffix. Full upstream chronology and rendering still run. Native video continuation and direct seeking are not implemented. Cache budget covers logical sealed entries only, not staging copies/all artifacts/disk/RAM/VRAM. A durable queue workspace is required for immutable batch snapshots.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.batch, self.workspace, self.budget, self.report = QLineEdit(), QLineEdit(), QLineEdit('1073741824'), QLineEdit()
        self.report.setReadOnly(True)
        for label, widget in [('Batch manifest', self.batch), ('Separate cache workspace', self.workspace),
                              ('Maximum cache bytes (0 disables bundles)', self.budget), ('Actual report', self.report)]:
            form.addRow(label, widget)
        layout.addLayout(form)
        flags = QHBoxLayout()
        self.foundations = QCheckBox('Durable IID foundations')
        self.rvm = QCheckBox('Durable chronological RVM')
        self.hann = QCheckBox('Durable Hann event replay')
        self.stage2 = QCheckBox('Durable Stage 2 predictions')
        self.stage2.setToolTip('Optional exact prediction batches in the shared cache budget. Repeat this option on explicit Verify / resume. Oversize entries may be skipped; inference quality and batch grouping stay unchanged.')
        for control in (self.foundations, self.rvm, self.hann, self.stage2):
            flags.addWidget(control)
        layout.addLayout(flags)
        self.finishing_edits = QCheckBox('Reuse Stage 2 across expansion / feather / despill edits')
        self.finishing_edits.setToolTip('Requires durable Stage 2. Private v5 reuses only whole matching RGB/baton calls across final controls, correction and first/full hint edits. Changed inputs, grouping or execution recompute. Full current assets/source/recipe validation and fresh output remain mandatory. Upstream reuse has a separate conservative policy. Decode, chronology, constraints and writing still run; no GPU speed claim.')
        layout.addWidget(self.finishing_edits)
        self.upstream_finishing_edits = QCheckBox('Reuse enabled upstream caches across expansion / feather / despill edits')
        self.upstream_finishing_edits.setToolTip('Requires an enabled IID/RVM/Hann cache. IID keeps conservative v2-v4 dependencies. RVM/Hann use temporal-only v5 for exact matching input/state events across corrections and first/full hint edits. Current constraints rebuild outgoing aux carry after raw Hann hits; changed next events recompute. Full current source/assets/recipe admission, fresh output and cancellation checks remain mandatory. Stage 2 has a separate opt-in. No GPU speed or producer-batch roundoff claim.')
        layout.addWidget(self.upstream_finishing_edits)
        self.writer_prefix = QCheckBox('Durable EXR writer prefix')
        self.writer_prefix.setToolTip('Opt in to sealed, contiguous EXR frames across all requested outputs within the shared cache budget. Only explicit Verify / resume copies a verified prefix into a fresh destination. Original partial outputs remain untouched; inference/render chronology still runs. No video continuation or direct seeking.')
        layout.addWidget(self.writer_prefix)
        buttons = QHBoxLayout()
        for label, callback in [('Batch…', self.choose_batch), ('Cache workspace…', self.choose_workspace),
                                ('Run batch', lambda: self.request(False)), ('Verify / resume', lambda: self.request(True)),
                                ('Cancel current', self.cancel_requested.emit), ('Workspace metadata', self.inspect_workspace),
                                ('Inspect report', self.inspect_current), ('Open report…', self.choose_report)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel('No recovery request submitted. Nothing resumes automatically.')
        self.snapshot_label = QLabel('No queued batch snapshot')
        self.snapshot_label.setWordWrap(True)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(self.snapshot_label)
        self.replay_summary = QLabel('No Stage 2 recovery report inspected.')
        self.replay_summary.setWordWrap(True)
        layout.addWidget(self.replay_summary)
        self.writer_summary = QLabel('No EXR writer-prefix recovery report inspected.')
        self.writer_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.writer_summary.setWordWrap(True)
        layout.addWidget(self.writer_summary)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)

    @property
    def io_busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    def choose_batch(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select existing inference batch manifest', '', 'JSON (*.json)')
        if path:
            self.batch.setText(path)

    def choose_workspace(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose separate empty/owned recovery workspace')
        if path:
            self.workspace.setText(path)

    def prepare_job(self, parent: Path, job: JobRequest, *, weights: Path | None = None):
        if self.io_busy:
            raise ValueError('Wait for current recovery I/O before preparing a recipe.')
        self.io_thread = threading.Thread(target=self._prepare, args=(parent, job, weights), daemon=True)
        self.io_thread.start()
        self.status.setText('Preparing immutable one-job recipe; nothing submitted.')
        self.show()

    def _prepare(self, parent: Path, job: JobRequest, weights: Path | None):
        from .desktop_recipe import prepare_recipe
        try:
            path = prepare_recipe(parent, job, weights=weights)
            self.results.put(({'prepared_batch': str(path), 'scope': 'Recipe prepared; choose a separate cache workspace and Run explicitly. Reuse this recipe for Verify/resume. Nothing submitted.'}, None))
        except Exception as exc:
            logging.getLogger(__name__).exception('Recovery recipe preparation failed')
            self.results.put((None, f'{type(exc).__name__}: {exc}'))

    def request(self, resume: bool):
        try:
            if self.io_busy:
                raise ValueError('Wait for recovery I/O before submission.')
            if not self.batch.text().strip() or not self.workspace.text().strip():
                raise ValueError('Choose a batch manifest and separate recovery workspace.')
            if self.finishing_edits.isChecked() and not self.stage2.isChecked():
                raise ValueError('Finishing-edit reuse requires Durable Stage 2 predictions.')
            if self.upstream_finishing_edits.isChecked() and not any(control.isChecked() for control in
                    (self.foundations, self.rvm, self.hann)):
                raise ValueError('Upstream finishing-edit reuse requires an enabled IID/RVM/Hann cache.')
            payload = {'batch': str(Path(self.batch.text()).resolve()), 'workspace': str(Path(self.workspace.text()).resolve()),
                       'max_cache_bytes': int(self.budget.text()), 'resume': resume,
                       'durable_foundations': self.foundations.isChecked(), 'durable_rvm': self.rvm.isChecked(),
                       'durable_hann': self.hann.isChecked(), 'durable_stage2': self.stage2.isChecked(),
                       'durable_writer_prefix': self.writer_prefix.isChecked(),
                       'reuse_stage2_finishing_edits': self.finishing_edits.isChecked(),
                       'reuse_upstream_finishing_edits': self.upstream_finishing_edits.isChecked()}
            self.run_requested.emit(payload)
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def load_payload(self, payload: dict[str, Any]):
        if self.io_busy:
            self.status.setText('Wait for recovery I/O before loading settings.')
            return
        self.batch.setText(payload['batch'])
        self.workspace.setText(payload['workspace'])
        self.budget.setText(str(payload['max_cache_bytes']))
        self.foundations.setChecked(payload.get('durable_foundations', False))
        self.rvm.setChecked(payload.get('durable_rvm', False))
        self.hann.setChecked(payload.get('durable_hann', False))
        self.stage2.setChecked(payload.get('durable_stage2', False))
        self.writer_prefix.setChecked(payload.get('durable_writer_prefix', False))
        self.finishing_edits.setChecked(payload.get('reuse_stage2_finishing_edits', False))
        self.upstream_finishing_edits.setChecked(payload.get('reuse_upstream_finishing_edits', False))
        self.current_request = None
        self.report.clear()
        self.replay_summary.setText('No Stage 2 recovery report inspected.')
        self.writer_summary.setText('No EXR writer-prefix recovery report inspected.')
        self.status.setText('Saved request settings loaded. Choose Run or Verify/resume explicitly; no work submitted.')
        self.show()

    def track(self, request_id: str):
        self.current_request = request_id
        self.report.clear()
        self.replay_summary.setText('No Stage 2 recovery report inspected.')
        self.writer_summary.setText('No EXR writer-prefix recovery report inspected.')
        self.snapshot_label.setText('Waiting for durable batch snapshot acknowledgement')
        self.status.setText('Recovery submitted locally; waiting for persistence/worker acceptance.')

    def observe(self, event: dict[str, Any]):
        if self.current_request is None or event.get('request_id') != self.current_request:
            return
        payload = event.get('payload', {})
        if event['kind'] == 'recorded':
            self.snapshot_label.setText('Queued immutable batch: ' + str(payload.get('batch', '')))
        if event['kind'] == 'result':
            self.report.setText(payload['record']['report_path'])
        detail = payload.get('error') or payload.get('stage') or ''
        self.status.setText(f"{event['kind']} · {detail}")

    def inspect_workspace(self):
        if self.workspace.text().strip():
            self.inspect(Path(self.workspace.text()), metadata=True)

    def inspect_current(self):
        if self.report.text():
            self.inspect(Path(self.report.text()))

    def choose_report(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open actual recovery report', '', 'JSON (*.json)')
        if path:
            self.report.setText(path)
            self.inspect(Path(path))

    def inspect(self, path: Path, *, metadata: bool = False):
        if self.io_busy:
            self.status.setText('Wait for current recovery report/metadata I/O.')
            return
        self.io_thread = threading.Thread(target=self._inspect, args=(path, metadata), daemon=True)
        self.io_thread.start()

    def _inspect(self, path: Path, metadata: bool):
        try:
            if metadata:
                marker, _ = read_object(path / 'workspace.json', max_bytes=4 * 1024 * 1024)
                if marker.get('kind') != 'corridorkey_recovery' or type(marker.get('schema_version')) is not int or marker['schema_version'] != 1:
                    raise ValueError('Not a supported recovery workspace.')
                data = {'workspace': str(path), 'journals': [str(p) for p in sorted((path / 'journals').glob('*.json'))[:128]],
                        'reports': [str(p) for p in sorted((path / 'reports').glob('*.json'))[:128]],
                        'scope': 'Metadata only; payload/backend/outputs NOT VERIFIED. No owner inferred from PID/files.'}
            else:
                report, _ = read_object(path, max_bytes=32 * 1024 * 1024)
                if report.get('kind') != 'recoverable_exr_batch' or type(report.get('schema_version')) is not int or report['schema_version'] != 1 or not isinstance(report.get('jobs'), list):
                    raise ValueError('Not a supported recovery report.')
                priority = ('kind', 'status', 'note', 'error', 'jobs')
                ordered = {key: report[key] for key in priority if key in report}
                ordered.update({key: value for key, value in report.items() if key not in ordered})
                data = {'scope': 'Saved controller report, not a new output validation. Origin timings are historical; replay is not neural FPS.',
                        'report': ordered}
            self.results.put((data, None))
        except Exception as exc:
            logging.getLogger(__name__).exception('Recovery report inspection failed')
            self.results.put((None, f'{type(exc).__name__}: {exc}'))

    def poll(self):
        try:
            data, error = self.results.get_nowait()
        except queue.Empty:
            return
        if error is not None:
            self.status.setText(error)
        else:
            if 'prepared_batch' in data:
                self.batch.setText(data['prepared_batch'])
                self.current_request = None
                self.report.clear()
                self.snapshot_label.setText('Prepared recipe: ' + data['prepared_batch'])
            if isinstance(data.get('report'), dict):
                self.replay_summary.setText(stage2_summary(data['report']))
                self.writer_summary.setText(writer_prefix_summary(data['report']))
            else:
                self.replay_summary.setText('No Stage 2 recovery report inspected.')
                self.writer_summary.setText('No EXR writer-prefix recovery report inspected.')
            text = json.dumps(data, indent=2)
            self.details.setPlainText(text[:2 * 1024 * 1024] + ('\n[View truncated; actual report unchanged.]' if len(text) > 2 * 1024 * 1024 else ''))
            self.status.setText(data['scope'])
