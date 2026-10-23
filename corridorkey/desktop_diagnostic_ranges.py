"""Explicit diagnostic navigation and fresh range preparation; never submission."""
from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .control import CancellationToken, CancelledError, ProgressEvent
from .session import JobRequest

MAX_RESULT_BYTES = 32 * 1024 * 1024
PAGE_ROWS = 500


@dataclass(frozen=True)
class RangeRequest:
    operation: str
    job: JobRequest | None = None
    record: bytes | None = None
    snapshot: bytes | None = None
    start: int = 0
    end: int = 0
    parent: str = ''
    full_quality: bool = True
    correction_policy: str = 'reject'
    initial_hint_policy: str = 'reject'
    protected_paths: tuple[str, ...] = ()


def run_operation(request: RangeRequest, *, check, progress):
    from .diagnostic_ranges import (
        DiagnosticSnapshot,
        candidates,
        inspect_diagnostics,
        prepare_range,
        revalidate,
    )

    if request.operation == 'inspect' and request.job is not None and request.record is not None:
        snapshot = inspect_diagnostics(request.job, json.loads(request.record), check=check, progress=progress)
        return {'status': 'complete', 'snapshot': snapshot.to_dict(),
                'candidates': [asdict(candidate) for candidate in candidates(snapshot)]}
    if request.operation == 'preview' and request.snapshot is not None:
        snapshot = DiagnosticSnapshot(request.snapshot.decode('utf-8'))
        revalidate(snapshot, check=check, progress=progress)
        source = snapshot.to_dict()['source']
        return {'status': 'complete', 'options': {'path': source['path'], 'index': request.start,
                'linear': source['scene_linear'], 'source_colour_profile': source['source_colour_profile'],
                'alpha_mode': source['alpha_mode'], 'expected_hw': source['native_hw']}}
    if request.operation == 'prepare' and request.snapshot is not None:
        snapshot = DiagnosticSnapshot(request.snapshot.decode('utf-8'))
        progress(ProgressEvent('prepare_range', 0, 1))
        job = prepare_range(snapshot, request.start, request.end, Path(request.parent),
                            full_quality=request.full_quality, correction_policy=request.correction_policy,
                            initial_hint_policy=request.initial_hint_policy,
                            protected_paths=request.protected_paths, check=check)
        if not isinstance(job, JobRequest):
            raise TypeError('Range preparation must return an immutable shared job.')
        # Preparation can reserve a directory. Do not cancel or run ancillary
        # callbacks after the authoritative successful return.
        return {'status': 'complete', 'job': asdict(job), 'reserved_root': str(Path(job.output_dir).parent)}
    raise ValueError('Choose a captured diagnostic selection before starting.')


class DiagnosticRangeTask:
    def __init__(self, request: RangeRequest, runner: Callable = run_operation):
        self.request, self.runner = request, runner
        self.cancellation = CancellationToken()
        self.results: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest: ProgressEvent | None = None
        self.thread = threading.Thread(target=self._run, name='diagnostic-ranges', daemon=True)
        self.thread.start()

    @property
    def busy(self):
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Diagnostic range operation cancelled by user')

    def _progress(self, event: ProgressEvent):
        with self.lock:
            self.latest = event

    def progress(self):
        with self.lock:
            return self.latest

    def _run(self):
        data: dict[str, Any] = {}
        raw = b'{"status":"failed","error":"Unexpected diagnostic worker termination"}'
        try:
            self.cancellation.check()
            data = self.runner(self.request, check=self.cancellation.check, progress=self._progress)
            if not isinstance(data, dict):
                raise TypeError('Diagnostic operation must return a terminal object.')
            raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
            if len(raw) > MAX_RESULT_BYTES:
                raise ValueError('Diagnostic interface result exceeds 32 MiB.')
        except CancelledError as exc:
            raw = json.dumps({'status': 'cancelled', 'error': str(exc)[:4096]}).encode()
        except BaseException as exc:
            logging.getLogger(__name__).exception('Diagnostic range operation failed')
            fallback: dict[str, Any] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:4096]}
            if self.request.operation == 'prepare' and isinstance(data, dict) and isinstance(data.get('reserved_root'), str):
                fallback['reserved_root'] = data['reserved_root'][:4096]
            raw = json.dumps(fallback).encode()
        finally:
            self.results.put(raw)
            self.done.set()


class DiagnosticRangesPanel(QDialog):
    preview_requested = Signal(object, str)
    prepare_started = Signal(str)
    prepared_job = Signal(object, str)

    def __init__(self, parent=None, *, runner: Callable = run_operation):
        super().__init__(parent)
        self.runner = runner
        self.task: DiagnosticRangeTask | None = None
        self.snapshot: bytes | None = None
        self.context_id: str | None = None
        self.revision = 0
        self.task_revision = -1
        self.task_context = ''
        self.selection: RangeRequest | None = None
        self.rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.close_when_done = False
        self.setWindowTitle('CorridorKey · Diagnostic ranges')
        self.resize(1150, 780)
        layout = QVBoxLayout(self)
        note = QLabel('Inspect recorded Quick Key / shot heuristics and prepare a standalone source range. Flags are not calibrated quality scores. Trace events can precede writer submission; missing coverage is not a pass. Trimmed processing starts fresh chronological/RVM state and may differ from the same frames in a longer render. Preparation reserves a fresh folder and loads future settings only; Render remains explicit.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel('Select a terminal job and inspect explicitly. No work submitted.')
        self.coverage = QLabel('No diagnostics inspected.')
        self.result_path = QLabel('No preparation folder reserved.')
        for label in (self.status, self.coverage, self.result_path):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(label)
        toolbar = QHBoxLayout()
        self.inspect_button = QPushButton('Inspect captured selection')
        self.inspect_button.clicked.connect(self.inspect)
        toolbar.addWidget(self.inspect_button)
        self.view = QComboBox()
        self.view.addItems(['Suggested ranges', 'Diagnostic events'])
        self.view.currentIndexChanged.connect(self._change_view)
        toolbar.addWidget(self.view)
        toolbar.addWidget(QLabel('Page'))
        self.page = QSpinBox()
        self.page.setRange(1, 1)
        self.page.valueChanged.connect(self._render_rows)
        toolbar.addWidget(self.page)
        self.page_note = QLabel('0 rows')
        toolbar.addWidget(self.page_note)
        layout.addLayout(toolbar)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Source In', 'Source Out (exclusive)', 'Selected-relative frame(s)', 'Observation', 'Recorded reasons'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._stage_selection)
        layout.addWidget(self.table, 1)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        interval = QHBoxLayout()
        self.range_in, self.range_out = QLineEdit(), QLineEdit()
        interval.addWidget(self.range_in)
        interval.addWidget(QLabel('Out (exclusive)'))
        interval.addWidget(self.range_out)
        form.addRow('Absolute source In', interval)
        self.parent_path = QLineEdit()
        parent_row = QHBoxLayout()
        parent_row.addWidget(self.parent_path)
        choose = QPushButton('Existing parent…')
        choose.clicked.connect(self.choose_parent)
        parent_row.addWidget(choose)
        form.addRow('New range folder parent', parent_row)
        self.full_quality = QCheckBox('Full quality: native scale, full-rate neural processing')
        self.full_quality.setChecked(True)
        self.full_quality.setToolTip('Checked resets spatial scale to 1 and disables Quick Key in the new job. Uncheck explicitly to preserve its original spatial/Quick mode and rebase cut/reset indices.')
        form.addRow(self.full_quality)
        self.initial_hint_policy, self.correction_policy = QComboBox(), QComboBox()
        for control in (self.initial_hint_policy, self.correction_policy):
            control.addItem('Reject incompatible attachment', 'reject')
            control.addItem('Drop from the new job explicitly', 'drop')
        form.addRow('First-frame hint when In changes', self.initial_hint_policy)
        form.addRow('Corrections when selected range changes', self.correction_policy)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.preview_button, self.prepare_button, self.cancel_button = QPushButton('Preview source In'), QPushButton('Prepare range settings'), QPushButton('Cancel operation')
        for button, callback in ((self.preview_button, self.preview), (self.prepare_button, self.prepare), (self.cancel_button, self.cancel)):
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        for control in (self.range_in, self.range_out, self.parent_path):
            control.textChanged.connect(self._prepare_inputs_changed)
        self.full_quality.toggled.connect(self._prepare_inputs_changed)
        self.initial_hint_policy.currentIndexChanged.connect(self._prepare_inputs_changed)
        self.correction_policy.currentIndexChanged.connect(self._prepare_inputs_changed)
        self._controls()

    @property
    def io_busy(self):
        return self.task is not None and self.task.busy

    def invalidate_context(self):
        self.revision += 1
        self.context_id = None
        self.selection = None
        self.snapshot = None
        self.rows, self.events = [], []
        self.table.setRowCount(0)
        self.coverage.setText('Selection changed; inspect again. Historical source bytes are not authenticated.')
        if self.task is not None and self.task.busy:
            self.task.cancel()
        self._controls()

    def inspect_selection(self, job: JobRequest, record: dict, context_id: str, existing_parent: str = '', *, protected_paths=()):
        self.invalidate_context()
        self.close_when_done = False
        self.context_id = context_id
        self.selection = RangeRequest('inspect', job=job, record=json.dumps(record, allow_nan=False).encode(),
                                      protected_paths=tuple(str(path) for path in protected_paths))
        self.parent_path.setText(existing_parent)
        self.full_quality.setChecked(True)
        self.initial_hint_policy.setCurrentIndex(0)
        self.correction_policy.setCurrentIndex(0)
        self.show()
        if self.io_busy:
            self.status.setText('Previous operation is stopping. Inspect this captured selection after it reaches terminal state.')
            self._controls()
            return False
        return self.inspect()

    def inspect(self):
        if self.selection is None or self.context_id is None:
            self.status.setText('Select a terminal job first.')
            return False
        return self._start(self.selection)

    def _start(self, request: RangeRequest):
        if self.io_busy or self.close_when_done or self.context_id is None:
            self.status.setText('Wait for the owned operation to reach terminal state.')
            return False
        if request.operation == 'inspect':
            self.snapshot = None
            self.rows, self.events = [], []
            self.table.setRowCount(0)
        self.task_revision, self.task_context = self.revision, self.context_id or ''
        self.task = DiagnosticRangeTask(request, self.runner)
        self.status.setText({'inspect': 'Inspecting local diagnostics…', 'preview': 'Revalidating diagnostic selection before preview…', 'prepare': 'Validating and preparing fresh range settings…'}[request.operation])
        self._controls()
        return True

    def _prepare_inputs_changed(self, *_):
        # Even A->B->A edits revoke a running preparation's authorization.
        if self.task is not None and self.task.busy and self.task.request.operation in ('prepare', 'preview'):
            self.revision += 1
            self.task.cancel()
        self._controls()

    def _interval(self):
        start, end = int(self.range_in.text()), int(self.range_out.text())
        if self.snapshot is None:
            raise ValueError('Inspect a current selection first.')
        source = json.loads(self.snapshot)['source']
        if not source['start_frame'] <= start < end <= source['start_frame'] + source['num_frames']:
            raise ValueError('Choose a nonempty half-open interval within the observed source range.')
        return start, end

    def preview(self):
        try:
            if self.io_busy or self.snapshot is None or self.context_id is None:
                raise ValueError('Wait for current inspection before previewing.')
            start, _ = self._interval()
            self._start(RangeRequest('preview', snapshot=self.snapshot, start=start))
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))

    def prepare(self):
        try:
            if self.snapshot is None or self.context_id is None or self.selection is None or self.io_busy:
                raise ValueError('Inspect a current selection and wait for owned I/O first.')
            start, end = self._interval()
            if not self.parent_path.text().strip():
                raise ValueError('Choose an existing parent for the fresh range folder.')
            request = RangeRequest('prepare', snapshot=self.snapshot, start=start, end=end,
                                   parent=self.parent_path.text(), full_quality=self.full_quality.isChecked(),
                                   correction_policy=self.correction_policy.currentData(),
                                   initial_hint_policy=self.initial_hint_policy.currentData(),
                                   protected_paths=self.selection.protected_paths)
            self.prepare_started.emit(self.context_id)
            self._start(request)
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))

    def choose_parent(self):
        path = QFileDialog.getExistingDirectory(self, 'Existing parent for a fresh standalone range folder')
        if path:
            self.parent_path.setText(path)

    def _data_rows(self):
        return self.rows if self.view.currentIndex() == 0 else self.events

    def _change_view(self):
        self.page.setValue(1)
        self._render_rows()

    def _render_rows(self, *_):
        rows = self._data_rows()
        self.page.setMaximum(max(1, (len(rows) + PAGE_ROWS - 1) // PAGE_ROWS))
        offset = (self.page.value() - 1) * PAGE_ROWS
        visible = rows[offset:offset + PAGE_ROWS]
        self.table.setRowCount(0)
        self.table.setRowCount(len(visible))
        source_start = json.loads(self.snapshot)['source']['start_frame'] if self.snapshot is not None else 0
        for index, row in enumerate(visible):
            start: int = row['start_frame'] if 'start_frame' in row else row['frame']
            end = row.get('end_frame', start + 1)
            relative = f'[{start - source_start}, {end - source_start})'
            observed = str(row['observed_frames']) + ' trace event(s)' if 'observed_frames' in row else 'Trace event' if row.get('observed') else 'Summary only; trace metrics unavailable'
            values = (start, end, relative, observed, ', '.join(row['reasons']))
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(str(value)))
        self.page_note.setText(f'{offset + 1 if rows else 0}–{offset + len(visible)} of {len(rows)} rows; 500 per page')
        self.table.resizeColumnsToContents()

    def _stage_selection(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        index = (self.page.value() - 1) * PAGE_ROWS + selected[0].row()
        row = self._data_rows()[index]
        start: int = row['start_frame'] if 'start_frame' in row else row['frame']
        self.range_in.setText(str(start))
        self.range_out.setText(str(row.get('end_frame', start + 1)))

    def _controls(self):
        busy = self.io_busy
        self.inspect_button.setEnabled(not busy and self.selection is not None and not self.close_when_done)
        ready = not busy and self.snapshot is not None and self.context_id is not None and not self.close_when_done
        self.preview_button.setEnabled(ready)
        self.prepare_button.setEnabled(ready)
        self.cancel_button.setEnabled(busy)

    def cancel(self):
        if self.task is not None and self.task.busy:
            self.revision += 1
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual terminal state.')

    def poll(self):
        if self.task is None:
            return
        progress = self.task.progress()
        if progress is not None and self.task.busy:
            self.status.setText(f'{progress.stage}: {progress.completed}/{progress.total}')
        try:
            raw = self.task.results.get_nowait()
        except queue.Empty:
            self._controls()
            if self.close_when_done and not self.io_busy:
                self.close()
            return
        data = json.loads(raw)
        reserved = data.get('reserved_root')
        if isinstance(reserved, str):
            self.result_path.setText('Reserved range folder: ' + reserved)
        current = self.task_revision == self.revision and self.task_context == self.context_id
        if not current:
            self.status.setText('Superseded operation finished; no settings applied.' + (' Reserved folder remains: ' + reserved if isinstance(reserved, str) else ''))
        elif data.get('status') != 'complete':
            if self.task.request.operation == 'preview':
                self.snapshot = None
                self.rows, self.events = [], []
                self.table.setRowCount(0)
                self.coverage.setText('Selection failed revalidation; inspect again before using it.')
            self.status.setText(str(data.get('error') or data.get('status') or 'Operation failed')[:4096])
        elif self.task.request.operation == 'inspect':
            self.snapshot = json.dumps(data['snapshot'], allow_nan=False, separators=(',', ':')).encode()
            self.rows, self.events = data['candidates'], data['snapshot']['events']
            coverage = data['snapshot']['coverage']
            self.coverage.setText(f"Observed source [{data['snapshot']['source']['start_frame']}, {data['snapshot']['source']['start_frame'] + data['snapshot']['source']['num_frames']}). Coverage: {coverage['quick_observed_frames']} Quick Key / {coverage['shot_observed_frames']} shot events of {coverage['expected_frames']} frames; {coverage['summary_only_events']} summary-only events. Historical source identity NOT VERIFIED.")
            self.status.setText('Recorded heuristic events; no quality verdict. ' + ' '.join(data['snapshot'].get('warnings', []))[:4096])
            self.page.setValue(1)
            self._render_rows()
        elif self.task.request.operation == 'preview':
            options = dict(data['options'])
            options['expected_hw'] = tuple(options['expected_hw'])
            self.status.setText('Current diagnostic selection revalidated; source preview requested. No render submitted.')
            self.preview_requested.emit(options, self.task_context)
        else:
            job = JobRequest(**data['job'])
            self.status.setText('Fresh range settings prepared; no render submitted. RVM restarts at the selected In.')
            self.prepared_job.emit(job, self.task_context)
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close()

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
