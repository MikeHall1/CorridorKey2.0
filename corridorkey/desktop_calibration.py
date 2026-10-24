"""Explicit bounded calibration capture/verification and owned-run presentation.

Opening this dialog does not inspect files or initialize a model. Metadata and
hashing require separate user actions. Recommendations have no apply signal.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .asset_inventory import AssetSelection
from .control import CancellationToken
from .session import JobRequest

MAX_TEXT_BYTES = 64 * 1024


def _short(value: Any, maximum: int = 2048) -> str:
    if value is None:
        return 'unavailable'
    if type(value) not in (str, int, float, bool):
        return f'{type(value).__name__} (details retained in the local report)'
    return str(value)[:maximum]


def calibration_view(value: dict[str, Any]) -> str:
    """Bound GUI text without serializing full records, manifests or file lists."""
    lines = []
    labels = {
        'status': 'Status', 'identity': 'Captured identity', 'plan_identity': 'Plan identity',
        'binding_identity': 'Measured binding identity', 'expiry_note': 'Expiry / validity',
        'asset_root': 'Selected asset root', 'weights': 'Selected weights',
        'scratch_parent': 'Scratch parent', 'report_parent': 'Report parent',
        'source': 'Source', 'sample_start': 'Source In', 'frame_count': 'Sample frames',
        'total_jobs': 'Actual jobs including warmups', 'total_frames': 'Actual frames including warmups',
        'warmup_jobs': 'Unmeasured warmup jobs', 'read_bytes_required': 'Bytes required for verification',
        'read_budget': 'Read budget bytes', 'max_jobs': 'Job budget', 'max_total_frames': 'Frame budget',
        'max_wall_seconds': 'Wall budget seconds', 'max_output_bytes': 'Output budget bytes',
        'created_utc': 'Created UTC', 'expires_utc': 'Expiry UTC', 'validity': 'Validity',
        'expiry_status': 'Expiry status', 'report_path': 'Local report', 'cleanup_error': 'Cleanup blocker',
        'error': 'Error', 'artifact_error': 'Artifact blocker', 'scope': 'Scope', 'warning': 'Limits',
        'excluded_trials': 'Excluded trials', 'duration_scope': 'Duration estimate limits',
        'recommendation_scope': 'Recommendation scope', 'quality_objective': 'Calibration objective',
    }
    for key, label in labels.items():
        if key in value and value[key] is not None:
            lines.append(f'{label}: {_short(value[key])}')
    for name in ('sample', 'budgets', 'limits', 'progress', 'hardware', 'recommendations', 'expiry', 'retained_paths'):
        group = value.get(name)
        if isinstance(group, dict):
            lines.append(name.replace('_', ' ').title() + ':')
            for index, (key, item) in enumerate(group.items()):
                if index >= 24:
                    lines.append('  Further fields retained in the local report.')
                    break
                lines.append(f'  {str(key)[:80]}: {_short(item, 1024)}')
    for name in ('profiles', 'case_summaries', 'groups', 'recommendations', 'observations', 'duration_estimates', 'failures', 'exclusions', 'warnings', 'missing_files', 'missing', 'quality_constraints'):
        rows = value.get(name)
        if isinstance(rows, (list, tuple)):
            lines.append(f'{name.replace("_", " ").title()} ({len(rows)}):')
            for row in rows[:72]:
                if isinstance(row, dict):
                    parts = []
                    for key in ('id', 'identifier', 'profile', 'purpose', 'state', 'phase', 'repetition',
                                'status', 'model_scale', 'quick_key', 'median_seconds', 'min_seconds',
                                'max_seconds', 'throughput_fps', 'median_fps', 'successful_repetitions', 'eligible', 'basis',
                                'sample_seconds_range', 'frames', 'count', 'scope', 'first_rendered_batch_seconds',
                                'processing_seconds', 'peak_allocated_vram_bytes', 'peak_reserved_vram_bytes', 'resource_scope',
                                'latency_scope', 'error', 'reason', 'path'):
                        if key in row:
                            parts.append(f'{key.replace("_", " ")}: {_short(row[key], 512)}')
                    if 'sample_seconds_range' in row and isinstance(row['sample_seconds_range'], (list, tuple)):
                        parts.append('sample seconds range: ' + ', '.join(_short(item, 64) for item in row['sample_seconds_range'][:2]))
                    job = row.get('job')
                    if isinstance(job, dict) and isinstance(job.get('config_json'), str) and len(job['config_json']) <= MAX_TEXT_BYTES:
                        settings = json.loads(job['config_json'])
                        if isinstance(settings, dict):
                            for key in ('model_scale', 'hann_chunk', 'hann_stride', 'stage2_batch', 'low_vram',
                                        'quick_key', 'quick_interval', 'foundation_cache_mb', 'exr_dtype', 'output_format'):
                                if key in settings:
                                    parts.append(f'{key}={_short(settings[key], 128)}')
                    duration = row.get('estimated_render_duration')
                    if isinstance(duration, dict):
                        for key in ('status', 'frames', 'min_seconds', 'central_seconds', 'max_seconds',
                                    'min_fps', 'median_fps', 'max_fps', 'scope'):
                            if key in duration:
                                parts.append(f'render duration {key.replace("_", " ")}: {_short(duration[key], 512)}')
                    overrides = row.get('overrides')
                    if isinstance(overrides, dict):
                        parts.extend(f'{str(key)[:64]}={_short(item, 128)}' for key, item in list(overrides.items())[:24])
                    lines.append('  ' + (' · '.join(parts) if parts else 'Details retained in the local report.'))
                else:
                    lines.append('  ' + _short(row, 1024))
            if len(rows) > 72:
                lines.append(f'  {len(rows) - 72} further entries retained in the local report.')
    lines.append('Read-only recommendations. No draft or queued settings are changed. Cold process is not cold filesystem. '
                 'Sample timing does not establish visual quality or a universal hardware guarantee. '
                 'Missing CUDA/model/resource evidence remains NOT VERIFIED.')
    raw = '\n'.join(lines).encode('utf-8')
    if len(raw) > MAX_TEXT_BYTES:
        raw = raw[:MAX_TEXT_BYTES - 100]
        return raw.decode('utf-8', errors='ignore') + '\nFurther details retained in the local report.'
    return raw.decode('utf-8')


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError('Duplicate profile JSON field.')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f'Nonfinite profile JSON number: {value}')


@dataclass(frozen=True)
class _Capture:
    job: JobRequest
    selection: AssetSelection
    sample_start: int
    frame_count: int
    profiles: tuple[Any, ...]
    scratch_parent: Path
    report_parent: Path
    limits: Any
    quality_policy_path: Path | None = None

    def arguments(self) -> dict[str, Any]:
        return {'sample_start': self.sample_start, 'frame_count': self.frame_count,
                'profiles': self.profiles, 'asset_selection': self.selection,
                'scratch_parent': self.scratch_parent, 'report_parent': self.report_parent,
                'limits': self.limits, 'quality_policy_path': self.quality_policy_path}


def _task_factory(plan):
    from .calibration_task import CalibrationTask
    return CalibrationTask(plan)


class CalibrationDialog(QDialog):
    run_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(self, job: JobRequest | None, parent=None, *, frame=0,
                 selection: AssetSelection | None = None, factory=None):
        super().__init__(parent)
        self.job = job
        self.captured_selection = selection or AssetSelection(Path(__file__).resolve().parents[1])
        self.factory = factory or _task_factory
        self.plan = None
        self.capture = None
        self.inventory = None
        self.task = None
        self.waiting = False
        self._closing = False
        self._revision = 0
        self._shown_task_revision = -1
        self._io_serial = 0
        self._io_thread: threading.Thread | None = None
        self._io_results: queue.Queue[tuple[int, int, str, Any, str | None]] = queue.Queue(maxsize=1)
        self._cancellation = CancellationToken()
        self.setWindowTitle('CorridorKey · Machine calibration')
        self.resize(980, 820)
        layout = QVBoxLayout(self)
        intro = QLabel('Explicit local sample measurements through the shared CUDA engine. Capture inspects metadata; '
                       'Verify reads the selected source/model files under your budget. Start waits for existing processing '
                       'ownership to retire. No model work occurs until Start. Recommendations stay read-only.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.source = QLabel(_short(job.input_path if job is not None else 'No source captured; open calibration from a configured job.'))
        self.source.setWordWrap(True)
        layout.addWidget(self.source)
        form = QFormLayout()
        self.sample_start = QLineEdit(str(frame))
        self.frame_count = self._spin(1, 120, 16)
        self.repetitions = self._spin(3, 7, 3)
        self.max_jobs = self._spin(1, 168, 72)
        self.max_total_frames = self._spin(1, 20160, 8640)
        self.max_wall_seconds = self._spin(1, 86400, 900)
        self.output_budget = self._spin(1, 1048576, 512)
        self.read_budget = self._spin(0, 1048576, 0)
        self.scratch_parent, self.report_parent = QLineEdit(), QLineEdit()
        for label, widget in (('Sample source In (absolute)', self.sample_start), ('Sample frames', self.frame_count),
                              ('Repetitions per profile', self.repetitions), ('Maximum actual jobs', self.max_jobs),
                              ('Maximum total frames', self.max_total_frames), ('Wall budget seconds', self.max_wall_seconds),
                              ('Output budget MiB', self.output_budget), ('Verification read budget MiB', self.read_budget)):
            form.addRow(label, widget)
        for label, field in (('Existing scratch parent', self.scratch_parent), ('Existing report parent', self.report_parent)):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(field)
            button = QPushButton('Choose…')
            button.clicked.connect(lambda checked=False, target=field, title=label: self._choose_directory(target, title))
            row_layout.addWidget(button)
            form.addRow(label, row)
        self.quality_policy = QLineEdit()
        self.quality_policy.setPlaceholderText('Optional version1 reference-agreement policy JSON; empty = timing only')
        quality_row = QWidget()
        quality_layout = QHBoxLayout(quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.addWidget(self.quality_policy)
        choose_quality = QPushButton('Choose policy…')
        choose_quality.clicked.connect(self._choose_quality_policy)
        quality_layout.addWidget(choose_quality)
        form.addRow('Optional quality policy', quality_row)
        layout.addLayout(form)
        profile_note = QLabel('One to eight profiles: JSON objects with id, purpose (preview/final), overrides. '
                              'Explicit overrides can select model_scale, hann_chunk/hann_stride, stage2_batch, low_vram, '
                              'quick_key settings and foundation_cache_mb. Final profiles require native scale/full-rate mode; '
                              'quality remains unverified without references. Each repetition runs cold + unmeasured warmup + warm.')
        profile_note.setWordWrap(True)
        layout.addWidget(profile_note)
        self.profiles = QPlainTextEdit('[{"id":"requested","purpose":"final","overrides":{}}]')
        self.profiles.setMaximumHeight(130)
        self.profiles.document().setMaximumBlockCount(256)
        layout.addWidget(self.profiles)
        self.workload = QLabel()
        self.workload.setWordWrap(True)
        layout.addWidget(self.workload)
        buttons = QHBoxLayout()
        self.capture_button = QPushButton('Capture metadata')
        self.verify_button = QPushButton('Verify captured plan')
        self.start_button = QPushButton('Start reviewed calibration')
        self.cancel_button = QPushButton('Cancel')
        self.report_button = QPushButton('Inspect report / expiry')
        for button in (self.capture_button, self.verify_button, self.start_button, self.cancel_button, self.report_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel('Idle · No files inspected or models loaded.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.document().setMaximumBlockCount(600)
        layout.addWidget(self.details)
        self.capture_button.clicked.connect(self.capture_metadata)
        self.verify_button.clicked.connect(self.verify)
        self.start_button.clicked.connect(self.request_start)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.report_button.clicked.connect(self.inspect_report)
        self._form_widgets = [self.sample_start, self.frame_count, self.repetitions, self.max_jobs,
                              self.max_total_frames, self.max_wall_seconds, self.output_budget, self.read_budget,
                              self.scratch_parent, self.report_parent, self.profiles, self.quality_policy]
        for widget in self._form_widgets:
            if isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._changed)
            else:
                widget.textChanged.connect(self._changed)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self._update_workload()
        self._refresh_buttons()

    @staticmethod
    def _spin(low, high, value):
        field = QSpinBox()
        field.setRange(low, high)
        field.setValue(value)
        return field

    @property
    def prepared_plan(self):
        return self.plan

    @property
    def io_busy(self):
        return (self._io_thread is not None and self._io_thread.is_alive()) or not self._io_results.empty()

    @property
    def busy(self):
        return self.waiting or self.io_busy or (self.task is not None and not self.task.done.is_set())

    def _changed(self, *args):
        self._revision += 1
        self.plan, self.capture, self.inventory = None, None, None
        if self.io_busy:
            self._cancellation.cancel('Calibration fields changed; captured operation superseded.')
        self._update_workload()
        if not self.waiting and (self.task is None or self.task.done.is_set()):
            self.status.setText('Settings changed. Capture metadata, then explicitly verify the new plan.')
        self._refresh_buttons()

    def _update_workload(self):
        try:
            raw = self.profiles.toPlainText().encode()
            if len(raw) > MAX_TEXT_BYTES:
                raise ValueError('Profile JSON exceeds 64 KiB.')
            rows = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
            if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
                raise ValueError('Choose one to eight explicit profiles.')
            total = len(rows) * self.repetitions.value() * 3
            frames = total * self.frame_count.value()
            warmups = len(rows) * self.repetitions.value()
            self.workload.setText(f'Reviewed workload: {len(rows)} profiles × {self.repetitions.value()} repetitions × '
                                  f'3 jobs = {total} actual jobs / {frames} frames; {warmups} full-sample warmups are excluded '
                                  'from warm timing. Read budget starts at zero. Output budget is admission/observation, not a hard disk limit.')
        except (ValueError, TypeError, RecursionError) as exc:
            self.workload.setText(_short(str(exc)))

    def _choose_directory(self, target, title):
        if self.busy or self._closing:
            return
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            target.setText(path)

    def _choose_quality_policy(self):
        if self.busy or self._closing:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Select reference-agreement policy', '', 'JSON (*.json)')
        if path:
            self.quality_policy.setText(path)

    def _capture_fields(self):
        from .calibration import CalibrationLimits, CalibrationProfile
        if self.job is None:
            raise ValueError('Capture a configured local source job before opening calibration.')
        raw = self.profiles.toPlainText().encode()
        if len(raw) > MAX_TEXT_BYTES:
            raise ValueError('Profile JSON exceeds 64 KiB.')
        rows = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
            raise ValueError('Choose one to eight explicit profiles.')
        profiles = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'id', 'purpose', 'overrides'} or not isinstance(row['overrides'], dict):
                raise ValueError('Each profile requires exactly id, purpose and overrides object.')
            profiles.append(CalibrationProfile.capture(row['id'], row['overrides'], row['purpose']))
        start_text = self.sample_start.text().strip()
        if not start_text.isascii() or not start_text.isdecimal():
            raise ValueError('Sample source In must be an exact nonnegative integer.')
        start = int(start_text)
        if start > 2**63 - 1:
            raise ValueError('Sample source In exceeds the shared integer bound.')
        if not self.scratch_parent.text().strip() or not self.report_parent.text().strip():
            raise ValueError('Choose separate existing scratch and report parents explicitly.')
        limits = CalibrationLimits(repetitions=self.repetitions.value(), max_jobs=self.max_jobs.value(),
                                   max_total_frames=self.max_total_frames.value(), max_wall_seconds=self.max_wall_seconds.value(),
                                   max_output_bytes=self.output_budget.value() * 1024**2,
                                   read_budget=self.read_budget.value() * 1024**2)
        return _Capture(self.job, self.captured_selection, start, self.frame_count.value(), tuple(profiles),
                        Path(self.scratch_parent.text()), Path(self.report_parent.text()), limits,
                        Path(self.quality_policy.text().strip()) if self.quality_policy.text().strip() else None)

    def _begin_io(self, kind, capture):
        if self.busy or self._closing:
            raise RuntimeError('Wait for the current calibration operation and cleanup.')
        self._cancellation = CancellationToken()
        token = self._cancellation
        self._io_serial += 1
        serial, revision = self._io_serial, self._revision

        def run():
            result: Any
            summary: dict[str, Any]
            try:
                from .calibration import inventory_calibration, plan_calibration
                if kind == 'report':
                    from .local_json import read_object
                    token.check()
                    result, _ = read_object(capture, max_bytes=32 * 1024**2, regular_file=True)
                    token.check()
                    if result.get('type') != 'corridorkey_calibration_report' or result.get('schema_version') != 1:
                        raise ValueError('Expected a versioned local calibration report.')
                    summary = dict(result.get('summary') or {})
                    summary.update({key: result[key] for key in ('status', 'created_utc', 'binding_identity', 'cleanup_error', 'artifact_error', 'error') if key in result})
                    summary['report_path'] = str(capture)
                    failures = {}
                    for trial in result.get('trials', [])[:168]:
                        token.check()
                        if isinstance(trial, dict) and trial.get('eligible') is not True:
                            key = (_short(trial.get('profile'), 64), _short(trial.get('state'), 32))
                            group = failures.setdefault(key, {'profile': key[0], 'state': key[1], 'count': 0,
                                                              'reason': _short(trial.get('error') or trial.get('exclusion_reason') or trial.get('eligibility_note'), 1024)})
                            group['count'] += 1
                    summary['exclusions'] = list(failures.values())[:16]
                    observations = {}
                    for trial in result.get('trials', [])[:168]:
                        token.check()
                        measurement = trial.get('measurement') if isinstance(trial, dict) else None
                        if not isinstance(measurement, dict):
                            continue
                        key = (_short(trial.get('profile'), 64), _short(trial.get('state'), 32))
                        row = observations.setdefault(key, {'profile': key[0], 'state': key[1], 'first': [], 'processing': [],
                                                            'allocated': [], 'reserved': []})
                        for field, destination in (('first_rendered_batch_seconds', 'first'), ('processing_seconds', 'processing')):
                            value = measurement.get(field)
                            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                                row[destination].append(value)
                        resource = measurement.get('resources')
                        if isinstance(resource, dict):
                            for field, destination in (('peak_allocated_vram_bytes', 'allocated'), ('peak_reserved_vram_bytes', 'reserved')):
                                value = resource.get(field)
                                if type(value) is int and value >= 0:
                                    row[destination].append(value)
                    summary['observations'] = []
                    for row in list(observations.values())[:16]:
                        summary['observations'].append({'profile': row['profile'], 'state': row['state'],
                            'first_rendered_batch_seconds': f"{min(row['first']):.6g} to {max(row['first']):.6g}" if row['first'] else 'NOT VERIFIED',
                            'processing_seconds': f"{min(row['processing']):.6g} to {max(row['processing']):.6g}" if row['processing'] else 'NOT VERIFIED',
                            'peak_allocated_vram_bytes': max(row['allocated']) if row['allocated'] else 'NOT VERIFIED',
                            'peak_reserved_vram_bytes': max(row['reserved']) if row['reserved'] else 'NOT VERIFIED',
                            'resource_scope': 'Runtime observations only. Missing allocator metrics remain unverified; no whole-device or long-clip fit claim.',
                            'latency_scope': 'First rendered batch excludes model load/preflight; not per-frame UI latency.'})
                    created = datetime.fromisoformat(result['created_utc'])
                    if created.tzinfo is None:
                        raise ValueError('Calibration report creation time lacks a time zone.')
                    now = datetime.now(UTC)
                    expired = now < created or now > created + timedelta(days=7)
                    summary['expires_utc'] = (created + timedelta(days=7)).isoformat()
                    summary['expiry_note'] = ('Expired or clock moved backwards. ' if expired else 'Within the seven-day age window. ') + 'Current hardware/model/sample/settings binding has NOT been reverified; this is not renewed calibration.'
                else:
                    if kind == 'capture':
                        result = inventory_calibration(capture.job, **capture.arguments(), check=token.check)
                        summary = result
                    else:
                        prepared = plan_calibration(capture.job, **capture.arguments(), check=token.check)
                        result = prepared
                        summary = dict(prepared.summary, plan_identity=prepared.identity)
                token.check()
                text = calibration_view(summary)
                error = None
            except Exception as exc:
                logging.getLogger(__name__).exception('Calibration plan/report operation failed')
                result, text, error = None, '', f'{type(exc).__name__}: {exc}'[:4096]
            self._io_results.put((serial, revision, kind, (capture, result, text), error))

        self._io_thread = threading.Thread(target=run, name='calibration-plan-io', daemon=False)
        self._io_thread.start()
        self.status.setText('Inspecting bounded metadata…' if kind == 'capture' else 'Explicitly verifying source/model identities under the read budget…')
        self._refresh_buttons()

    def capture_metadata(self):
        if self.busy or self._closing:
            self.status.setText('Wait for the current calibration operation and cleanup.')
            return
        try:
            self.plan = None
            self._begin_io('capture', self._capture_fields())
        except (OSError, ValueError, TypeError, RuntimeError, RecursionError) as exc:
            self.status.setText(_short(str(exc)))
        self._refresh_buttons()

    def verify(self):
        if self.busy or self._closing:
            self.status.setText('Wait for the current calibration operation and cleanup.')
            return
        try:
            if self.capture is None or self.inventory is None:
                raise ValueError('Capture metadata for these exact settings before verification.')
            self.plan = None
            self._begin_io('verify', self.capture)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self.status.setText(_short(str(exc)))
        self._refresh_buttons()

    def request_start(self):
        if self.plan is None or self.busy or self._closing:
            self.status.setText('Verify and review a current immutable plan before starting.')
            return
        self.waiting = True
        self.status.setText('Start requested; waiting for the existing processing owner to retire.')
        self._refresh_buttons()
        self.run_requested.emit(self.plan)

    def reject_start(self, message):
        self.waiting = False
        self.status.setText(_short(message, 4096))
        self._refresh_buttons()

    def begin(self, plan):
        if self._closing or not self.waiting or plan is not self.plan or self.io_busy:
            raise ValueError('Calibration start no longer matches the reviewed pending plan.')
        if self.task is not None and not self.task.done.is_set():
            raise RuntimeError('The previous calibration owner has not retired.')
        if self.task is not None and self.task.cleanup_error:
            raise RuntimeError('Previous calibration cleanup remains unverified.')
        self.task = self.factory(plan)
        self._shown_task_revision = -1
        self.waiting = False
        self.status.setText('Calibration owner started; waiting for actual worker evidence.')
        self._refresh_buttons()

    def _request_cancel(self):
        self.cancel()
        self.cancel_requested.emit()

    def cancel(self):
        self._io_serial += 1
        self._cancellation.cancel('Calibration cancelled from desktop.')
        self.waiting = False
        if self.task is not None and not self.task.done.is_set():
            self.task.cancel()
        if self.task is not None and self.task.cleanup_error:
            self.status.setText('Cleanup NOT VERIFIED; owner retained: ' + _short(self.task.cleanup_error))
        else:
            self.status.setText('Cancellation requested. Waiting for owned work and cleanup; no further jobs will be started.')
        self._refresh_buttons()

    def poll(self):
        try:
            serial, revision, kind, payload, error = self._io_results.get_nowait()
        except queue.Empty:
            pass
        else:
            capture, result, text = payload
            if serial == self._io_serial and revision == self._revision and not self._closing:
                if error:
                    self.status.setText(error)
                elif kind == 'capture':
                    self.capture, self.inventory, self.plan = capture, result, None
                    self.details.setPlainText(text)
                    self.status.setText('Metadata captured. Review required bytes and workload; Verify is a separate explicit read operation.')
                elif kind == 'verify':
                    self.plan = result
                    self.details.setPlainText(text)
                    self.status.setText('Verified immutable plan. Review workload, budgets and warnings before Start.')
                else:
                    self.details.setPlainText(text)
                    self.status.setText('Local report inspected. Age is shown; current binding has NOT been reverified. Recommendations remain read-only.')
        if self.task is not None:
            snapshot = self.task.snapshot()
            revision = snapshot.get('revision')
            if revision == self._shown_task_revision or self.io_busy:
                self._refresh_buttons()
                return
            self._shown_task_revision = revision
            compact = dict(snapshot)
            if isinstance(snapshot.get('summary'), dict):
                for key in ('groups', 'recommendations', 'excluded_trials', 'duration_scope', 'expiry', 'validity', 'expires_utc'):
                    if key in snapshot['summary']:
                        compact[key] = snapshot['summary'][key]
            self.details.setPlainText(calibration_view(compact))
            state = _short(snapshot.get('status'))
            if snapshot.get('cleanup_error') or self.task.cleanup_error:
                self.status.setText('Cleanup NOT VERIFIED; owner retained: ' + _short(snapshot.get('cleanup_error') or self.task.cleanup_error))
            elif not self._closing:
                self.status.setText('Calibration ' + state + '. Recommendations are read-only; inspect the local report for complete evidence.')
        self._refresh_buttons()

    def inspect_report(self):
        try:
            if self.task is None or self.task.report_path is None:
                raise ValueError('No local calibration report is available yet.')
            self._begin_io('report', self.task.report_path)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self.status.setText(_short(str(exc)))

    def show_report_location(self):
        if self.task is None or self.task.report_path is None:
            self.status.setText('No local calibration report is available yet.')
        else:
            self.status.setText('Local report: ' + _short(str(self.task.report_path), 4096))

    def _refresh_buttons(self):
        busy = self.busy or self._closing
        cleanup_blocked = self.task is not None and bool(self.task.cleanup_error)
        for widget in self._form_widgets:
            widget.setEnabled(not busy and not cleanup_blocked)
        self.capture_button.setEnabled(not busy and not cleanup_blocked and self.job is not None)
        self.verify_button.setEnabled(not busy and not cleanup_blocked and self.capture is not None)
        self.start_button.setEnabled(not busy and not cleanup_blocked and self.plan is not None)
        self.cancel_button.setEnabled(self.busy)
        self.report_button.setEnabled(not busy and self.task is not None and self.task.report_path is not None)

    def shutdown(self):
        self._closing = True
        self.cancel()
        self.poll()
        complete = not self.busy and (self.task is None or not self.task.cleanup_error)
        if complete:
            self.hide()
        return complete

    def closeEvent(self, event):
        self._closing = True
        self._request_cancel()
        self.hide()
        event.ignore()  # Parent retains this dialog until owned I/O/worker cleanup retires.
