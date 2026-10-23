"""Explicit readiness checks and immutable short-render preparation for Qt."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
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
from .desktop_comparison import RenderView, completed_view
from .desktop_preview import preview
from .output_paths import output_layout
from .readiness import check_readiness, export_report, prepare_short_render
from .readiness_capabilities import check_capabilities
from .session import JobRequest
from .worker_protocol import MAX_FRAME_BYTES, job_to_wire

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
        self.latest_job: JobRequest | None = None
        self.latest_request_id: str | None = None
        self.completed_record: dict | None = None
        self.completed_output: RenderView | None = None
        self.render_revision = 0
        self.inspection_revision = 0
        self.operation = ''
        self.thumbnail_pixmap: QPixmap | None = None
        self.capability_observation: dict | None = None
        self.capability_protected: list[str] = []
        layout = QVBoxLayout(self)
        note = QLabel('Lightweight checks run only when requested. They do not load models, run inference, install or download assets. Prepare snapshots the current job; Run submits that snapshot through the existing render worker. Model loading and actual inference are only tested by that explicit render. Inspect completed frame explicitly decodes a selected completed output and presents a bounded thumbnail. Worker-reported CUDA work is separate from independent pretrained-model acceptance.')
        note.setWordWrap(True)
        layout.addWidget(note)
        actions = QHBoxLayout()
        self.check_button = QPushButton('Check readiness')
        self.check_button.clicked.connect(self.inspect)
        actions.addWidget(self.check_button)
        self.capability_button = QPushButton('Probe backend / encoders')
        self.capability_button.clicked.connect(self.probe_capabilities)
        actions.addWidget(self.capability_button)
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
        inspection = QHBoxLayout()
        self.inspection_mode = QComboBox()
        self.inspection_frame = QSpinBox()
        self.inspection_frame.setRange(0, 0)
        inspection.addWidget(QLabel('Completed output / source frame'))
        inspection.addWidget(self.inspection_mode)
        inspection.addWidget(self.inspection_frame)
        self.inspect_button = QPushButton('Inspect completed frame')
        self.inspect_button.clicked.connect(self.inspect_completed)
        inspection.addWidget(self.inspect_button)
        layout.addLayout(inspection)
        self.thumbnail = QLabel('No completed output inspected. Run a guided job, then inspect explicitly.')
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setMinimumHeight(100)
        self.thumbnail.setMaximumHeight(320)
        self.thumbnail.setWordWrap(True)
        layout.addWidget(self.thumbnail)
        self.start.valueChanged.connect(self.invalidate_prepared)
        self.count.valueChanged.connect(self.invalidate_prepared)
        self._buttons()

    @property
    def io_busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    def _buttons(self):
        idle = not self.io_busy and not self.close_when_done and not self.workspace.closing
        self.check_button.setEnabled(idle)
        self.capability_button.setEnabled(idle)
        self.prepare_button.setEnabled(idle and self.submitted_id is None)
        self.run_button.setEnabled(idle and self.prepared is not None and self.submitted_id is None)
        self.export_button.setEnabled(idle and self.report is not None)
        inspect_ready = self.submitted_id is None and self.completed_output is not None and self.inspection_mode.count() > 0
        self.inspect_button.setEnabled(idle and inspect_ready)
        self.inspection_mode.setEnabled(idle and inspect_ready)
        self.inspection_frame.setEnabled(idle and inspect_ready)

    def invalidate_prepared(self):
        self.prepare_revision += 1
        self.prepared = None
        self._buttons()

    def _begin(self, kind, function, *, context=None):
        if self.io_busy or self.close_when_done or self.workspace.closing:
            self.status.setText('Wait for the current operation or shutdown.')
            return
        self.cancellation = CancellationToken()
        self.operation = kind

        def work():
            try:
                result = function()
                error = None
            except Exception as exc:
                logging.getLogger(__name__).exception('Readiness operation failed')
                result, error = None, f'{type(exc).__name__}: {exc}'[:4096]
            self.results.put((kind, result, error, context))

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

    def probe_capabilities(self):
        if self.io_busy or self.close_when_done or self.workspace.closing:
            self.status.setText('Wait for the current operation or shutdown before probing capabilities.')
            return
        try:
            job = self.prepared or self.latest_job or self.workspace.job()
            job.configurations()
        except (ValueError, OSError) as exc:
            self.status.setText('Capability probe requires a valid captured job: ' + str(exc))
            return
        selection = self.workspace.asset_selection
        self.invalidate_prepared()  # probe details replace the displayed render snapshot
        self.details.setPlainText('Explicit owned subprocess metadata probes for this captured job. No models load, inference, test encode, install or download. Independent children launched by executable wrappers are unsupported; native Windows cleanup NOT VERIFIED.\n\n' + json.dumps(job_to_wire(job), indent=2))
        self._begin('capabilities', lambda: (job, selection, check_capabilities(job, cancellation=self.cancellation)))

    def _record_capabilities(self, result):
        job, selection, report = result
        self.capability_observation = {'requested_job': job_to_wire(job), 'result': report}
        config, _ = job.configurations()
        protected = [path for path in (job.input_path, job.initial_hint_path, job.hint_video_path) if path]
        if config.corrections_path:
            protected.append(str(Path(config.corrections_path).parent))
        protected.append(str(selection.root) if selection is not None else str(Path(__file__).resolve().parents[1]))
        if selection is not None and selection.weights is not None:
            protected.append(str(selection.weights))
        self.capability_protected = list(dict.fromkeys(protected))
        if self.report is None:
            self.report = {'schema_version': 1, 'type': 'corridorkey_readiness', 'stages': [], 'protected_paths': []}
        self.report['capabilities'] = self.capability_observation
        self.details.setPlainText(json.dumps(self.capability_observation, indent=2))
        self.status.setText('Explicit backend/encoder metadata probes ' + str(report.get('status', 'NOT VERIFIED')) + '. Model loading, actual inference, encoder execution and native Windows cleanup remain separately unverified.')

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
        self.render_revision += 1
        self.inspection_revision += 1
        self.latest_job, self.latest_request_id = self.prepared, request_id
        self.completed_record, self.completed_output = None, None
        self.inspection_mode.clear()
        self.thumbnail_pixmap = None
        self.thumbnail.setText('Waiting for this guided job to complete. No output decoded.')
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
        self._cancel_inspection()
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
            if kind == 'result':
                try:
                    self._capture_completed(payload.get('record'))
                except (ValueError, TypeError, OSError) as exc:
                    if self.render_observation is not None:
                        self.render_observation['record_validation'] = {'status': 'FAIL', 'error': str(exc)[:4096]}
                    self.status.setText('Completed-output inspection unavailable: ' + str(exc))
            else:
                self.status.setText(f'Guided job {kind} · {payload.get("error", "")}. Inspection requires this job to complete successfully.')
            self._buttons()

    def _capture_completed(self, record):
        if self.latest_job is None or not isinstance(record, dict):
            raise ValueError('The matching terminal event has no completed job record.')
        raw = json.dumps(record, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(raw) > MAX_FRAME_BYTES:
            raise ValueError('Completed record exceeds the 4 MiB worker protocol limit.')
        captured = json.loads(raw)
        job = self.latest_job
        config, _ = job.configurations()
        view = completed_view(job, captured)
        if captured.get('output_dir') != job.output_dir:
            raise ValueError('Completed output destination differs from the submitted snapshot.')
        if captured.get('requested_settings') != json.loads(job.config_json) or captured.get('writer_settings') != json.loads(job.writer_json):
            raise ValueError('Completed requested/writer settings differ from the submitted snapshot.')
        if view.start != (config.start_frame or 0) or not 1 <= view.count <= config.num_frames:
            raise ValueError('Completed source range differs from the requested guided range.')
        expected = {role: str(Path(job.output_dir) / item['path']) for role, item in output_layout(job.input_path, config)['outputs'].items()}
        if dict(view.outputs) != expected:
            raise ValueError('Completed output roles/paths differ from the submitted job layout.')
        if captured.get('hints') != {'first_frame': job.initial_hint_path, 'full': job.hint_video_path}:
            raise ValueError('Completed hint bindings differ from the submitted snapshot.')
        self.completed_record, self.completed_output = captured, view
        self.inspection_frame.setRange(view.start, view.end - 1)
        self.inspection_frame.setValue(view.start)
        outputs = dict(view.outputs)
        self.inspection_mode.clear()
        if 'alpha' in outputs:
            self.inspection_mode.addItem('Alpha')
        if 'fg' in outputs:
            self.inspection_mode.addItem('Foreground')
        if 'fg' in outputs and 'alpha' in outputs:
            self.inspection_mode.addItem('Checker composite')
        assert self.render_observation is not None
        diagnostics = captured['diagnostics']
        execution = diagnostics.get('execution')
        execution = execution if isinstance(execution, dict) else {}
        positive = lambda value: type(value) is int and value > 0
        observed_cuda = (execution.get('backend') == 'pytorch' and execution.get('requested_device') == 'cuda'
                         and isinstance(execution.get('gpu'), str) and bool(execution['gpu'].strip())
                         and type(execution.get('device_index')) is int and execution['device_index'] >= 0
                         and isinstance(execution.get('cuda_version'), str) and bool(execution['cuda_version'].strip())
                         and positive(execution.get('stage2_calls'))
                         and positive(execution.get('stage2_frame_evaluations'))
                         and positive(execution.get('completed_prediction_frames')))
        self.render_observation.update({
            'record_validation': 'PASS: matching completed metadata; source/output bytes are not authenticated',
            'engine_reported': {'status': captured['status'], 'requested_settings': captured['requested_settings'],
                                'effective_settings': diagnostics.get('effective_settings'),
                                'writer_settings': captured['writer_settings'], 'source': diagnostics['source'],
                                'execution': execution, 'frames_submitted': diagnostics['frames_submitted'],
                                'model_reused': captured.get('model_reused'),
                                'weights': captured.get('weights'), 'asset_paths': captured.get('asset_paths'),
                                'provenance_note': captured.get('provenance_note'),
                                'model_load_seconds': captured.get('model_load_seconds'),
                                'model_initialization_seconds': captured.get('model_initialization_seconds'),
                                'model_load_status': ('engine-reported reused model' if captured.get('model_reused') is True else
                                                      'engine-reported load completed' if captured.get('model_reused') is False else 'NOT VERIFIED'),
                                'outputs': diagnostics['outputs'],
                                'range': {'requested_start': config.start_frame or 0, 'requested_count': config.num_frames,
                                          'observed_start': view.start, 'observed_count': view.count,
                                          'note': 'Engine reports fewer frames than requested; reason/source-end truncation and exact requested count NOT VERIFIED' if view.count < config.num_frames else 'Requested range matches recorded selection'}},
            'actual_inference': 'ENGINE REPORTED CUDA INFERENCE' if observed_cuda else 'NOT VERIFIED',
            'pretrained_model_authenticity_and_quality': 'NOT VERIFIED: worker metadata is not independent model/hardware or visual-quality acceptance',
        })
        self.details.setPlainText(json.dumps(self.render_observation, indent=2))
        self.status.setText('Matching completed record captured. Select a source frame and inspect explicitly; no output has been decoded.' if self.inspection_mode.count() else
                            'Job completed; inspection currently requires Alpha or Foreground exports. Prepare a job requesting those outputs.')

    def inspect_completed(self):
        if self.io_busy or self.close_when_done or self.workspace.closing:
            self.status.setText('Wait for the current operation or shutdown before inspecting outputs.')
            return
        if self.completed_output is None or self.completed_record is None or self.submitted_id is not None or not self.inspection_mode.count():
            self.status.setText('Inspection requires this latest guided job to complete with matching Alpha or Foreground exports.')
            return
        mode, index = self.inspection_mode.currentText(), self.inspection_frame.value()
        try:
            options = self.completed_output.options(mode, index)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.invalidate_prepared()  # Inspection will replace the reviewed render details.
        self.inspection_revision += 1
        context = (self.render_revision, self.inspection_revision, self.latest_request_id)
        self.thumbnail_pixmap = None
        self.thumbnail.setText('Decoding the selected completed output…')
        assert self.render_observation is not None
        self.render_observation['output_decode'] = 'PENDING'
        self.render_observation['display'] = 'NOT VERIFIED'
        self._begin('inspection', lambda: (mode, index, options, preview(**options, max_side=512)), context=context)

    def _cancel_inspection(self):
        self.inspection_revision += 1
        if self.io_busy and self.operation == 'inspection' and self.render_observation is not None:
            self.render_observation['output_decode'] = 'CANCELLED: late result will not be activated'
            self.render_observation['display'] = 'NOT VERIFIED'
            self.thumbnail_pixmap = None
            self.thumbnail.setText('Inspection cancelled; late decoded pixels will not be displayed.')

    def _display_completed(self, result):
        mode, index, options, frame = result
        pixels = frame.pixels
        assert self.render_observation is not None
        evidence = {'status': 'PASS', 'scope': 'One explicitly selected completed output frame; no whole-clip or model-quality acceptance',
                    'request_id': self.latest_request_id, 'mode': mode, 'source_frame': index,
                    'export_frame': options['index'], 'path': options['path'],
                    'native_hw': [frame.height, frame.width], 'thumbnail_hw': list(pixels.shape[:2]),
                    'display_note': frame.note}
        self.render_observation['output_decode'] = evidence
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        if image.isNull() or pixmap.isNull():
            self.status.setText('Output decoded, but Qt could not construct its thumbnail.')
            self.render_observation['display'] = 'FAIL: Qt thumbnail construction'
            self.details.setPlainText(json.dumps(self.render_observation, indent=2))
            return
        self.thumbnail_pixmap = pixmap
        self.thumbnail.setPixmap(pixmap.scaled(self.thumbnail.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
        self.render_observation['display'] = {'status': 'PASS', 'scope': 'Qt thumbnail assigned on the GUI thread; human/native display acceptance NOT VERIFIED'}
        self.details.setPlainText(json.dumps(self.render_observation, indent=2))
        self.status.setText(f'Inspected {mode}, source frame {index}, export frame {options["index"]}. Decode and Qt presentation recorded separately from inference.')

    def choose_export(self):
        if self.report is None or self.io_busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save fresh readiness report', '', 'JSON (*.json)')
        if path:
            report = json.loads(json.dumps(self.report))
            if self.capability_observation is not None:
                report['capabilities'] = json.loads(json.dumps(self.capability_observation))
                report['protected_paths'] = list(dict.fromkeys([*report.get('protected_paths', []), *self.capability_protected]))
            if self.render_observation is not None:
                report['render_observation'] = json.loads(json.dumps(self.render_observation))
                report['protected_paths'] = list(dict.fromkeys(
                    [*report.get('protected_paths', []), *self.render_protected]))
            self._begin('export', lambda: export_report(report, Path(path)))

    def poll(self):
        try:
            kind, result, error, context = self.results.get_nowait()
        except queue.Empty:
            self._buttons()
            if self.close_when_done and not self.io_busy:
                self.close_when_done = False
                self.close()
            return
        if kind == 'inspection' and context != (self.render_revision, self.inspection_revision, self.latest_request_id):
            self._buttons()
            return  # Cancelled/closed/superseded decode owns no current evidence.
        if error:
            if kind == 'inspection' and self.render_observation is not None:
                self.render_observation['output_decode'] = {'status': 'FAIL', 'error': error}
                self.render_observation['display'] = 'NOT VERIFIED'
                self.thumbnail.setText('Output decode failed: ' + error)
            self.status.setText(error)
        elif self.cancellation.cancelled and kind != 'export':
            if kind == 'checks' and isinstance(result, dict) and result.get('status') == 'cancelled':
                self.report = result
                self.details.setPlainText(json.dumps(result, indent=2))
            if kind == 'capabilities' and isinstance(result, tuple) and result[2].get('status') == 'cancelled':
                self._record_capabilities(result)
            self.status.setText('Readiness operation cancelled; any partial cancelled check/probe report can be exported.')
        elif kind == 'capabilities':
            self._record_capabilities(result)
        elif kind == 'inspection':
            self._display_completed(result)
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

    def reject(self):
        # Escape is also a close request; preserve task ownership and fences.
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self._cancel_inspection()
        self.cancellation.cancel('Readiness dialog closed')
        if self.io_busy:
            self.close_when_done = True
            event.ignore()
        else:
            event.accept()
