"""Explicit transient previews through the desktop's existing engine owner."""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from .control import CancellationToken, CancelledError
from .desktop_comparison import completed_view
from .desktop_preview import preview
from .preview_execution import PreviewPreparationError, prepare_preview
from .progressive_preview import PreviewArtifact, ProgressivePreview
from .worker_protocol import job_to_wire


class ProgressivePreviewPanel(QDialog):
    """One inference request and one bounded IO operation, retained to retirement.

    The parent supplies job/asset snapshots and the existing DesktopBridge. These
    previews never enter the parent's export queue or alter form settings.
    """

    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.setWindowTitle('Progressive preview')
        self.resize(750, 760)
        self.recipe = None
        self.scheduler = None
        self.request = None
        self.prepared = None
        self.retained_workspace = None
        self.worker_id: str | None = None
        self.cancel_sent = False
        self.cancel_error: str | None = None
        self.failure: str | None = None
        self.io_thread: threading.Thread | None = None
        self.io_token: CancellationToken | None = None
        self.io_events: queue.Queue = queue.Queue(maxsize=1)
        self.context_key = None
        self.desired_frame = 0
        self.desired_playing = False
        self.validated_frame: int | None = None
        self.last_focus = time.monotonic()
        self.closing = False
        self.enabled = QCheckBox('Enable progressive inference previews (off by default)')
        self.enabled.toggled.connect(self.toggle)
        self.scratch = QLineEdit()
        choose = QPushButton('Choose preview scratch…')
        choose.clicked.connect(self.choose_scratch)
        self.draft = QCheckBox('First show an explicit reduced-scale draft')
        self.scale = QDoubleSpinBox()
        self.scale.setRange(.01, .99)
        self.scale.setValue(.5)
        self.scale.setDecimals(2)
        self.idle = QCheckBox('Refine surrounding frames after 1 second idle')
        self.idle.setChecked(True)
        self.view = QComboBox()
        self.view.addItems(['Alpha', 'Foreground', 'Checker composite'])
        for control in (self.scratch,):
            control.textChanged.connect(self.settings_changed)
        self.draft.toggled.connect(self.settings_changed)
        self.scale.valueChanged.connect(self.settings_changed)
        self.view.currentIndexChanged.connect(self.settings_changed)
        self.start = QPushButton('Capture settings and start')
        self.start.clicked.connect(self.begin)
        retry = QPushButton('Retry retained cleanup')
        retry.clicked.connect(self.retry_cleanup)
        stop = QPushButton('Cancel / disable')
        stop.clicked.connect(lambda: self.enabled.setChecked(False))
        self.status = QLabel('Disabled. Opening this panel performs no model or filesystem work.')
        self.status.setWordWrap(True)
        self.image = QLabel('No progressive preview')
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(260)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        form = QFormLayout()
        form.addRow(choose, self.scratch)
        form.addRow(self.draft, self.scale)
        form.addRow('Display', self.view)
        row = QHBoxLayout()
        row.addWidget(self.start)
        row.addWidget(stop)
        row.addWidget(retry)
        layout = QVBoxLayout(self)
        layout.addWidget(self.enabled)
        layout.addLayout(form)
        layout.addWidget(self.idle)
        layout.addLayout(row)
        layout.addWidget(QLabel('Final preview stage preserves the captured processing settings; single-frame temporal context restarts. This is not full-range export equivalence.'))
        layout.addWidget(self.status)
        layout.addWidget(self.image)
        layout.addWidget(self.details)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    @property
    def io_busy(self):
        return self.io_thread is not None or not self.io_events.empty()

    @property
    def busy(self):
        return self.io_busy or self.worker_id is not None or self.prepared is not None or self.retained_workspace is not None

    def choose_scratch(self):
        selected = QFileDialog.getExistingDirectory(self, 'Choose a separate existing preview scratch parent')
        if selected:
            self.scratch.setText(selected)

    def settings_changed(self, *_):
        if self.recipe is not None or self.busy:
            self.enabled.setChecked(False)
            self.invalidate('Preview settings changed; capture a new snapshot explicitly.')

    def toggle(self, checked):
        if not checked:
            self.invalidate('Disabled; waiting for owned preview cleanup.' if self.busy else 'Disabled.')
        else:
            self.status.setText('Enabled. Review settings and choose Capture settings and start.')

    def _key(self):
        job = self.workspace.job(preview_only=True)
        selection = self.workspace.asset_selection
        return (job, selection, self.view.currentText(), self.draft.isChecked(), self.scale.value(), self.scratch.text(), id(self.workspace.bridge))

    def _io(self, kind, operation):
        if self.io_busy:
            raise RuntimeError('Wait for the owned preview IO operation.')
        token = CancellationToken()
        self.io_token = token

        def run():
            try:
                result, error = operation(token.check), None
            except CancelledError as exc:
                result, error = None, exc
            except Exception as exc:
                logging.getLogger(__name__).exception('Progressive preview IO failed')
                result, error = None, exc
            self.io_events.put((kind, result, error, token))

        self.io_thread = threading.Thread(target=run, name='progressive-preview-io', daemon=False)
        self.io_thread.start()

    def begin(self):
        if self.closing or not self.enabled.isChecked() or self.busy:
            self.status.setText('Enable previews and wait for current work to retire before capturing.')
            return
        try:
            if not self._engine_available():
                raise ValueError('Wait for normal renders, recovery, benchmark or worker shutdown before previewing.')
            key = self._key()
            job, selection = key[:2]
            config, _ = job.configurations()
            required_outputs = {'alpha'} if self.view.currentText() == 'Alpha' else {'fg'} if self.view.currentText() == 'Foreground' else {'alpha', 'fg'}
            if not required_outputs.issubset(config.outputs):
                raise ValueError('The selected display needs captured outputs ' + ', '.join(sorted(required_outputs)) + '; change the display or explicitly configure those outputs first.')
            references = {'asset_root': str(selection.root), 'weights': None if selection.weights is None else str(selection.weights)} if selection is not None else {'selection': 'worker defaults; compatibility NOT VERIFIED'}
            parent, scale = Path(self.scratch.text()), self.scale.value() if self.draft.isChecked() else None
            if not self.scratch.text():
                raise ValueError('Choose a separate existing scratch parent explicitly.')
            self.invalidate('Pending new captured preview; previous pixels are stale.')
            self.context_key = key
            self.failure = None
            self.validated_frame = None
            self.desired_frame, self.desired_playing = self.workspace.playback.progressive_focus(job)
            self._io('capture', lambda check: prepare_preview(job, scratch_parent=parent, declared_fps=config.frame_dir_fps, draft_scale=scale, model_references=references, check=check))
            self.status.setText('Capturing bounded source/settings identity; no model loaded yet.')
        except (ValueError, OSError, TypeError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def _engine_available(self):
        parent = self.workspace
        return (not parent.closing and not parent.bridge.done.is_set() and not parent.benchmark.busy
                and parent.pending_queue_root is None and parent.pending_benchmark is None
                and all(state['status'] in {'result', 'failed', 'cancelled', 'rejected', 'lost'} for state in parent.jobs.values()))

    def _hide_pixels(self, message):
        self.image.clear()
        self.image.setText(message)
        self.image.setToolTip('')

    def invalidate(self, message):
        self._hide_pixels('Stale preview hidden. ' + message)
        if self.scheduler is not None:
            self.scheduler.set_enabled(False)
        if self.io_token is not None:
            self.io_token.cancel(message)
        self.recipe = None
        self.validated_frame = None
        self.status.setText(message)
        self._cancel_worker()

    def suspend_for_export(self):
        self.enabled.setChecked(False)
        self.invalidate('Normal rendering takes priority; capture a new preview snapshot afterward.')

    def _cancel_worker(self):
        if self.worker_id is not None and not self.cancel_sent:
            if self.cancel_error is not None:
                self.status.setText(self.cancel_error)
                return
            try:
                self.workspace.bridge.send('cancel', {'target': self.worker_id, 'reason': 'Progressive preview superseded or disabled'})
                self.cancel_sent = True
                self.cancel_error = None
                self.status.setText('Cancellation requested; waiting for owned preview cleanup.')
            except (RuntimeError, ValueError, queue.Full) as exc:
                self.cancel_error = f'Preview cancellation delivery failed: {exc or type(exc).__name__}. Ownership retained; choose Retry retained cleanup to retry cancellation.'
                self.status.setText(self.cancel_error)

    def observe(self, event):
        if self.worker_id is None or event.get('request_id') != self.worker_id:
            return False
        kind, payload = event['kind'], event.get('payload', {})
        if kind == 'progress':
            self.status.setText(f'Pending {self.request.stage.name if self.request else "preview"}: {payload.get("stage", "processing")}')
        if kind in {'result', 'failed', 'cancelled', 'rejected', 'lost'}:
            self.worker_id = None
            self.cancel_sent = False
            self.cancel_error = None
            prepared, request, scheduler = self.prepared, self.request, self.scheduler
            if prepared is None or request is None or scheduler is None:
                self.failure = 'Preview ownership metadata missing; no further preview scheduled.'
                return True
            mode = self.view.currentText()
            source = self.recipe.source if self.recipe is not None else None

            def finish(check):
                artifact = None
                primary = None
                try:
                    scheduler.check(request)
                    if kind != 'result':
                        raise RuntimeError(str(payload.get('error', kind)))
                    check()
                    if source is None:
                        raise CancelledError('Preview snapshot was invalidated.')
                    prepared.check_source(check=check)
                    prepared.workspace.check_budget(check=check)
                    view = completed_view(prepared.job, payload['record'])
                    frame = preview(**view.options(mode, prepared.frame), max_side=512)
                    check()
                    scheduler.check(request)
                    execution = payload['record'].get('diagnostics', {}).get('execution') or {}
                    observed = {key: value for key, value in execution.items() if key in {'neural_inference', 'fixture_loads', 'backend', 'requested_device', 'autocast_dtype', 'gpu', 'cuda_version', 'stage2_calls', 'completed_prediction_frames'} and type(value) in (str, int, float, bool) and len(str(value)) <= 1024}
                    metadata = {'execution_observations': observed, 'width': frame.pixels.shape[1], 'height': frame.pixels.shape[0],
                                'source_frame': prepared.frame, 'stage': prepared.stage,
                                'prepared_identity': prepared.identity, 'note': frame.note,
                                'requested_settings_sha256': hashlib.sha256(prepared.requested_settings_json).hexdigest(),
                                'effective_settings_sha256': hashlib.sha256(prepared.effective_settings_json).hexdigest(),
                                'provenance_sha256': hashlib.sha256(prepared.provenance_json).hexdigest(),
                                'temporal_context': 'single frame; restarted', 'inference_evidence': 'Completed engine record; selected execution observations retained. Captured model references are metadata, not authentication.'}
                    artifact = PreviewArtifact(request.identity, frame.pixels.tobytes(), json.dumps(metadata, allow_nan=False).encode())
                except CancelledError as exc:
                    primary = exc
                except Exception as exc:
                    logging.getLogger(__name__).exception('Progressive preview completion failed')
                    primary = exc
                try:
                    prepared.workspace.cleanup()
                except Exception as cleanup_error:
                    raise RuntimeError(f'Preview cleanup NOT VERIFIED; retained {prepared.workspace.root}: {cleanup_error}; original result: {primary}') from cleanup_error
                if primary is not None:
                    raise primary
                return artifact

            self._io('finish', finish)
        return True

    def _display(self):
        if self.scheduler is None:
            return
        snapshot = self.scheduler.snapshot(self.desired_frame)
        result = snapshot.result
        state = snapshot.status
        self.status.setText(f'Source frame {self.desired_frame} · {state}{' / pending ' + snapshot.pending.stage.name if snapshot.pending else ''} · bounded512px display; final means captured preview recipe')
        if result is not None:
            metadata = json.loads(result.artifact.metadata_json)
            width, height = metadata['width'], metadata['height']
            pixels = result.artifact.payload
            if len(pixels) != width * height * 3 or width > 512 or height > 512:
                raise ValueError('Invalid bounded RGB preview artifact.')
            image = QImage(pixels, width, height, width * 3, QImage.Format.Format_RGB888).copy()
            self.image.setPixmap(QPixmap.fromImage(image))
            self.image.setToolTip(f'Displayed source frame {result.request.source_frame}; {result.request.stage.name}; {metadata["note"]}; execution observations {json.dumps(metadata["execution_observations"], sort_keys=True)}')

    def _capture_matches(self, token):
        if token.cancelled or not self.enabled.isChecked() or self.closing:
            return False
        try:
            return self._key() == self.context_key
        except (ValueError, OSError, TypeError) as exc:
            self.failure = 'Captured preview was rejected because current settings are invalid: ' + str(exc)
            return False

    def poll(self):
        if self.io_thread is not None and not self.io_thread.is_alive():
            self.io_thread.join()
            self.io_thread = None
            self.io_token = None
        if self.io_thread is None:
            try:
                kind, result, error, token = self.io_events.get_nowait()
            except queue.Empty:
                pass
            else:
                if kind == 'capture':
                    if error is None and self._capture_matches(token):
                        try:
                            frame, playing = self.workspace.playback.progressive_focus(self.workspace.job(preview_only=True))
                            if not result.context.source_start <= frame < result.context.source_end:
                                raise ValueError(f'Progressive source frame {frame} is outside captured range [{result.context.source_start}, {result.context.source_end}); select a frame within that range and capture again.')
                            scheduler = ProgressivePreview(result.context, result.stages, enabled=True)
                            scheduler.focus(frame, playing=playing)
                            self.details.setPlainText(json.dumps({'summary': result.summary(), 'requested_job': json.loads(result.context.requested_settings_json), 'provenance': json.loads(result.context.provenance_json), 'preview_stages': [{'name': stage.name, 'effective_settings': json.loads(stage.effective_settings_json), 'estimated_work_bytes': stage.estimated_work_bytes} for stage in result.stages]}, indent=2))
                            self.recipe, self.scheduler = result, scheduler
                            self.desired_frame, self.desired_playing = frame, playing
                            self.validated_frame = frame
                        except (ValueError, OSError, TypeError, RuntimeError) as exc:
                            self.failure = str(exc)
                    elif error is not None:
                        self.failure = str(error)
                elif kind == 'prepare':
                    if error is None:
                        self.prepared = result
                        if self.enabled.isChecked() and self.recipe is not None and self._engine_available():
                            try:
                                scheduler, request = self.scheduler, self.request
                                if scheduler is None or request is None:
                                    raise RuntimeError('Prepared preview has no owned scheduler request.')
                                scheduler.check(request)
                                self.worker_id = self.workspace.bridge.send('preview', {'job': job_to_wire(result.job)})
                            except CancelledError:
                                pass  # Superseded preparation still owns its workspace until cleanup.
                            except (ValueError, RuntimeError, queue.Full) as exc:
                                self.failure = str(exc)
                        if self.worker_id is None:
                            self._io('cleanup', lambda _check: result.workspace.cleanup())
                    else:
                        self._retire(error)
                elif kind in {'finish', 'cleanup'}:
                    if kind == 'finish' and error is None and result is not None:
                        scheduler, request = self.scheduler, self.request
                        if scheduler is None or request is None:
                            self.failure = 'Completed preview has no owned scheduler request; result refused.'
                        else:
                            scheduler.complete(request, result)
                    else:
                        if error is not None and 'cleanup NOT VERIFIED' in str(error):
                            self.failure = str(error)
                        if self.scheduler is not None and self.request is not None:
                            self.scheduler.fail(self.request, str(error or 'Preview cancelled before dispatch'))
                    cleanup_failed = error is not None and (kind == 'cleanup' or 'cleanup NOT VERIFIED' in str(error))
                    if cleanup_failed:
                        self.failure = f'Preview cleanup NOT VERIFIED; retained {self.prepared.workspace.root if self.prepared else "unknown"}: {error}'
                    else:
                        self.prepared = None
                        self.retained_workspace = None
                        self.request = None
                        if error is not None and not isinstance(error, CancelledError):
                            self.failure = str(error)
                elif kind == 'validate':
                    if error is None:
                        self.validated_frame = result
                    elif not isinstance(error, CancelledError):
                        self.failure = str(error)
        if self.worker_id is not None and self.workspace.bridge.done.is_set():
            self.observe({'request_id': self.worker_id, 'kind': 'lost', 'payload': {'error': self.workspace.bridge.failure or 'Worker stopped'}})
        if self.cancel_error:
            self.status.setText(self.cancel_error)
            return
        if self.failure:
            self.enabled.setChecked(False)
            self.status.setText(self.failure)
            return
        if self.closing or not self.enabled.isChecked() or self.recipe is None or self.scheduler is None:
            return
        try:
            if self._key() != self.context_key:
                self.enabled.setChecked(False)
                self.invalidate('Source/settings/model selection changed; capture a new snapshot.')
                return
            frame, playing = self.workspace.playback.progressive_focus(self.workspace.job(preview_only=True))
            if frame != self.desired_frame or playing != self.desired_playing:
                self.desired_frame, self.desired_playing = frame, playing
                self.validated_frame = None
                self.last_focus = time.monotonic()
                self.scheduler.focus(frame, playing=playing)
                self._hide_pixels(f'Pending source frame {frame}; previous pixels are stale.')
                self._cancel_worker()
                if self.io_token is not None:
                    self.io_token.cancel('Playhead superseded')
            if self.busy or not self._engine_available():
                return
            if self.validated_frame != self.desired_frame:
                source, frame = self.recipe.source, self.desired_frame
                def validate(check):
                    source.check(frame, check=check)
                    return frame
                self._io('validate', validate)
                return
            self._display()
            request = self.scheduler.next_request(idle=self.idle.isChecked() and time.monotonic() - self.last_focus >= 1)
            if request is not None:
                self.request = request
                recipe = self.recipe
                self._io('prepare', lambda check: recipe.job_for(request.source_frame, request.stage, check=check))
        except (ValueError, OSError, TypeError, RuntimeError) as exc:
            self.failure = str(exc)
            self.enabled.setChecked(False)
            self.status.setText(str(exc))

    def _retire(self, error):
        if isinstance(error, PreviewPreparationError):
            self.retained_workspace = error.workspace
        if self.scheduler is not None and self.request is not None:
            self.scheduler.fail(self.request, str(error))
        self.request = None
        self.prepared = None
        if not isinstance(error, CancelledError):
            self.failure = str(error)

    def retry_cleanup(self):
        if self.worker_id is not None and not self.cancel_sent:
            self.cancel_error = None
            self._cancel_worker()
            return
        workspace = self.retained_workspace or (self.prepared.workspace if self.prepared is not None else None)
        if workspace is not None and self.worker_id is None and not self.io_busy:
            self.failure = None
            self._io('cleanup', lambda _check: workspace.cleanup())

    def shutdown(self):
        self.closing = True
        self.enabled.setChecked(False)
        self.invalidate('Closing progressive previews; waiting for owned work.')
        if not self.busy:
            self.timer.stop()
            self.hide()
            return True
        return False

    def reject(self):
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.enabled.setChecked(False)
        self.invalidate('Progressive preview window closed; owned work is retiring.')
        event.accept()
