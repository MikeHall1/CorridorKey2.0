"""Explicit OFX capture intake; one owned file operation and the existing worker."""
from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
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
    QSpinBox,
    QVBoxLayout,
)

from .asset_inventory import AssetSelection
from .control import CancellationToken, CancelledError, JobControl
from .worker_protocol import decode_frame, encode_frame


def review_text(value: Any) -> str:
    raw = json.dumps(value, indent=2, allow_nan=False)
    if len(raw.encode('utf-8')) > 65536:
        raise ValueError('Capture review summary exceeds the 64 KiB display limit.')
    return raw


class CaptureOperation:
    """Single bounded terminal and coalesced progress; cancellation retains owner."""
    def __init__(self, action: str, operation: Callable[[JobControl], Any]):
        self.action = action
        self.cancellation = CancellationToken()
        self.results: queue.Queue[tuple[Any, str | None]] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest = None
        self.thread = threading.Thread(target=self._run, args=(operation,), name='ofx-capture-intake', daemon=True)
        self.thread.start()

    def _progress(self, event):
        with self.lock:
            self.latest = event

    def progress(self):
        with self.lock:
            return self.latest

    def _run(self, operation):
        try:
            self.cancellation.check()
            value = operation(JobControl(self.cancellation, self._progress))
            result = (value, None)
        except CancelledError as exc:
            result = (None, 'Cancelled: ' + str(exc))
        except BaseException as exc:
            logging.getLogger(__name__).exception('OFX capture operation failed')
            result = (None, f'{type(exc).__name__}: {exc}'[:4096])
        self.results.put(result)

    def cancel(self):
        self.cancellation.cancel('OFX capture operation cancelled by user')


class OfxCaptureDialog(QDialog):
    render_requested = Signal(object)
    cancel_requested = Signal(str)

    def __init__(self, config_json: str, writer_json: str, parent=None, *, publications=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · OFX upstream capture')
        self.resize(850, 760)
        self.config_json, self.writer_json = config_json, writer_json
        self.publications: list[dict[str, str]] = list(publications or [])[-8:]
        self.asset_selection: AssetSelection | None = None
        self.operation: CaptureOperation | None = None
        self.plan: Any = None
        self.prepared: Any = None
        self.candidate: Any = None
        self.record: dict[str, Any] | None = None
        self.submitted_id: str | None = None
        self.rendering = False
        self.review_ready = False
        self.closing = False
        self.generation = 0
        self.operation_generation = 0
        self._cancel_sent = False
        layout = QVBoxLayout(self)
        note = QLabel('In the cached OFX effect, explicitly declare opaque scene-linear BT.709/D65 upstream pixels, choose an integer range and fresh capture parent, then Capture upstream. Select its completed capture.json here. These are effect-location pixels; original media, retiming, proxies and managed colour require target-host validation. Opening this panel does no file I/O or inference. Existing host cache parameters remain unchanged.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.manifest = self._path(form, 'Capture manifest', self.choose_manifest)
        self.scratch = self._path(form, 'Existing conversion parent', self.choose_scratch)
        self.output = self._path(form, 'Fresh render output directory', self.choose_output)
        self.destination = self._path(form, 'Existing cache / receipt parent', self.choose_destination)
        self.read_budget = QSpinBox()
        self.read_budget.setRange(0, 512)
        self.read_budget.setValue(512)
        self.read_budget.setSuffix(' MiB')
        form.addRow('Explicit verification read budget', self.read_budget)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.inventory_button = QPushButton('Inspect metadata')
        self.inventory_button.clicked.connect(self.inspect)
        self.verify_button = QPushButton('Verify captured pixels')
        self.verify_button.clicked.connect(self.verify)
        self.materialize_button = QPushButton('Create reviewed EXR job')
        self.materialize_button.clicked.connect(self.materialize)
        for button in (self.inventory_button, self.verify_button, self.materialize_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        render = QHBoxLayout()
        self.render_button = QPushButton('Submit reviewed render')
        self.render_button.clicked.connect(self.submit_render)
        self.seal_button = QPushButton('Validate render and prepare cache')
        self.seal_button.clicked.connect(self.seal)
        self.cancel_button = QPushButton('Cancel current operation')
        self.cancel_button.clicked.connect(self.cancel)
        for button in (self.render_button, self.seal_button, self.cancel_button):
            render.addWidget(button)
        layout.addLayout(render)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.status = QLabel('No capture inspected. Choose a completed manifest and Inspect metadata.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.attachment = QLineEdit()
        self.attachment.setReadOnly(True)
        self.attachment.setPlaceholderText('Verified attachment receipt appears after preparation')
        layout.addWidget(self.attachment)
        self.retained = QPlainTextEdit()
        self.retained.setReadOnly(True)
        self.retained.setMaximumHeight(100)
        self.retained.setPlaceholderText('Committed local artifacts retained this session appear here, including superseded operations.')
        layout.addWidget(self.retained)
        self._show_publications()
        for field in (self.manifest, self.scratch, self.output):
            field.textChanged.connect(self.invalidate)
        self.read_budget.valueChanged.connect(self.invalidate)
        self._controls()

    def _path(self, form, title, choose):
        row = QHBoxLayout()
        field = QLineEdit()
        row.addWidget(field)
        button = QPushButton('Choose…')
        button.clicked.connect(choose)
        row.addWidget(button)
        form.addRow(title, row)
        return field

    @property
    def io_busy(self):
        return self.operation is not None

    @property
    def busy(self):
        return self.io_busy or self.rendering

    def _controls(self):
        idle = not self.busy and not self.closing
        self.inventory_button.setEnabled(idle)
        self.verify_button.setEnabled(idle)
        self.materialize_button.setEnabled(idle and self.plan is not None and self.candidate is not None)
        self.render_button.setEnabled(idle and self.prepared is not None and self.review_ready and self.submitted_id is None)
        self.seal_button.setEnabled(idle and self.prepared is not None and self.record is not None)
        self.cancel_button.setEnabled(self.busy)
        for field in (self.manifest, self.scratch, self.output, self.destination, self.read_budget):
            field.setEnabled(idle)

    def invalidate(self, *_):
        self.generation += 1
        if self.operation is not None:
            self.operation.cancel()
        if self.rendering:
            self.cancel()
        self.plan = self.prepared = self.candidate = None
        self.review_ready = False
        self.record = None
        if not self.rendering:
            self.submitted_id = None
        self.attachment.clear()
        self.details.clear()
        self.status.setText('Capture choices changed. Inspect and verify again; no render submitted.')
        self._controls()

    def choose_manifest(self):
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Select completed OFX capture manifest', filter='Capture manifest (capture.json)')
        if path:
            self.manifest.setText(path)

    def _choose_parent(self, field, title):
        if self.busy:
            return
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            field.setText(path)

    def choose_scratch(self):
        self._choose_parent(self.scratch, 'Choose existing conversion parent')

    def choose_output(self):
        if self.busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Choose a fresh render output directory')
        if path:
            self.output.setText(path)

    def choose_destination(self):
        self._choose_parent(self.destination, 'Choose existing cache and receipt parent')

    def _start(self, action, operation):
        if self.busy or self.closing:
            return
        try:
            self.operation_generation = self.generation
            self.operation = CaptureOperation(action, operation)
            self.status.setText(action + ' off the interface thread…')
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc)[:4096])
        self._controls()

    def inspect(self):
        manifest = Path(self.manifest.text())
        def operation(control):
            from .ofx_capture import inventory_capture
            return review_text(inventory_capture(manifest, check=control.check))
        self._start('Inspect', operation)

    def verify(self):
        if self.busy or self.closing:
            return
        manifest = Path(self.manifest.text())
        budget = self.read_budget.value() * 1024 * 1024
        config_json, writer_json = self.config_json, self.writer_json
        def operation(control):
            from .ofx_capture import capture_job_config, verify_capture
            from .session import JobRequest
            from .worker_protocol import job_from_wire
            base = asdict(JobRequest(str(Path.cwd()), str(Path.cwd()), config_json, writer_json))
            job = job_from_wire(base)
            config, writer = job.configurations()
            plan = verify_capture(manifest, read_budget=budget, check=control.check)
            candidate = capture_job_config(plan, config)
            from .diagnostics import json_value
            text = review_text({'capture': plan.summary, 'derived_job_settings': json_value(candidate), 'writer': json_value(writer), 'scope': 'Local source starts at0; host offset retained separately. FLOAT EXR inputs preserve captured bits. Explicit Create does no inference; Submit is separate.'})
            return plan, candidate, writer, text
        self.plan = self.prepared = self.candidate = None
        self.review_ready = False
        self.record = None
        self.submitted_id = None
        self.attachment.clear()
        self._start('Verify', operation)

    def materialize(self):
        if self.plan is None or self.candidate is None:
            return
        plan, config, writer = self.plan, self.candidate[0], self.candidate[1]
        scratch, output = Path(self.scratch.text()), Path(self.output.text())
        def operation(control):
            from .ofx_capture import materialize_capture
            prepared = materialize_capture(plan, scratch_parent=scratch, output_dir=output, config=config, writer=writer, check=control.check)
            try:
                return prepared, review_text(prepared.summary), None
            except Exception as exc:
                # Conversion already committed. Preserve its object and path;
                # a display limit must not masquerade as failed file creation.
                logging.getLogger(__name__).exception('Committed capture review unavailable')
                text = 'Committed input workspace retained: ' + str(prepared.workspace)
                return prepared, text, f'{type(exc).__name__}: {exc}'[:2048]
        self._start('Convert', operation)

    def submit_render(self):
        if not self.busy and not self.closing and self.prepared is not None and self.review_ready and self.submitted_id is None:
            self.render_requested.emit(self.prepared.job)

    def mark_submitted(self, request_id: str):
        self.submitted_id = request_id
        self.rendering = True
        self._cancel_sent = False
        self.record = None
        self.status.setText('Explicit render submitted to the existing shared worker. No host cache changed.')
        self._controls()

    def observe(self, event):
        if self.submitted_id is None or event.get('request_id') != self.submitted_id:
            return
        kind = event.get('kind')
        payload = event.get('payload', {})
        if kind in ('result', 'failed', 'cancelled', 'rejected', 'lost'):
            self.rendering = False
            value = payload.get('record')
            self.record = decode_frame(encode_frame(value)) if kind == 'result' and isinstance(value, dict) else None
            self.status.setText('Render completed. Validate and prepare its cache explicitly.' if self.record is not None else f'Render {kind}: {str(payload.get("error", "No completed render"))[:2048]}. Create a new reviewed job to retry.')
        elif kind == 'progress':
            self.status.setText('Shared worker: ' + str(payload.get('stage', 'rendering'))[:256])
        self._controls()

    def seal(self):
        if self.prepared is None or self.record is None:
            return
        prepared, record = self.prepared, decode_frame(encode_frame(self.record))
        parent = Path(self.destination.text())
        def operation(control):
            import tempfile

            from .ofx_cache import seal
            from .ofx_capture import (
                attachment_receipt,
                validate_capture_destination,
                verify_completed_capture,
            )
            destination = validate_capture_destination(prepared, parent, check=control.check)
            verify_completed_capture(prepared, record, check=control.check)
            control.check()
            workspace = Path(tempfile.mkdtemp(prefix='ofx-capture-cache-', dir=destination))
            source = workspace / 'completed-record.json'
            with source.open('xb') as stream:
                stream.write(encode_frame(record))
            cache = seal(source, workspace, control=control)
            try:
                receipt = attachment_receipt(prepared, record, cache=cache, destination_parent=workspace, check=control.check)
            except BaseException as exc:
                # The cache has committed independently of the receipt. Keep
                # its location even when validation fails or is cancelled.
                logging.getLogger(__name__).exception('Committed cache attachment unavailable')
                return cache, None, prepared.host_start, f'{type(exc).__name__}: {exc}'[:2048]
            return cache, receipt, prepared.host_start, None
        self._start('Prepare cache', operation)

    def cancel(self):
        self.status.setText('Cancellation requested; waiting for owned file/worker operation to retire.')
        if self.operation is not None:
            self.operation.cancel()
        if self.rendering and self.submitted_id is not None and not self._cancel_sent:
            self.cancel_requested.emit(self.submitted_id)

    def _show_publications(self):
        raw = json.dumps(self.publications, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(raw) > 65536:
            note = 'Retained history display truncated; newest text follows. Exact paths remain in session records.\n'
            text = note + raw[-(65536 - len(note.encode('utf-8'))):].decode('utf-8', errors='ignore')
        else:
            text = raw.decode('utf-8')
        self.retained.setPlainText(text)

    def _retain_publication(self, action, value):
        if action == 'Convert':
            prepared = value[0]
            record = {'type': 'input_snapshot', 'workspace': str(prepared.workspace),
                      'capture_identity': prepared.capture_identity}
        elif action == 'Prepare cache':
            cache, receipt, start, error = value
            if error is None:
                record = {'type': 'cache_attachment', 'cache': str(cache), 'receipt': str(receipt), 'cache_start': str(start)}
            else:
                record = {'type': 'cache_without_attachment', 'cache': str(cache), 'error': error}
        else:
            return
        self.publications.append(record)
        self.publications = self.publications[-8:]
        self._show_publications()

    def poll(self):
        operation = self.operation
        if operation is not None:
            event = operation.progress()
            if event is not None and not operation.cancellation.cancelled:
                self.status.setText(f'{operation.action}: {event.stage} · {event.completed}/{event.total}')
            # Terminal publication alone does not retire the thread owner.
            if not operation.thread.is_alive():
                try:
                    value, error = operation.results.get_nowait()
                except queue.Empty:
                    value, error = None, 'Owned operation exited without a terminal result.'
                self.operation = None
                if error is None and value is not None:
                    self._retain_publication(operation.action, value)
                if not self.closing and self.operation_generation == self.generation:
                    if error is not None or value is None:
                        self.status.setText(error or 'Owned operation returned no result.')
                    elif operation.action == 'Inspect':
                        self.details.setPlainText(value)
                        self.status.setText('Metadata only. Review range, colour and byte requirements; Verify hashes captured pixels explicitly.')
                    elif operation.action == 'Verify':
                        self.plan, config, writer, text = value
                        self.candidate = config, writer
                        self.details.setPlainText(text)
                        self.status.setText('Verified capture. Review exact derived job settings above before Create reviewed EXR job.')
                    elif operation.action == 'Convert':
                        self.prepared, text, review_error = value
                        self.submitted_id = None
                        self.record = None
                        self._cancel_sent = False
                        self.attachment.clear()
                        self.review_ready = review_error is None
                        self.details.setPlainText(text)
                        self.status.setText('Native FLOAT EXRs verified against captured bits. Review and Submit explicitly; no render or host attachment occurred.' if review_error is None else f'Conversion committed at {self.prepared.workspace}; review unavailable: {review_error}. Submit disabled; retained files were not removed.')
                    else:
                        cache, receipt, start, receipt_error = value
                        if receipt_error is not None:
                            self.attachment.clear()
                            self.status.setText(f'Committed cache retained: {cache}\nValidated attachment unavailable: {receipt_error}. No host cache changed; no attachment receipt was produced.')
                        else:
                            self.attachment.setText(str(receipt))
                            self.status.setText(f'Validated cache: {cache}\nIn the effect explicitly set CacheFile to this cache and CacheStart to {start}. Previous host cache is unchanged; receipt: {receipt}')
                self._controls()
        if self.closing and not self.busy:
            self.close()

    def shutdown(self):
        self.close()
        return not self.busy

    def reject(self):
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.closing = True
        if self.busy:
            self.cancel()
            self._controls()
            event.ignore()
        else:
            event.accept()
