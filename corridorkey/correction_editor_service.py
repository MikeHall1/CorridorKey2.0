"""Single background owner for correction probing/hash/I/O and history operations."""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .control import CancelledError
from .correction_editor import CorrectionEditor
from .session import JobRequest


class EditorService:
    def __init__(self, job: JobRequest, parent: Path, *, existing: Path | None = None):
        self.job, self.parent, self.existing = job, parent, existing
        self.commands: queue.Queue[tuple[int, str, dict[str, Any]]] = queue.Queue(maxsize=1)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self.stopping, self.done, self.ready = threading.Event(), threading.Event(), threading.Event()
        self.cancel_current = threading.Event()
        self.operation_pending = threading.Event()
        self.request_lock = threading.Lock()
        self.serial = 0
        self.thread = threading.Thread(target=self._run, name='correction-editor', daemon=True)
        self.thread.start()

    def check(self):
        if self.stopping.is_set():
            raise CancelledError('Correction editor closed')
        if self.cancel_current.is_set():
            raise CancelledError('Correction editor operation cancelled')

    def request(self, action: str, **payload: Any) -> int:
        with self.request_lock:
            if not self.ready.is_set() or self.stopping.is_set() or self.done.is_set():
                raise RuntimeError('Correction editor is not ready.')
            if self.operation_pending.is_set():
                raise RuntimeError('Wait for the owned correction operation to retire before another request.')
            serial = self.serial + 1
            self.cancel_current.clear()
            self.operation_pending.set()
            try:
                self.commands.put_nowait((serial, action, payload))
            except queue.Full:
                self.operation_pending.clear()
                raise
            self.serial = serial
            return serial

    def _publish(self, event: dict[str, Any]):
        self.operation_pending.clear()
        while not self.stopping.is_set():
            try:
                self.events.put(event, timeout=.05)
                return
            except queue.Full:
                continue

    def stop(self):
        self.stopping.set()

    def _run(self):
        model = None
        try:
            model = CorrectionEditor(self.job, self.parent, existing=self.existing, check=self.check)
            self.ready.set()
            self._publish({'kind': 'ready', 'serial': 0, 'snapshot': model.snapshot()})
            while not self.stopping.is_set():
                try:
                    serial, action, payload = self.commands.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    if action in ('subject_prepare', 'subject_predict', 'subject_clear', 'subject_discard'):
                        from .subject_guidance_editor import invalidate_subject_result
                        invalidate_subject_result(model)
                    self.check()
                    view = None
                    if action == 'plate_view':
                        overlay_kind = 'trimap' if payload.get('mode', 'guidance') == 'guidance' else 'coverage'
                    if action == 'view':
                        view = model.view(**payload)
                    elif action == 'brush':
                        model.commit_brush(**payload)
                    elif action == 'import':
                        model.import_mask(**payload)
                    elif action == 'import_trimap':
                        model.import_trimap(**payload)
                    elif action == 'reset':
                        model.reset(**payload)
                    elif action == 'edit':
                        model.edit(**payload)
                    elif action == 'remove':
                        model.remove(**payload)
                    elif action == 'move':
                        model.move(**payload)
                    elif action == 'shot':
                        model.shot(**payload)
                    elif action == 'undo':
                        model.undo()
                    elif action == 'redo':
                        model.redo()
                    elif action == 'track_preview':
                        model.prepare_tracking(**payload)
                    elif action == 'track_view':
                        view = model.tracking_view(**payload)
                    elif action == 'track_overlay_view':
                        view = model.tracking_overlay_view(**payload)
                    elif action == 'track_accept':
                        model.accept_tracking()
                    elif action == 'track_discard':
                        model.discard_tracking()
                    elif action == 'track_clear':
                        model.clear_tracking()
                    elif action == 'plate_preview':
                        model.prepare_plate(**payload)
                    elif action == 'plate_view':
                        view = model.plate_view(**payload)
                    elif action == 'plate_accept':
                        model.accept_plate()
                    elif action == 'plate_discard':
                        model.discard_plate()
                    elif action == 'detach_guidance':
                        model.detach_guidance(**payload)
                    elif action == 'subject_prepare':
                        model.prepare_subject(**payload)
                        view = model.subject_view()
                    elif action == 'subject_predict':
                        model.predict_subject(**payload)
                    elif action == 'subject_view':
                        view = model.subject_view()
                    elif action == 'subject_accept':
                        model.accept_subject()
                    elif action == 'subject_clear':
                        model.clear_subject()
                    elif action == 'subject_discard':
                        model.discard_subject()
                    elif action == 'save':
                        model.save()
                    else:
                        raise ValueError('Unsupported correction editor action.')
                    snapshot = model.snapshot()
                    overlay_kind = 'coverage'
                    if action == 'plate_view':
                        overlay_kind = 'trimap' if payload.get('mode', 'guidance') == 'guidance' else 'coverage'
                    if action == 'view':
                        selected = next((e for e in snapshot['document']['entries'] if e['id'] == payload.get('identifier')), None)
                        if selected is not None and selected['kind'] == 'trimap':
                            overlay_kind = 'trimap'
                    self._publish({'kind': 'result', 'serial': serial, 'action': action,
                                   'snapshot': snapshot, 'view': view, 'overlay_kind': overlay_kind,
                                   'plate_reference': model.plate_reference()})
                except Exception as exc:
                    logging.getLogger(__name__).exception('Correction editor operation failed')
                    failure: dict[str, Any] = {'kind': 'error', 'serial': serial, 'action': action,
                                               'error': f'{type(exc).__name__}: {exc}'}
                    try:
                        failure['snapshot'] = model.snapshot()
                    except Exception as snapshot_error:
                        logging.getLogger(__name__).exception('Correction editor failure snapshot unavailable')
                        failure['snapshot_error'] = f'{type(snapshot_error).__name__}: {snapshot_error}'
                    self._publish(failure)
        except CancelledError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).exception('Correction editor binding failed')
            self._publish({'kind': 'error', 'serial': 0, 'error': f'{type(exc).__name__}: {exc}'})
        finally:
            if model is not None:
                try:
                    model.discard_tracking()
                    model.discard_plate()
                    model.discard_subject()
                    # The dialog must retain its service owner while a failed
                    # child teardown is still live/unverified. Never advertise
                    # done merely because the editor thread reached finally.
                    while model.subject_preview is not None:
                        time.sleep(.1)
                        model.discard_subject()
                except OSError:
                    logging.getLogger(__name__).exception('Could not remove private tracking preview')
            self.done.set()
