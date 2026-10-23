"""Single background owner for correction probing/hash/I/O and history operations."""
from __future__ import annotations

import logging
import queue
import threading
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
        self.serial = 0
        self.thread = threading.Thread(target=self._run, name='correction-editor', daemon=True)
        self.thread.start()

    def check(self):
        if self.stopping.is_set():
            raise CancelledError('Correction editor closed')

    def request(self, action: str, **payload: Any) -> int:
        if not self.ready.is_set() or self.stopping.is_set() or self.done.is_set():
            raise RuntimeError('Correction editor is not ready.')
        serial = self.serial + 1
        self.commands.put_nowait((serial, action, payload))
        self.serial = serial
        return serial

    def _publish(self, event: dict[str, Any]):
        while not self.stopping.is_set():
            try:
                self.events.put(event, timeout=.05)
                return
            except queue.Full:
                continue

    def stop(self):
        self.stopping.set()

    def _run(self):
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
                    self.check()
                    view = None
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
                    elif action == 'save':
                        model.save()
                    else:
                        raise ValueError('Unsupported correction editor action.')
                    snapshot = model.snapshot()
                    overlay_kind = 'coverage'
                    if action == 'view':
                        selected = next((e for e in snapshot['document']['entries'] if e['id'] == payload.get('identifier')), None)
                        if selected is not None and selected['kind'] == 'trimap':
                            overlay_kind = 'trimap'
                    self._publish({'kind': 'result', 'serial': serial, 'action': action,
                                   'snapshot': snapshot, 'view': view, 'overlay_kind': overlay_kind})
                except Exception as exc:
                    logging.getLogger(__name__).exception('Correction editor operation failed')
                    self._publish({'kind': 'error', 'serial': serial, 'action': action,
                                   'error': f'{type(exc).__name__}: {exc}'})
        except CancelledError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).exception('Correction editor binding failed')
            self._publish({'kind': 'error', 'serial': 0, 'error': f'{type(exc).__name__}: {exc}'})
        finally:
            self.done.set()
