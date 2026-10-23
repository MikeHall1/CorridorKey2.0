"""One owned cache preparation; immutable record bytes, coalesced progress."""
from __future__ import annotations

import logging
import queue
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .control import CancellationToken, CancelledError, JobControl, ProgressEvent
from .worker_protocol import encode_frame


@dataclass(frozen=True)
class PreparationResult:
    status: str
    workspace: Path | None
    cache: Path | None
    error: str | None


class CachePreparation:
    def __init__(self, record: dict[str, Any], parent: Path):
        self.raw = encode_frame(record)
        self.parent = parent.absolute()
        self.cancellation = CancellationToken()
        self.done = threading.Event()
        self.results: queue.Queue[PreparationResult] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest: ProgressEvent | None = None
        self.thread = threading.Thread(target=self._run, name='ofx-cache-preparation', daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        return self.thread.is_alive() or not self.results.empty()

    def progress(self) -> ProgressEvent | None:
        with self.lock:
            return self.latest

    def _progress(self, event: ProgressEvent):
        with self.lock:
            self.latest = event

    def cancel(self):
        self.cancellation.cancel('Cache preparation cancelled by user')

    def _run(self):
        workspace = None
        result = PreparationResult('failed', None, None, 'Unexpected preparation worker termination')
        try:
            control = JobControl(self.cancellation, self._progress)
            control.checkpoint('ofx_save_record')
            parent = self.parent.resolve()
            control.check()
            if not parent.is_dir():
                raise ValueError('Choose an existing destination folder.')
            control.check()
            workspace = Path(tempfile.mkdtemp(prefix='ofx-preparation-', dir=parent))
            record = workspace / 'completed-record.json'
            with record.open('xb') as stream:
                stream.write(self.raw)
            control.checkpoint('ofx_load_image_adapters')
            # Keep Torch/numeric imports out of desktop startup and the event thread.
            from .ofx_cache import seal

            cache = seal(record, workspace, control=control)
            result = PreparationResult('complete', workspace, cache, None)
        except CancelledError as exc:
            result = PreparationResult('cancelled', workspace, None, str(exc))
        except BaseException as exc:
            logging.getLogger(__name__).exception('Cache preparation failed')
            result = PreparationResult('failed', workspace, None, f'{type(exc).__name__}: {exc}'[:4096])
        finally:
            self.results.put(result)
            self.done.set()
