"""Latest-request-only preview decoding, independent of the Qt event thread."""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from .control import CancelledError
from .desktop_preview import PreviewFrame, PreviewPair, preview, preview_pair
from .preview_sequence import PreparedSequence, prepare_sequence


class PreviewLoader:
    def __init__(self):
        self.requests: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue(maxsize=1)
        self.results: queue.Queue[tuple[int, PreviewFrame | PreviewPair | PreparedSequence | None, str | None]] = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.serial = 0
        self.thread = threading.Thread(target=self._run, name='desktop-preview', daemon=True)
        self.thread.start()

    def submit(self, **settings: Any) -> int:
        self.serial += 1
        try:
            self.requests.get_nowait()
        except queue.Empty:
            pass
        self.requests.put_nowait((self.serial, settings))
        return self.serial

    def cancel(self, serial: int) -> None:
        """Cooperatively fence a queued/preparing source index; native reads finish."""
        if serial == self.serial:
            self.serial += 1

    def _check_request(self, serial: int) -> None:
        if self.stopping.is_set() or serial != self.serial:
            raise CancelledError('Source preparation superseded or cancelled.')

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                serial, settings = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if 'prepare_sequence' in settings:
                    frame = prepare_sequence(settings['prepare_sequence'], check=lambda serial=serial: self._check_request(serial))
                else:
                    frame = preview_pair(**settings['pair']) if 'pair' in settings else preview(**settings)
                error = None
            except CancelledError as exc:
                frame, error = None, str(exc)
            except Exception as exc:
                logging.getLogger(__name__).exception('Desktop preview failed')
                frame, error = None, f'{type(exc).__name__}: {exc}'
            # Keep at most one completed thumbnail, never a full sequence.
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self.results.put_nowait((serial, frame, error))
