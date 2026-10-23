"""Nonblocking desktop command/event adapter around the existing owned worker."""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .queue_store import QueueStore
from .worker_client import WorkerCleanupError, WorkerClient
from .worker_protocol import decode_frame, request_frame


class DesktopBridge:
    """One transport owner, bounded commands/events and coalesced progress.

    Construction/startup and shutdown never run on the GUI thread. A timeout
    observes the same worker; requests are never automatically retried.
    """

    def __init__(self, factory: Callable[..., WorkerClient] = WorkerClient, *, journal_workspace: Path | None = None):
        self.factory = factory
        self.journal_workspace = journal_workspace
        self.store: QueueStore | None = None
        self.commands: queue.Queue[tuple[str, dict[str, Any], str]] = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.events: deque[dict[str, Any]] = deque()
        self.progress: dict[str, dict[str, Any]] = {}
        self.stopping = threading.Event()
        self.done = threading.Event()
        self.failure: str | None = None
        self.cleanup_error: str | None = None
        self.thread = threading.Thread(target=self._run, name='desktop-worker-owner', daemon=True)
        self.thread.start()

    def send(self, action: str, payload: dict[str, Any]) -> str:
        if self.stopping.is_set() or self.done.is_set():
            raise RuntimeError(self.failure or 'Desktop worker connection is closed.')
        request_id = uuid.uuid4().hex
        normalized = decode_frame(request_frame(request_id, action, payload))['payload']
        self.commands.put_nowait((action, normalized, request_id))
        return request_id

    def _publish(self, event: dict[str, Any]) -> None:
        if self.store is not None:
            if event['kind'] == 'connected':
                self.store.connected(event['payload']['pid'], event['payload']['generation'])
            self.store.observe(event)
        with self.lock:
            request_id = event.get('request_id')
            if event['kind'] == 'progress':
                self.progress[str(request_id)] = event
            else:
                self.progress.pop(str(request_id), None)
                if len(self.events) >= 128:
                    raise BufferError('Desktop events were not consumed; connection stopped.')
                self.events.append(event)

    def poll(self) -> list[dict[str, Any]]:
        with self.lock:
            result = [*self.events, *self.progress.values()]
            self.events.clear()
            self.progress.clear()
            return result

    def stop(self) -> None:
        self.stopping.set()

    def _run(self) -> None:
        client: WorkerClient | None = None
        try:
            if self.journal_workspace is not None:
                self.store = QueueStore(self.journal_workspace)
                session = self.store.start_session()
                self._publish({'kind': 'queue_opened', 'request_id': None,
                               'payload': {'workspace': str(self.journal_workspace), 'session': session}})
            while not self.stopping.is_set():
                try:
                    action, payload, request_id = self.commands.get(timeout=0.02)
                except queue.Empty:
                    pass
                else:
                    sent = False
                    try:
                        tracked = self.store is not None and action in {'render', 'recover'}
                        if tracked:
                            assert self.store is not None
                            payload = self.store.prepare(request_id, action, payload)
                            self._publish({'kind': 'recorded', 'request_id': request_id,
                                           'payload': {'workspace': str(self.journal_workspace),
                                                       **({'batch': payload['batch']} if action == 'recover' else {})}})
                        if client is None:
                            client = self.factory()
                            self._publish({'kind': 'connected', 'request_id': None,
                                           'payload': {'pid': client.process.pid, 'generation': client.generation}})
                        if tracked:
                            assert self.store is not None
                            self.store.phase(request_id, 'dispatching')
                        client.send(action, payload, request_id=request_id)
                        sent = True
                        if tracked:
                            assert self.store is not None
                            self.store.phase(request_id, 'sent')
                    except WorkerCleanupError:
                        # A failed constructor has no client value to inspect in
                        # finally. Retain its typed ownership uncertainty below.
                        raise
                    except (OSError, ValueError, TypeError, RuntimeError, BufferError) as exc:
                        if sent:
                            raise
                        self._publish({'kind': 'rejected', 'request_id': request_id,
                                       'payload': {'error': str(exc)}})
                if client is not None:
                    while True:
                        try:
                            event = client.next_event(timeout=0)
                        except TimeoutError:
                            break
                        self._publish(event)
        except WorkerCleanupError as exc:
            logging.getLogger(__name__).exception('Desktop worker startup cleanup was not verified')
            self.cleanup_error = exc.cleanup_error
            self.failure = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            logging.getLogger(__name__).exception('Desktop worker connection failed')
            self.failure = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                while True:
                    try:
                        action, payload, request_id = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if action in {'render', 'recover'}:
                        if self.store is not None:
                            self.store.prepare(request_id, action, payload)
                        self._publish({'kind': 'cancelled', 'request_id': request_id,
                                       'payload': {'reason': 'Desktop stopped before dispatch', 'executed': False}})
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                logging.getLogger(__name__).exception('Desktop pending request persistence failed')
                self.failure = self.failure or str(exc)
            try:
                if client is not None:
                    clean = False
                    try:
                        clean = client.close()
                    except BaseException as exc:
                        logging.getLogger(__name__).exception('Desktop worker close failed')
                        self.cleanup_error = f'Worker close did not verify owned cleanup: {type(exc).__name__}: {exc}'[:4096]
                    finally:
                        if client.cleanup_error is not None:
                            self.cleanup_error = str(client.cleanup_error)[:4096]
                        live = [name for name in ('reader', 'writer', 'log_reader')
                                if isinstance(getattr(client, name, None), threading.Thread)
                                and getattr(client, name).is_alive()]
                        if live:
                            self.cleanup_error = self.cleanup_error or ('Owned worker I/O threads remain live: ' + ', '.join(live))
                    if self.cleanup_error is not None:
                        self.failure = self.failure or self.cleanup_error
                    # Drain actual terminal/lost records generated during close.
                    try:
                        while True:
                            self._publish(client.next_event(timeout=0))
                    except (TimeoutError, EOFError, RuntimeError, BufferError):
                        pass
                    if not clean:
                        self.failure = self.failure or str(client.cleanup_error or client.failure or 'Unclean worker shutdown')
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                logging.getLogger(__name__).exception('Desktop worker cleanup failed')
                self.failure = self.failure or str(exc)
            finally:
                if self.store is not None:
                    try:
                        self.store.finish(self.failure is None, self.failure)
                    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                        logging.getLogger(__name__).exception('Desktop queue session close failed')
                        self.failure = self.failure or str(exc)
                    finally:
                        try:
                            self.store.close()
                        except (OSError, sqlite3.Error) as exc:
                            self.failure = self.failure or str(exc)
                self.done.set()
