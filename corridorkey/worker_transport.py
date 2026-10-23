"""Bounded/coalesced output and isolated stdio serving for an owned child."""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import uuid
from collections import deque

from .worker_core import SharedEngine, WorkerCore
from .worker_protocol import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
    identifier,
    parse_request,
)

logger = logging.getLogger(__name__)


class Outbox:
    def __init__(self, stream, *, generation=None, max_events=128, max_bytes=32 * 1024 * 1024):
        self.stream = stream
        self.generation = generation or uuid.uuid4().hex
        self.max_events, self.max_bytes = max_events, max_bytes
        self.condition = threading.Condition()
        self.critical: deque[bytes] = deque()
        self.progress: dict[str, bytes] = {}
        self.bytes_pending = 0
        self.closing = False
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self._write, name='protocol-writer', daemon=True)

    def start(self):
        self.thread.start()

    def emit(self, kind, request_id, payload):
        # Reserve largest practical sequence spelling; snapshot caller values now.
        frame = encode_frame({'schema_version': 1, 'generation': self.generation, 'sequence': 2**63 - 1,
                              'kind': kind, 'request_id': request_id, 'payload': payload})
        with self.condition:
            if self.closing or self.failed.is_set():
                raise BrokenPipeError('worker output is closed')
            if kind == 'progress':
                old = self.progress.pop(request_id, None)
                if old is not None:
                    self.bytes_pending -= len(old)
                if len(self.progress) >= 64 or self.bytes_pending + len(frame) > self.max_bytes:
                    return  # Telemetry is coalesced, never allowed to crowd terminal events.
                self.progress[request_id] = frame
            else:
                if kind in ('result', 'failed', 'cancelled'):
                    old = self.progress.pop(request_id, None)
                    if old is not None:
                        self.bytes_pending -= len(old)
                if len(self.critical) >= self.max_events or self.bytes_pending + len(frame) > self.max_bytes:
                    self.failed.set()
                    raise BufferError('worker critical output backpressure limit exceeded')
                self.critical.append(frame)
            self.bytes_pending += len(frame)
            self.condition.notify_all()

    def _write(self):
        sequence = 0
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.critical or self.progress or self.closing or self.failed.is_set())
                    if self.failed.is_set():
                        return
                    if self.critical:
                        frame = self.critical.popleft()
                    elif self.progress:
                        frame = self.progress.pop(next(iter(self.progress)))
                    else:
                        return
                    self.bytes_pending -= len(frame)
                value = decode_frame(frame)
                value['sequence'] = sequence
                data = encode_frame(value)
                view = memoryview(data)
                while view:
                    count = self.stream.write(view)
                    if count is None or count <= 0:
                        raise BrokenPipeError('worker output stopped accepting bytes')
                    view = view[count:]
                self.stream.flush()
                sequence += 1
        except (OSError, ValueError, TypeError):
            logger.exception('worker output disconnected')
            self.failed.set()
        finally:
            with self.condition:
                self.condition.notify_all()

    def close(self, timeout=5.0):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout)
        return not self.thread.is_alive() and not self.failed.is_set()


def _read_input(fd, messages, stop):
    """Use os.read so a daemon blocked on stdin holds no Python buffered-I/O lock."""
    buffer = bytearray()
    search = 0

    def put(value):
        while not stop.is_set():
            try:
                messages.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    try:
        while not stop.is_set():
            chunk = os.read(fd, 4096)
            if not chunk:
                if buffer:
                    raise ValueError('unterminated worker input frame')
                break
            buffer.extend(chunk)
            while True:
                end = buffer.find(b'\n', search)
                if end < 0:
                    search = len(buffer)
                    if len(buffer) > MAX_FRAME_BYTES:
                        raise ValueError('worker input frame exceeds byte limit')
                    break
                frame = bytes(buffer[:end + 1])
                if len(frame) > MAX_FRAME_BYTES:
                    raise ValueError('worker input frame exceeds byte limit')
                del buffer[:end + 1]
                search = 0
                put(('frame', frame))
    except (OSError, ValueError) as exc:
        put(('error', str(exc)))
    finally:
        put(('eof', None))


def serve_stdio(*, capacity=4, engine=None, shutdown_timeout=5.0):
    # Duplicate the protocol descriptor, then route Python AND native fd1 logs to stderr.
    sys.stdout.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with os.fdopen(protocol_fd, 'wb', buffering=0) as protocol:
        outbox = Outbox(protocol)
        outbox.start()
        outbox.emit('ready', None, {'pid': os.getpid(), 'capacity': capacity, 'protocol': 1,
                                    'progress': 'coalesced', 'max_frame_bytes': MAX_FRAME_BYTES})
        core = WorkerCore(engine if engine is not None else SharedEngine(), outbox.emit, capacity=capacity)
        messages = queue.Queue(maxsize=8)
        stop_reader = threading.Event()
        reader = threading.Thread(target=_read_input, args=(sys.stdin.fileno(), messages, stop_reader),
                                  name='protocol-reader', daemon=True)
        reader.start()
        clean_input = True
        interrupted = False
        try:
            while not core.done.is_set():
                if outbox.failed.is_set():
                    core.stop()
                    break
                try:
                    kind, frame = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == 'eof':
                    core.stop()
                    break
                if kind == 'error':
                    outbox.emit('rejected', None, {'code': 'framing_error', 'message': frame})
                    clean_input = False
                    core.stop()
                    break
                request_id = None
                try:
                    value = decode_frame(frame)
                    try:
                        request_id = identifier(value.get('request_id'))
                    except ValueError:
                        pass
                    request = parse_request(value)
                except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                    outbox.emit('rejected', request_id, {'code': 'invalid_request', 'message': str(exc)})
                    continue
                core.dispatch(request)
                if request.action == 'shutdown':
                    break
        except KeyboardInterrupt:
            interrupted = True
            core.stop()
        except (OSError, BufferError):
            logger.exception('worker connection failed')
            clean_input = False
            core.stop()
        finally:
            stop_reader.set()
            clean_engine = core.finish(shutdown_timeout)
            clean_output = outbox.close(shutdown_timeout)
        return 130 if interrupted else 0 if clean_input and clean_engine and clean_output else 2
