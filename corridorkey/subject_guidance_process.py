"""Owned persistent CPU guidance child; stdlib transport, no automatic assets.

The parent performs no model work. One synchronous call belongs on an external
background owner. Wire frames and queues are bounded; cancel/timeout retires the
child before reuse. Source execution uses the current application interpreter;
frozen application dispatch is explicitly unavailable until separately wired.
No CUDA/MPS resource-handoff or unobserved detached-descendant guarantee is made.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .control import CancelledError, JobControl
from .subject_guidance import (
    Click,
    ExclusiveResourceLease,
    GuidanceFrame,
    GuidanceResult,
    GuidanceSession,
    GuidanceUnavailable,
    LocalSAM2Provider,
)

MAX_MESSAGE_BYTES = 32 * 1024**2
MAX_PIXELS = 4 * 1024**2
ROOT = Path(__file__).resolve().parents[1]


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate guidance protocol field.')
        result[key] = value
    return result


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('Nonfinite guidance protocol JSON number.')
    return parsed


def _invalid_constant(value):
    raise ValueError('Nonfinite guidance protocol JSON is unsupported: ' + value)


def _encode(value) -> bytes:
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError('Guidance protocol payload exceeds 32 MiB.')
    return struct.pack('>I', len(data)) + data


class _Reader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


def _read(stream: _Reader):
    header = stream.read(4)
    if not header:
        raise EOFError('Guidance child protocol closed.')
    if len(header) != 4:
        raise ValueError('Truncated guidance frame header.')
    size = struct.unpack('>I', header)[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError('Guidance protocol frame exceeds its admitted byte bound.')
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(min(remaining, 65536))
        if not chunk:
            raise EOFError('Truncated guidance child frame.')
        chunks.append(chunk)
        remaining -= len(chunk)
    value = json.loads(b''.join(chunks), object_pairs_hook=_pairs, parse_constant=_invalid_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise TypeError('Guidance protocol frame must be an object.')
    return value


def _bytes(value, limit):
    if not isinstance(value, str) or len(value) > ((limit + 2) // 3) * 4:
        raise ValueError('Encoded guidance pixels exceed their byte bound.')
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('Invalid guidance pixel encoding.') from exc
    if len(result) > limit:
        raise ValueError('Decoded guidance pixels exceed their byte bound.')
    return result


class SubjectGuidanceProcess:
    """Lazy persistent CPU float32 owner; fixture commands are test-only injection.

    While awaiting a reply, check uses <=50 ms polling intervals; bounded
    serialization/validation is checked before and after its local operations.
    Clear during prediction cancels and reaps that owner; an idle clear preserves
    the frame embedding and resets prompts/logits. Cleanup uncertainty prevents
    another owner being started. No workspace files are created by this transport.
    """
    def __init__(self, config: Path, checkpoint: Path, *, workspace: Path,
                 fixture_command: Sequence[str] | None = None, startup_timeout: float = 20,
                 operation_timeout: float = 120, grace_seconds: float = 1):
        for value in (startup_timeout, operation_timeout, grace_seconds):
            if type(value) not in (int, float) or not 0 < value <= 3600:
                raise ValueError('Guidance process deadlines must be finite positive seconds <=3600.')
        if fixture_command is not None and (isinstance(fixture_command, (str, bytes)) or not fixture_command
                or any(not isinstance(item, str) or not item for item in fixture_command)):
            raise ValueError('Fixture command must be an explicit nonempty argument vector.')
        self.config, self.checkpoint = Path(config).absolute(), Path(checkpoint).absolute()
        self.workspace = Path(workspace).absolute()
        if not self.workspace.is_dir():
            raise ValueError('Guidance workspace must be an existing local directory.')
        self.fixture_command = None if fixture_command is None else tuple(fixture_command)
        self.startup_timeout, self.operation_timeout, self.grace_seconds = startup_timeout, operation_timeout, grace_seconds
        self._state, self._lifecycle, self._operation = threading.Lock(), threading.RLock(), threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._events: queue.Queue[dict] = queue.Queue(maxsize=2)
        self._writes: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._fatal: str | None = None
        self._logs: deque[bytes] = deque(maxlen=16)
        self._serial, self.generation = 0, 0
        self.closed = False
        self.cleanup_error: str | None = None

    @property
    def pid(self) -> int | None:
        with self._lifecycle:
            return None if self._process is None else self._process.pid

    def _notice(self, event):
        try:
            self._events.put_nowait(event)
        except queue.Full:
            with self._state:
                self._fatal = 'Guidance child exceeded its bounded reply capacity.'

    def _launch(self, check, generation):
        with self._lifecycle:
            with self._state:
                if self.closed or self.generation != generation:
                    raise CancelledError('Guidance owner closed or superseded before child admission.')
            if self.cleanup_error is not None:
                raise GuidanceUnavailable('Previous guidance cleanup is NOT VERIFIED: ' + self.cleanup_error)
            if self._process is not None:
                return
            if self.fixture_command is None and getattr(sys, 'frozen', False):
                raise GuidanceUnavailable('Frozen application guidance dispatch is not wired; source interpreter execution is required.')
            command = list(self.fixture_command) if self.fixture_command is not None else [
                sys.executable, '-u', '-m', 'corridorkey.subject_guidance_process', '--serve',
                '--config', str(self.config), '--checkpoint', str(self.checkpoint)]
            self._stop_event = threading.Event()
            self._events, self._writes = queue.Queue(maxsize=2), queue.Queue(maxsize=1)
            self._fatal = None
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            self._process = process
            assert process.stdout is not None and process.stdin is not None and process.stderr is not None
            output, input_stream, error_stream = process.stdout, process.stdin, process.stderr
            stopping, writes = self._stop_event, self._writes
            def reader():
                try:
                    while not stopping.is_set():
                        self._notice(_read(output))
                except (OSError, EOFError, ValueError, TypeError, RecursionError) as exc:
                    if not stopping.is_set():
                        self._notice({'transport_error': f'{type(exc).__name__}: {exc}'[:2048]})
            def writer():
                try:
                    while not stopping.is_set():
                        try:
                            payload = writes.get(timeout=.05)
                        except queue.Empty:
                            continue
                        view = memoryview(payload)
                        while view and not stopping.is_set():
                            written = os.write(input_stream.fileno(), view[:65536])
                            if written <= 0:
                                raise BrokenPipeError('Guidance pipe stopped accepting bytes.')
                            view = view[written:]
                except OSError as exc:
                    if not stopping.is_set():
                        self._notice({'transport_error': f'{type(exc).__name__}: {exc}'[:2048]})
            def logs():
                try:
                    while chunk := os.read(error_stream.fileno(), 4096):
                        with self._state:
                            self._logs.append(chunk)
                except OSError:
                    pass
            self._threads = [threading.Thread(target=fn, name='guidance-' + name, daemon=True)
                             for fn, name in ((reader, 'reader'), (writer, 'writer'), (logs, 'logs'))]
            for thread in self._threads:
                thread.start()
        ready = self._wait(check, generation, self.startup_timeout)
        if type(ready.get('version')) is not int or ready != {'type': 'ready', 'version': 1, 'device': 'cpu', 'precision': 'float32'}:
            raise ValueError('Guidance child returned an invalid CPU float32 handshake.')

    def _wait(self, check, generation, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if check is not None:
                check()
            with self._state:
                if self.closed or self.generation != generation:
                    raise CancelledError('Guidance request superseded or owner closed.')
                if self._fatal is not None:
                    raise RuntimeError(self._fatal)
            if time.monotonic() >= deadline:
                raise TimeoutError('Guidance child operation timed out; its owner will be retired.')
            try:
                event = self._events.get(timeout=min(.05, max(.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if 'transport_error' in event:
                raise RuntimeError(event['transport_error'])
            return event

    def _exchange(self, action, payload, check, generation, timeout=None):
        self._serial += 1
        serial = self._serial
        if check is not None:
            check()
        wire = _encode({'version': 1, 'id': serial, 'generation': generation, 'action': action, 'payload': payload})
        if check is not None:
            check()
        self._writes.put_nowait(wire)
        reply = self._wait(check, generation, self.operation_timeout if timeout is None else timeout)
        if (set(reply) != {'version', 'id', 'generation', 'type', 'payload'} or type(reply['version']) is not int or reply['version'] != 1
                or type(reply['id']) is not int or reply['id'] != serial or type(reply['generation']) is not int
                or reply['generation'] != generation or reply['type'] not in ('result', 'unavailable', 'error')):
            raise ValueError('Guidance reply identity/schema does not match its owned request.')
        if reply['type'] != 'result':
            message = reply['payload']
            if not isinstance(message, str) or len(message) > 4096:
                raise ValueError('Malformed guidance error reply.')
            if reply['type'] == 'unavailable':
                raise GuidanceUnavailable(message)
            raise RuntimeError(message)
        return reply['payload']

    def predict(self, frame: GuidanceFrame, clicks: tuple[Click, ...], *, check: Callable[[], None] | None = None) -> GuidanceResult:
        if not isinstance(frame, GuidanceFrame):
            raise TypeError('Guidance prediction requires a frame instance.')
        frame.validate(MAX_PIXELS)
        if type(clicks) is not tuple or not 1 <= len(clicks) <= 256:
            raise ValueError('Guidance clicks must be an immutable bounded tuple.')
        for click in clicks:
            if not isinstance(click, Click):
                raise TypeError('Guidance clicks must be Click instances.')
            click.validate(frame.width, frame.height)
        if not self._operation.acquire(blocking=False):
            raise RuntimeError('Guidance process is busy; cancel/clear and await the current operation.')
        try:
            with self._state:
                if self.closed:
                    raise RuntimeError('Guidance process is closed.')
                generation = self.generation
            if check is not None:
                check()
            self._launch(check, generation)
            fields = {'source_sha256': frame.source_sha256, 'source_frame': frame.source_frame,
                      'decode_identity': frame.decode_identity, 'width': frame.width, 'height': frame.height,
                      'rgb_bytes': base64.b64encode(frame.rgb_bytes).decode('ascii')}
            payload = self._exchange('predict', {'frame': fields, 'clicks': [asdict(c) for c in clicks]}, check, generation)
            if not isinstance(payload, dict) or set(payload) != set(GuidanceResult.__dataclass_fields__):
                raise ValueError('Guidance result has unknown/missing fields.')
            if (payload['frame_identity'] != frame.identity or payload['source_sha256'] != frame.source_sha256
                    or type(payload['source_frame']) is not int or payload['source_frame'] != frame.source_frame
                    or payload['decode_identity'] != frame.decode_identity or type(payload['width']) is not int
                    or type(payload['height']) is not int or (payload['width'], payload['height']) != (frame.width, frame.height)
                    or type(payload['generation']) is not int or payload['generation'] < 0 or payload['requires_acceptance'] is not True):
                raise ValueError('Guidance result changed source/frame/decode/geometry identity.')
            if payload['mask_format'] not in ('binary_u8', 'coverage_f32'):
                raise ValueError('Unsupported guidance mask format.')
            size = frame.width * frame.height * (4 if payload['mask_format'] == 'coverage_f32' else 1)
            raw = _bytes(payload['mask_bytes'], size)
            if len(raw) != size:
                raise ValueError('Guidance result mask geometry disagrees.')
            if payload['mask_format'] == 'binary_u8':
                if any(v not in (0, 255) for v in raw):
                    raise ValueError('Guidance binary mask contains nonbinary values.')
            else:
                if any(not math.isfinite(v[0]) or not 0 <= v[0] <= 1 for v in struct.iter_unpack('<f', raw)):
                    raise ValueError('Guidance coverage is not finite float32 in [0,1].')
            provenance = payload['provenance_json']
            if not isinstance(provenance, str) or len(provenance.encode()) > 128 * 1024:
                raise ValueError('Guidance result provenance exceeds bounds.')
            observed = json.loads(provenance, object_pairs_hook=_pairs, parse_constant=_invalid_constant, parse_float=_finite_float)
            json.dumps(observed, allow_nan=False)
            if (not isinstance(observed, dict) or observed.get('frame_identity') != frame.identity
                    or observed.get('source_sha256') != frame.source_sha256 or observed.get('source_frame') != frame.source_frame
                    or observed.get('decode_identity') != frame.decode_identity or observed.get('generation') != payload['generation']
                    or observed.get('mask_sha256') != hashlib.sha256(raw).hexdigest() or observed.get('device') != 'cpu'
                    or observed.get('mask_format') != payload['mask_format']
                    or json.dumps(observed.get('clicks'), sort_keys=True) != json.dumps([asdict(c) for c in clicks], sort_keys=True)):
                raise ValueError('Guidance result provenance does not bind the observed CPU frame and mask.')
            if check is not None:
                check()
            with self._state:
                if self.closed or self.generation != generation:
                    raise CancelledError('Guidance result became stale before publication.')
            return GuidanceResult(**{**payload, 'mask_bytes': raw})
        except BaseException as original:
            try:
                self._retire()
            except RuntimeError as cleanup:
                original.add_note(str(cleanup))
            raise
        finally:
            self._operation.release()

    def clear(self) -> None:
        with self._state:
            if self.closed:
                raise RuntimeError('Guidance process is closed.')
            self.generation += 1
            generation = self.generation
        if not self._operation.acquire(blocking=False):
            self._retire()
            return
        try:
            if self._process is not None and self._exchange('clear', {}, None, generation, timeout=min(5, self.operation_timeout)) != {'cleared': True}:
                raise ValueError('Malformed guidance clear acknowledgement.')
        except BaseException as original:
            try:
                self._retire()
            except RuntimeError as cleanup:
                original.add_note(str(cleanup))
            raise
        finally:
            self._operation.release()

    def _retire(self) -> None:
        with self._lifecycle:
            process = self._process
            if process is None:
                return
            self._stop_event.set()
            try:
                # Only this lifecycle lock polls/reaps/signals the Popen root.
                if process.poll() is None:
                    try:
                        if os.name == 'nt':
                            process.terminate()
                        else:
                            os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=self.grace_seconds)
                    except subprocess.TimeoutExpired:
                        if process.poll() is None:
                            try:
                                if os.name == 'nt':
                                    process.kill()
                                else:
                                    os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        process.wait(timeout=5)
                for thread in self._threads:
                    if thread.ident is not None:
                        thread.join(2)
                if any(thread.is_alive() for thread in self._threads):
                    raise RuntimeError('Guidance transport thread remains live; descendant cleanup NOT VERIFIED.')
                if os.name != 'nt':
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError('Former guidance process group remains present without signal authority.')
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
                self._process = None
                self._threads = []
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.cleanup_error = f'{type(exc).__name__}: {exc}'[:2048]
                raise RuntimeError('Guidance cleanup NOT VERIFIED: ' + self.cleanup_error) from exc

    def close(self) -> None:
        with self._state:
            self.closed = True
            self.generation += 1
        self._retire()


def _serve(config=None, checkpoint=None, *, provider=None) -> None:
    # Keep an exclusive copy of the original protocol FD; all Python/native
    # stdout from imported providers goes to the bounded parent stderr channel.
    with os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        lease = ExclusiveResourceLease('cpu', 'isolated-guidance-child', lambda: True)
        if provider is None:
            if config is None or checkpoint is None:
                raise ValueError('Explicit local provider configuration/checkpoint paths are required.')
            selected = LocalSAM2Provider(config, checkpoint, device='cpu', precision='float32', lease=lease)
        else:
            selected = provider
        session = GuidanceSession(selected, lease)
        protocol.write(_encode({'type': 'ready', 'version': 1, 'device': 'cpu', 'precision': 'float32'}))
        try:
            while True:
                try:
                    message = _read(sys.stdin.buffer)
                except EOFError:
                    break
                if (set(message) != {'version', 'id', 'generation', 'action', 'payload'} or type(message['version']) is not int or message['version'] != 1
                        or type(message['id']) is not int or message['id'] < 1 or type(message['generation']) is not int
                        or message['generation'] < 0 or message['action'] not in ('predict', 'clear')):
                    raise ValueError('Malformed guidance request schema.')
                try:
                    payload = message['payload']
                    if message['action'] == 'clear':
                        if payload != {}:
                            raise ValueError('Clear requires an empty payload.')
                        session.clear()
                        result = {'cleared': True}
                    else:
                        if not isinstance(payload, dict) or set(payload) != {'frame', 'clicks'}:
                            raise ValueError('Malformed guidance frame request.')
                        fields = payload['frame']
                        if not isinstance(fields, dict) or set(fields) != set(GuidanceFrame.__dataclass_fields__):
                            raise ValueError('Malformed guidance frame fields.')
                        fields = {**fields, 'rgb_bytes': _bytes(fields['rgb_bytes'], MAX_PIXELS * 3)}
                        frame = GuidanceFrame(**fields)
                        frame.validate(MAX_PIXELS)
                        clicks = payload['clicks']
                        if not isinstance(clicks, list) or not 1 <= len(clicks) <= 256:
                            raise ValueError('Malformed guidance clicks.')
                        if any(not isinstance(c, dict) or set(c) != {'x', 'y', 'positive'} for c in clicks):
                            raise ValueError('Malformed guidance click fields.')
                        prompts = tuple(Click(**c) for c in clicks)
                        if session._frame is None or session._frame.identity != frame.identity:
                            session.bind(frame)
                        predicted = session.predict(session.request(prompts), control=JobControl())
                        result = asdict(predicted)
                        result['mask_bytes'] = base64.b64encode(predicted.mask_bytes).decode('ascii')
                    response = {'version': 1, 'id': message['id'], 'generation': message['generation'], 'type': 'result', 'payload': result}
                except Exception as exc:
                    logging.getLogger(__name__).exception('Owned guidance child operation failed')
                    response = {'version': 1, 'id': message['id'], 'generation': message['generation'],
                                'type': 'unavailable' if isinstance(exc, GuidanceUnavailable) else 'error',
                                'payload': f'{type(exc).__name__}: {exc}'[:4096]}
                protocol.write(_encode(response))
        finally:
            session.close()


def main():
    parser = argparse.ArgumentParser(description='Owned local CPU subject-guidance child; no downloads.')
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    _serve(args.config, args.checkpoint)


if __name__ == '__main__':
    main()
