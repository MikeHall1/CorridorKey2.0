"""Explicit subprocess capability inspection, separate from lightweight checks.

Only the direct owned process is supervised; executable wrappers that launch
independent children are outside this probe's cleanup contract. No models load.
"""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any

from .app_process import module_command
from .control import CancellationToken, CancelledError
from .session import JobRequest

MAX_OUTPUT_BYTES = 256 * 1024


def _capture(command: list[str], token: CancellationToken, *, timeout: float = 20.0) -> str:
    """Bound output before allocation, cooperatively stop/reap the owned child."""
    token.check()
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    assert process.stdout is not None
    output = bytearray()
    try:
        os.set_blocking(process.stdout.fileno(), False)
        while True:
            token.check()
            if time.monotonic() - started > timeout:
                raise TimeoutError('Capability probe exceeded its explicit deadline.')
            try:
                block = os.read(process.stdout.fileno(), min(16384, MAX_OUTPUT_BYTES + 1 - len(output)))
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                block = None
            if block:
                output.extend(block)
                if len(output) > MAX_OUTPUT_BYTES:
                    raise ValueError('Capability probe output exceeds 256 KiB.')
            elif block == b'' and process.poll() is not None:
                break
            else:
                # No Qt or model work; cancellation is checked every 20 ms.
                time.sleep(.02)
        token.check()
        text = output.decode('utf-8', errors='replace')
        if process.returncode:
            raise RuntimeError(f'Capability subprocess exited {process.returncode}: {text[-4096:]}')
        return text
    finally:
        # Popen supervises its own direct process handle; never signal a numeric
        # process group after polling/reaping its leader.
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            process.stdout.close()


def check_capabilities(job: JobRequest, *, cancellation: CancellationToken | None = None) -> dict[str, Any]:
    """Explicit backend availability/encoder-help probes; never a test encode.

    A listed encoder/pixel format does not prove codec option compatibility or
    successful output. Exact settings remain unchanged for the real render.
    """
    token = cancellation if cancellation is not None else CancellationToken()
    config, writer = job.configurations()
    report: dict[str, Any] = {'schema_version': 1, 'status': 'complete', 'stages': [],
                              'scope': 'Explicit owned subprocess metadata probes. No model loading, inference, encode or download.',
                              'requested_backend': 'cuda', 'effective_backend': None,
                              'cleanup_scope': 'Direct subprocess only; independent children from executable wrappers are unsupported.'}
    active_stage = 'backend'
    try:
        token.check()
        try:
            backend = json.loads(_capture(module_command('corridorkey.backend_probe'), token))
            if not isinstance(backend, dict) or type(backend.get('cuda_available')) is not bool:
                raise ValueError('Backend subprocess returned invalid availability metadata.')
            report['stages'].append({'stage': 'backend', 'status': 'PASS' if backend['cuda_available'] else 'FAIL',
                                     'details': backend, 'limits': 'Availability only; model compatibility and actual inference NOT VERIFIED.'})
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, CancelledError):
                raise
            report['stages'].append({'stage': 'backend', 'status': 'FAIL', 'details': str(exc)[:4096]})
        token.check()
        active_stage = 'ffmpeg_capabilities'
        if config.output_format == 'exr':
            report['stages'].append({'stage': 'ffmpeg_capabilities', 'status': 'NOT REQUIRED', 'details': 'EXR output does not invoke FFmpeg.'})
            return report
        executable = shutil.which(writer.ffmpeg_bin)
        if executable is None:
            raise FileNotFoundError(f'Configured FFmpeg executable not found: {writer.ffmpeg_bin}')
        required = set()
        if config.outputs - {'cutout'}:
            required.add((writer.codec, writer.pix_fmt))
        if 'cutout' in config.outputs:
            required.add(('prores_ks', 'yuva444p10le'))
        for codec, pixel_format in sorted(required):
            token.check()
            raw = _capture([executable, '-hide_banner', '-h', f'encoder={codec}'], token)
            headers = list(re.finditer(r'^Encoder ([^\s\[]+)(?:\s|\[)', raw, re.MULTILINE))
            candidates = []
            for index, header in enumerate(headers):
                end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
                match = re.search(r'^[ \t]*Supported pixel formats:[ \t]*([^\r\n]*)',
                                  raw[header.end():end], re.MULTILINE)
                candidates.append({'encoder': header.group(1),
                                   'listed_pixel_formats': match.group(1).split() if match else None})
            matching = [item for item in candidates if item['encoder'] == codec]
            formats = matching[0]['listed_pixel_formats'] if len(matching) == 1 else None
            # Descriptor aliases can list multiple canonical encoders. Help is
            # not proof of which one a real encode will select for the alias.
            status = ('FAIL' if not candidates else 'NOT VERIFIED')
            if len(matching) == 1 and formats:
                status = 'PASS' if pixel_format in formats else 'FAIL'
            report['stages'].append({'stage': 'ffmpeg_capabilities', 'status': status,
                                     'details': {'executable': executable, 'encoder': codec,
                                                 'encoder_listed': bool(matching), 'requested_pixel_format': pixel_format,
                                                 'listed_pixel_formats': formats, 'encoder_candidates': candidates},
                                     'limits': 'Encoder help only. Preset/CRF/bitrate/muxing, hardware initialization and actual encode/decode NOT VERIFIED.'})
    except CancelledError as exc:
        report['status'], report['error'] = 'cancelled', str(exc)[:4096]
        report['interrupted_stage'] = active_stage
    except (OSError, ValueError, RuntimeError) as exc:
        report['stages'].append({'stage': 'ffmpeg_capabilities', 'status': 'FAIL', 'details': str(exc)[:4096]})
    return report
