"""Seal actual completed linear EXR foreground/alpha for the native reader."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import ofx_record
from .control import JobControl
from .evaluation_media import read_alpha_frame
from .identities import file_sha256
from .local_json import read_object
from .recovery_journal import export_paths
from .source_witness import load_witness
from .video_io import read_image_rgb

PREFIX = struct.Struct('<8sIIIqd32s')
MAGIC = b'CKOFX001'
MAX_FRAME_BYTES = ofx_record.MAX_FRAME_BYTES
MAX_FRAMES = ofx_record.MAX_FRAMES


def seal(record_path: Path, parent: Path, *, control: JobControl | None = None) -> Path:
    control = control if control is not None else JobControl()
    control.checkpoint('ofx_validate')
    record, raw = read_object(record_path, max_bytes=4 * 1024 * 1024)
    spec = ofx_record.cache_spec(record)
    paths = export_paths(record, check=control.check)
    diagnostics = record['diagnostics']
    outputs = diagnostics['outputs']
    width, height, start, count, fps = spec.width, spec.height, spec.start, spec.count, spec.fps
    binding = load_witness(record, width=width, height=height, count=count, start=start, fps=fps)
    if not parent.is_dir():
        raise ValueError('Choose an existing cache output parent.')
    def hashes(items, stage):
        result = {}
        for index, (name, path) in enumerate(items.items()):
            control.checkpoint(stage, index, len(items))
            result[name] = file_sha256(path, check=control.check)
        return result

    original = hashes(paths, 'ofx_hash_exports')
    indexed = {kind: {int(p.stem.rsplit('_', 1)[1]): p for p in paths.values()
                      if p.parent.resolve() == Path(outputs[kind]).resolve()} for kind in ('fg', 'alpha')}
    identity = hashlib.sha256(raw).digest()
    prefix = PREFIX.pack(b'CKOFX002' if binding is not None else MAGIC, width, height, count, start, fps, identity)
    directory = Path(tempfile.mkdtemp(prefix='ofx-cache-', dir=parent))
    draft = spec.draft
    staged = directory / '.cache-stage'
    source_rgb = record['diagnostics']['output_contracts']['fg'].get('foreground_mode') == 'source_rgb_draft'
    target = directory / ('frames-source-rgb-draft.ckcache' if source_rgb else
                          'frames-spatial-draft.ckcache' if draft else 'frames.ckcache')
    tables = []
    published = False
    try:
        with staged.open('xb') as stream:
            stream.write(prefix + bytes(32 + count * (64 if binding is not None else 32)))
            for frame in range(count):
                control.checkpoint('ofx_convert_frames', frame, count)
                fg = read_image_rgb(str(indexed['fg'][frame]), expected_channels=3)
                alpha = read_alpha_frame(indexed['alpha'][frame], channel='gray', check=control.check)[0][..., None]
                if fg.shape != (height, width, 3) or alpha.shape != (height, width, 1):
                    raise ValueError('Actual export geometry differs from completed metadata.')
                if not np.isfinite(fg).all() or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
                    raise ValueError('Invalid foreground or alpha coverage.')
                payload = np.ascontiguousarray(np.concatenate((fg, alpha), axis=2), dtype='<f4').tobytes()
                tables.append(hashlib.sha256(identity + struct.pack('<I', frame) + payload).digest())
                if binding is not None:
                    tables.append(binding[0][frame * 32:(frame + 1) * 32])
                stream.write(payload)
            table = b''.join(tables)
            stream.seek(PREFIX.size)
            stream.write(hashlib.sha256(prefix + table).digest() + table)
            stream.flush()
            os.fsync(stream.fileno())
        if original != hashes(export_paths(record, check=control.check), 'ofx_verify_exports'):
            raise ValueError('Completed export bytes changed while sealing.')
        if binding is not None and load_witness(record, width=width, height=height, count=count, start=start, fps=fps) != binding:
            raise ValueError('Producer source witness changed while sealing.')
        with (directory / 'record.json').open('xb') as stream:
            stream.write(raw)
        control.checkpoint('ofx_publish', count, count)
        os.link(staged, target)
        published = True
    finally:
        primary = sys.exception()
        try:
            staged.unlink(missing_ok=True)
        except OSError as cleanup:
            if primary is not None:
                primary.add_note(f'Owned stage cleanup failed: {cleanup}')
            elif not published:
                raise
            logging.getLogger(__name__).warning('Owned cache stage retained at %s after cleanup failure', staged, exc_info=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--output-parent', required=True, type=Path)
    args = parser.parse_args(argv)
    path = seal(args.record.resolve(), args.output_parent.resolve())
    print(json.dumps({'cache': str(path), 'scope': 'Completed export bytes; host/source/model equivalence and Resolve acceptance NOT VERIFIED.'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
