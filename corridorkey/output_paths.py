"""Shared deterministic output layout and read-only input-collision admission.

This guard covers planned writes and existing direct EXR leaves. It is a
cooperating-process preflight, not a filesystem reservation or atomic sandbox.
"""
from __future__ import annotations

import os
import re
import stat
from itertools import chain
from pathlib import Path
from typing import Any

MAX_PATH_ANCESTORS = 1024
MAX_SCAN_ENTRIES = 100000
MAX_CORRECTION_BYTES = 4 * 1024**2


def raw_clip_name(path):
    normalized = os.path.normpath(str(path))
    name = os.path.splitext(os.path.basename(normalized))[0]
    if name.lower() in {'input', 'frames', 'rgb'}:
        return os.path.basename(os.path.dirname(normalized)) or name
    return name


def output_layout(input_path, cfg) -> dict[str, Any]:
    """The names consumed by both admission and the actual runtime writers."""
    clip = raw_clip_name(input_path)
    quick = cfg.quick_key and cfg.quick_interval > 1
    suffixes = {'alpha': 'alpha', 'fg': 'fg', 'cutout': 'cutout',
                'checker': f'checker_comp_despill{int(cfg.despill_strength * 100):03d}'}
    if cfg.cutout_alpha_mode == 'premultiplied':
        suffixes['cutout'] += '_premultiplied'
    if quick:
        suffixes = {role: suffix + '_source_rgb_draft' for role, suffix in suffixes.items()}
    if cfg.model_scale < 1:
        suffixes = {role: suffix + '_spatial_draft' for role, suffix in suffixes.items()}
    if cfg.output_format not in {'exr', 'video'}:
        raise ValueError('unsupported output format')
    outputs = {}
    for role in sorted(cfg.outputs):
        if role not in suffixes:
            raise ValueError('unsupported output role')
        directory = cfg.output_format == 'exr'
        path = (f'{clip}_{suffixes[role]}_exr' if directory else
                f'{clip}_{suffixes[role]}' + ('.mov' if role == 'cutout' else '.mp4'))
        outputs[role] = {'path': path, 'frame_name': suffixes[role], 'is_directory': directory}
    metadata = [f'{clip}_settings.txt']
    if quick:
        metadata.append(f'{clip}_quick_key_frames.jsonl')
        if cfg.quick_auto_cuts:
            metadata.append(f'{clip}_shot_analysis.jsonl')
    return {'clip_name': clip, 'outputs': outputs, 'metadata': tuple(metadata)}


def _within(path, parent):
    return path == parent or parent in path.parents


def _lexical(path):
    return Path(os.path.abspath(os.fspath(path)))


def _info(path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _is_alias(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def validate_output_paths(input_path, output_dir, cfg, *, initial_hint_path=None,
                          hint_video_path=None, correction_document=None, check=None):
    """Refuse output writes that overlap source, hints or authored assets.

    No source decoding, recursive traversal, output creation or model imports.
    Existing unrelated outputs may be overwritten by the ordinary rerender path.
    """
    check = check or (lambda: None)
    check()
    layout = output_layout(input_path, cfg)
    output = _lexical(output_dir)
    resolved_output = output.resolve()
    existing = _info(output)
    if existing is not None and (_is_alias(existing) or not stat.S_ISDIR(existing.st_mode)):
        raise ValueError('output root must be a real directory, not a link/reparse or file')
    protected = [Path(input_path)]
    for value in (initial_hint_path, hint_video_path):
        if value is not None and str(value).strip().lower() not in {'', 'none', 'null', 'off', 'false', '-'}:
            protected.append(Path(value))
    correction_path = getattr(cfg, 'corrections_path', None)
    if correction_path is not None:
        correction_path = Path(correction_path)
        protected.append(correction_path)
        if correction_document is None:
            # Reuse the strict bounded stdlib reader and schema parser, without
            # asset hashes/decodes or a new source-content scan.
            from .corrections import CorrectionDocument
            from .local_json import read_object

            data, _ = read_object(correction_path.resolve(), max_bytes=MAX_CORRECTION_BYTES,
                                  regular_file=True)
            correction_document = CorrectionDocument.parse(data, correction_path.resolve().parent)
    if correction_document is not None:
        asset_paths = correction_document.asset_paths()
        if len(asset_paths) > MAX_SCAN_ENTRIES:
            raise ValueError('correction asset inventory exceeds output admission bounds')
        for asset_path in asset_paths:
            check()
            protected.append(asset_path)

    directories = set()
    directory_ids = set()
    files = set()
    inodes = set()
    count = 0

    def charge():
        nonlocal count
        check()
        count += 1
        if count > MAX_SCAN_ENTRIES:
            raise ValueError('output collision inspection entry limit exceeded')

    def protect_file(path):
        lexical, resolved = _lexical(path), path.resolve()
        files.update((lexical, resolved))
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('protected input must be a regular file or frame directory')
        inodes.add((info.st_dev, info.st_ino))

    for path in protected:
        check()
        resolved = path.resolve()
        if resolved.is_dir():
            directories.update((_lexical(path), resolved))
            info = resolved.stat()
            directory_ids.add((info.st_dev, info.st_ino))
        else:
            protect_file(path)
    def admit_ancestry(path):
        # Inode comparison closes spelling/case aliases that a lexical Path
        # comparison need not normalize on case-insensitive filesystems.
        for depth, ancestor in enumerate(chain((path,), path.parents)):
            check()
            if depth >= MAX_PATH_ANCESTORS:
                raise ValueError('output ancestry inspection limit exceeded')
            try:
                info = ancestor.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) in directory_ids:
                raise ValueError('output ancestry is a protected source/hint directory')

    if any(_within(output, directory) or _within(resolved_output, directory) for directory in directories):
        raise ValueError('output root would modify a protected source/hint directory')
    admit_ancestry(output)
    # Readers consume direct media children. Inode inspection also protects
    # differently named hard links in a planned output leaf elsewhere.
    for directory in sorted({path.resolve() for path in directories}):
        check()
        with os.scandir(directory) as children:
            for child in children:
                charge()
                info = child.stat(follow_symlinks=True)
                if stat.S_ISREG(info.st_mode):
                    files.update((_lexical(child.path), Path(child.path).resolve()))
                    inodes.add((info.st_dev, info.st_ino))

    def admit_target(path, *, directory=False):
        check()
        resolved = path.resolve()
        if (path in files or resolved in files
                or any(_within(path, parent) or _within(resolved, parent) for parent in directories)):
            raise ValueError(f'generated output would overwrite a protected input: {path}')
        admit_ancestry(path)
        info = _info(path)
        if info is None:
            return
        if _is_alias(info):
            raise ValueError(f'generated output link/reparse target is refused: {path}')
        if directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError('generated EXR directory is occupied by a non-directory')
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError('generated output file is occupied by a nonregular entry')
        if (info.st_dev, info.st_ino) in inodes:
            raise ValueError(f'generated output is a hard-link alias of a protected input: {path}')

    for name in layout['metadata']:
        admit_target(output / name)
    for planned in layout['outputs'].values():
        target = output / planned['path']
        admit_target(target, directory=planned['is_directory'])
        if planned['is_directory'] and target.is_dir():
            pattern = re.compile(re.escape(planned['frame_name']) + r'_[0-9]{6,}\.exr\Z', re.IGNORECASE)
            with os.scandir(target) as children:
                for child in children:
                    charge()
                    if pattern.fullmatch(child.name):
                        admit_target(Path(child.path))
    check()
