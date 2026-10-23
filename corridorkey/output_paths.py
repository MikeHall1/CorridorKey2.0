"""Shared deterministic output layout and read-only input-collision admission.

This guard covers planned writes and existing direct EXR leaves. It is a
cooperating-process preflight, not a filesystem reservation or atomic sandbox.
"""
from __future__ import annotations

import os
import re
import stat
import unicodedata
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


def prepare_report_destination(path, jobs, *, manifest_path=None, weights=None,
                                package_root=None, asset_root=None, check=None,
                                protected_paths=(), protected_directories=()):
    """Admit one report without overwriting job inputs, assets or generated media.

    Missing declared paths remain protected. The returned callable rechecks the
    destination without rescanning unchanged media. This cooperating-process
    guard is not a hostile-filesystem reservation.
    """
    from .corrections import CorrectionDocument
    from .local_json import read_object
    from .runtime_paths import RuntimePaths
    from .session import JobRequest

    jobs = tuple(jobs)
    protected_paths, protected_directories = tuple(protected_paths), tuple(protected_directories)
    check = check or (lambda: None)
    def path_key(value):
        # Conservative across native case/normalization rules, including names
        # that do not exist yet and therefore cannot be compared by inode.
        return tuple(unicodedata.normalize('NFC', part).casefold() for part in value.parts)

    target = _lexical(path)
    target_info = _info(target)
    if target_info is not None and (_is_alias(target_info) or not stat.S_ISREG(target_info.st_mode)):
        raise ValueError('report destination must be a regular file, not an alias or special entry')
    files, directories, inodes, directory_ids = set(), set(), set(), set()
    count = 0

    def charge():
        nonlocal count
        check()
        count += 1
        if count > MAX_SCAN_ENTRIES:
            raise ValueError('report protection inventory entry limit exceeded')

    def protect(value, *, directory=False, recursive=False):
        charge()
        lexical = _lexical(value)
        resolved = lexical.resolve()
        lexical_info = _info(lexical)
        if recursive and lexical_info is not None and _is_alias(lexical_info):
            raise ValueError('model report protection cannot inventory a linked/reparse root')
        info = _info(resolved)
        directory = directory or (info is not None and stat.S_ISDIR(info.st_mode))
        if directory:
            directories.update((lexical, resolved))
            if info is None:
                return
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError('protected report directory has an incompatible filesystem type')
            directory_ids.add((info.st_dev, info.st_ino))
            with os.scandir(resolved) as entries:
                for entry in entries:
                    check()
                    entry_info = entry.stat(follow_symlinks=False)
                    if recursive and _is_alias(entry_info):
                        raise ValueError('model report protection cannot inventory linked/reparse children')
                    if entry.is_file(follow_symlinks=True):
                        protect(entry.path)
                    elif recursive and entry.is_dir(follow_symlinks=False):
                        protect(entry.path, directory=True, recursive=True)
                    elif recursive:
                        raise ValueError('model report protection cannot inventory a special entry')
        else:
            files.update((lexical, resolved))
            if info is not None:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('protected report input must be a regular file or directory')
                inodes.add((info.st_dev, info.st_ino))

    if manifest_path is not None:
        protect(manifest_path)
    for value in protected_paths:
        protect(value)
    for value in protected_directories:
        protect(value, directory=True, recursive=True)
    for job in jobs:
        charge()
        if type(job) is not JobRequest:
            raise TypeError('report admission requires immutable JobRequest instances')
        config, _ = job.configurations()
        protect(job.input_path)
        for hint in (job.initial_hint_path, job.hint_video_path):
            if hint is not None:
                protect(hint)
        if config.corrections_path is not None:
            protect(config.corrections_path)
            correction = Path(config.corrections_path).resolve()
            data, _ = read_object(correction, max_bytes=MAX_CORRECTION_BYTES, regular_file=True)
            document = CorrectionDocument.parse(data, correction.parent)
            for asset in document.asset_paths():
                protect(asset)
        layout = output_layout(job.input_path, config)
        for name in layout['metadata']:
            protect(Path(job.output_dir) / name)
        for planned in layout['outputs'].values():
            protect(Path(job.output_dir) / planned['path'], directory=planned['is_directory'])
        # Witness names are generated per render and later consumed by sealing.
        output = _lexical(job.output_dir)
        witness_name = re.compile(r'source-witness-[0-9a-f]{32}\.bin\Z', re.IGNORECASE)
        if path_key(target.parent) == path_key(output) and witness_name.fullmatch(target.name):
            protect(target)
        if output.is_dir():
            with os.scandir(output) as entries:
                for entry in entries:
                    charge()
                    if witness_name.fullmatch(entry.name):
                        protect(entry.path)
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    assets = RuntimePaths.from_package(root, weights, asset_root=asset_root)
    for value in (assets.weights, assets.foundation_stats, assets.rvm_ckpt, assets.moge_checkpoint):
        protect(value)
    protect(assets.cradio_dir, directory=True, recursive=True)
    protected_keys = {path_key(value) for value in files | directories}

    def admit(*, refresh_hardlinks=True):
        resolved_target = target.resolve()
        target_info = _info(target)
        if target_info is not None and (_is_alias(target_info) or not stat.S_ISREG(target_info.st_mode)):
            raise ValueError('report destination must be a regular file, not an alias or special entry')
        if any(key[:len(protected)] == protected
               for key in (path_key(target), path_key(resolved_target)) for protected in protected_keys):
            raise ValueError('report destination overlaps protected inputs, assets or generated outputs')
        for depth, ancestor in enumerate(chain((target,), target.parents)):
            check()
            if depth >= MAX_PATH_ANCESTORS:
                raise ValueError('report ancestry inspection limit exceeded')
            info = _info(ancestor.resolve())
            if info is not None and stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) in directory_ids:
                raise ValueError('report ancestry overlaps a protected directory')
        if target_info is not None:
            if (target_info.st_dev, target_info.st_ino) in inodes:
                raise ValueError('report destination is a hard-link alias of protected data')
            if refresh_hardlinks and target_info.st_nlink > 1:
                # Only a multiply-linked destination needs new export inode discovery.
                fresh = prepare_report_destination(
                    target, jobs, manifest_path=manifest_path, weights=weights,
                    package_root=package_root, asset_root=asset_root, check=check,
                    protected_paths=protected_paths, protected_directories=protected_directories)
                return fresh(refresh_hardlinks=False)
        check()
        return target

    admit(refresh_hardlinks=False)
    return admit


def validate_report_destination(path, jobs, **kwargs):
    """One-shot admission; repeated publishers retain prepare_report_destination()."""
    return prepare_report_destination(path, jobs, **kwargs)(refresh_hardlinks=False)


def validate_batch_output_paths(jobs, *, weights=None, package_root=None, asset_root=None,
                                manifest_path=None, check=None, protected_paths=(), protected_directories=()):
    """Protect the entire admitted batch before any job can mutate its outputs."""
    from .corrections import CorrectionDocument
    from .local_json import read_object
    from .runtime_paths import RuntimePaths

    check = check or (lambda: None)
    jobs = tuple(jobs)
    protected = list(protected_paths)
    for job in jobs:
        check()
        config, _ = job.configurations()
        protected.append(job.input_path)
        protected.extend(hint for hint in (job.initial_hint_path, job.hint_video_path) if hint is not None)
        if config.corrections_path is not None:
            correction = Path(config.corrections_path).resolve()
            protected.append(correction)
            data, _ = read_object(correction, max_bytes=MAX_CORRECTION_BYTES, regular_file=True)
            protected.extend(CorrectionDocument.parse(data, correction.parent).asset_paths())
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    assets = RuntimePaths.from_package(root, weights, asset_root=asset_root)
    protected.extend((assets.weights, assets.foundation_stats, assets.rvm_ckpt, assets.moge_checkpoint))
    if manifest_path is not None:
        protected.append(manifest_path)
    pending = [assets.cradio_dir, *(Path(value) for value in protected_directories)]
    model_directories = []
    visited = set()
    inspected = 0
    while pending:
        check()
        directory = pending.pop()
        resolved = directory.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        model_directories.append(directory)
        info = _info(directory)
        if info is None:
            continue
        if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError('model output protection requires real directories')
        with os.scandir(directory) as entries:
            for entry in entries:
                check()
                inspected += 1
                if inspected > MAX_SCAN_ENTRIES:
                    raise ValueError('model output protection inventory entry limit exceeded')
                if _is_alias(entry.stat(follow_symlinks=False)):
                    raise ValueError('model output protection cannot inventory linked/reparse children')
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                else:
                    protected.append(entry.path)
    for job in jobs:
        config, _ = job.configurations()
        validate_output_paths(job.input_path, job.output_dir, config,
                              initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
                              protected_paths=protected, protected_directories=model_directories, check=check)


def validate_output_paths(input_path, output_dir, cfg, *, initial_hint_path=None,
                          hint_video_path=None, correction_document=None, check=None,
                          protected_paths=(), protected_directories=()):
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
    protected = [Path(input_path), *(Path(value) for value in protected_paths)]
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
        info = _info(resolved)
        if info is None:
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('protected input must be a regular file or frame directory')
        inodes.add((info.st_dev, info.st_ino))

    for value in protected_directories:
        path = _lexical(value)
        directories.update((path, path.resolve()))
        info = _info(path.resolve())
        if info is not None:
            directory_ids.add((info.st_dev, info.st_ino))

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
        if not directory.exists():
            continue
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
