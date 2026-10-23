"""Local Windows build recipe; never installs dependencies or bundles weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .asset_inventory import inspect_backend

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('torch', 'torchvision', 'numpy', 'opencv-python', 'transformers', 'timm',
            'safetensors', 'huggingface-hub', 'einops', 'scipy', 'pillow', 'tqdm',
            'utils3d', 'PySide6-Essentials', 'shiboken6', 'PyInstaller')


def preflight() -> dict[str, Any]:
    errors, versions = [], {}
    backend = None
    if sys.platform != 'win32' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        errors.append('Build Windows x86_64 on Windows x86_64; cross compilation is unsupported.')
    if sys.version_info[:2] != (3, 12):
        errors.append('Use the verified Python3.12 build environment.')
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f'Missing build dependency: {package}')
    for name, version in {'PyInstaller': '6.22.3', 'PySide6-Essentials': '6.10.2', 'shiboken6': '6.10.2'}.items():
        if name in versions and versions[name] != version:
            errors.append(f'{name} must match recipe pin {version}.')
    if sys.platform == 'win32' and 'torch' in versions:
        backend = inspect_backend()
        if not backend.get('cuda_build'):
            errors.append('Windows/NVIDIA bundle requires a CUDA-enabled Torch build; CPU-only or unverified builds are unsupported.')
    return {'ready': not errors, 'errors': errors, 'versions': versions,
            'backend_build': backend,
            'platform': platform.platform(), 'python': sys.version,
            'scope': 'Build prerequisites only; CUDA, model compatibility, licensing/distribution and native application acceptance NOT VERIFIED.'}


def data_files(root: Path) -> list[tuple[str, str]]:
    files = [(str(root / name), '.') for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt',
                                                                  'requirements-desktop.txt', 'requirements-build.txt')]
    files += [(str(root / 'third_party/OpenFX' / name), 'third_party/OpenFX')
              for name in ('LICENSE.md', 'provenance.json')]
    # Enumerated text/code only: never sweep model checkpoints, source footage or user outputs.
    for path in sorted((root / 'third_party/MoGe').rglob('*')):
        if path.is_file() and (path.suffix.lower() in {'.py', '.json', '.yaml', '.yml', '.txt', '.md'} or path.name == 'LICENSE'):
            files.append((str(path), str(path.parent.relative_to(root))))
    return files


def spec_text(root: Path) -> str:
    return f'''from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata
datas = {data_files(root)!r}
hidden = collect_submodules('corridorkey') + collect_submodules('moge')
for package in ('transformers', 'timm'):
    datas += collect_data_files(package)
for package in {PACKAGES!r}:
    datas += copy_metadata(package)
a = Analysis([{str(root / 'bundle_entry.py')!r}], pathex={[str(root), str(root / 'third_party/MoGe')]!r},
             binaries=[], datas=datas, hiddenimports=hidden, hookspath=[], runtime_hooks=[],
             module_collection_mode={{'corridorkey': 'py', 'moge': 'py'}},
             excludes=['tests', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='CorridorKey',
          debug=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='CorridorKey')
'''


def build(parent: Path) -> Path:
    report = preflight()
    if not report['ready']:
        raise ValueError('; '.join(report['errors']))
    if not parent.is_dir():
        raise ValueError('Choose an existing build output parent.')
    # Source gate must pass before build artifacts are created. No install/download.
    subprocess.run([sys.executable, str(ROOT / 'scripts/ralph_loop.py'), 'verify', '--python', sys.executable],
                   cwd=ROOT, check=True)
    from .benchmark import repository_identity

    report['repository'] = repository_identity()
    directory = Path(tempfile.mkdtemp(prefix='desktop-build-', dir=parent))
    spec = directory / 'CorridorKey.spec'
    spec.write_text(spec_text(ROOT), encoding='utf-8')
    command = [sys.executable, '-m', 'PyInstaller', '--distpath', str(directory / 'dist'),
               '--workpath', str(directory / 'work'), str(spec)]
    report.update(command=command, bundled_source_notices=[{'source': source, 'destination': target,
                  'sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()} for source, target in data_files(ROOT)],
                  status='building', assets='External; no model weights or FFmpeg binary included.')
    manifest = directory / 'build-report.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'third_party/MoGe'), environment.get('PYTHONPATH', '')])
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        report.update(returncode=result.returncode, status='built-unvalidated' if result.returncode == 0 else 'failed')
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Bundle build failed; inspect {directory}')
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'build'))
    parser.add_argument('--output-parent', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'check':
        report = preflight()
        print(json.dumps(report, indent=2))
        return 0 if report['ready'] else 2
    if args.output_parent is None:
        parser.error('build requires --output-parent')
    print(build(args.output_parent.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
