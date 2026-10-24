"""Report every static finding and fail on regressions against a reviewed baseline.

A successful comparison is NOT a clean static-analysis result. Baselines are
reviewed artifacts; this command never creates or updates one automatically.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'tests/static-baseline.json'
TOOLS = ('ruff', 'pyright')
IMPORTS = ('numpy', 'cv2', 'torch', 'torchvision', 'transformers', 'timm',
           'safetensors', 'huggingface_hub', 'einops', 'scipy', 'PIL', 'tqdm',
           'utils3d', 'corridorkey.runtime', 'corridorkey.video_io',
           'corridorkey.refinement', 'corridorkey.session')


def source_files():
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT, encoding='utf-8', errors='strict').split('\0')
    return sorted({name for name in names
                   if name.endswith('.py') and (ROOT / name).is_file()})


def fingerprints(tool, diagnostics):
    """Count identical diagnostics, ignoring positions but retaining message/severity."""
    result = Counter()
    for item in diagnostics:
        file = item['filename'] if tool == 'ruff' else item['file']
        path = Path(file).resolve().relative_to(ROOT).as_posix()
        rule = item['code'] if tool == 'ruff' else item.get('rule', '')
        payload = [path, rule, item.get('severity', 'error'), item['message']]
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=True).encode()).hexdigest()
        result[key] += 1
    return dict(result)


def compare(current, baseline):
    added = Counter(current) - Counter(baseline)
    removed = Counter(baseline) - Counter(current)
    return {'new': dict(added), 'resolved': dict(removed),
            'new_count': sum(added.values()), 'resolved_count': sum(removed.values())}


def dependencies():
    # packaging is part of the declared runtime stack via transformers/timm.
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError as exc:
        return {'status': 'NOT VERIFIED', 'errors': [str(exc)]}
    installed = {canonicalize_name(d.metadata['Name']): d
                 for d in importlib.metadata.distributions()}
    errors = []
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8', errors='strict').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        dist = installed.get(canonicalize_name(req.name))
        if dist is None or not req.specifier.contains(dist.version):
            errors.append(f'{req}: installed {dist.version if dist else "MISSING"}')
        elif req.url:
            direct = json.loads(dist.read_text('direct_url.json') or '{}')
            expected_repo, expected_commit = req.url.removeprefix('git+').rsplit('@', 1)
            if (direct.get('url') != expected_repo or
                    direct.get('vcs_info', {}).get('commit_id') != expected_commit):
                errors.append(f'{req.name}: direct VCS origin/commit not verified')
    for name, dist in installed.items():
        for text in dist.requires or ():
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            dependency = installed.get(canonicalize_name(req.name))
            if dependency is None or not req.specifier.contains(dependency.version):
                errors.append(f'{name} requires {req}; installed '
                              f'{dependency.version if dependency else "MISSING"}')
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
            'packages': {name: dist.version for name, dist in installed.items()}}


def run(output_dir, baseline_path=BASELINE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = source_files()
    syntax_errors = []
    for name in files:
        try:
            ast.parse((ROOT / name).read_bytes(), filename=name)
        except (SyntaxError, UnicodeError) as exc:
            syntax_errors.append(str(exc))
    imports_code = ('import importlib; '
                    f'[importlib.import_module(n) for n in {IMPORTS!r}]; '
                    'print("Selected runtime imports PASS; no model instantiated")')
    subprocess_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    imports = subprocess.run([sys.executable, '-c', imports_code], check=False, cwd=ROOT,
                             capture_output=True, encoding='utf-8', errors='strict',
                             env=subprocess_env, timeout=180)
    (output_dir / 'imports.txt').write_text(imports.stdout + imports.stderr, encoding='utf-8', errors='strict')
    baseline_bytes = Path(baseline_path).read_bytes() if Path(baseline_path).is_file() else None
    baseline = json.loads(baseline_bytes) if baseline_bytes is not None else {}
    checks = {}
    for tool in TOOLS:
        try:
            version = importlib.metadata.version(tool)
        except importlib.metadata.PackageNotFoundError:
            checks[tool] = {'status': 'NOT VERIFIED', 'reason': 'tool not installed'}
            continue
        args = (['check', '--output-format', 'json', *files] if tool == 'ruff' else
                ['--pythonpath', sys.executable, '--pythonversion', '3.12', '--outputjson', *files])
        completed = subprocess.run([sys.executable, '-m', tool, *args], check=False, cwd=ROOT,
                                   capture_output=True, encoding='utf-8', errors='strict',
                                   env=subprocess_env, timeout=180)
        (output_dir / f'{tool}.json').write_text(completed.stdout, encoding='utf-8', errors='strict')
        (output_dir / f'{tool}-stderr.txt').write_text(completed.stderr, encoding='utf-8', errors='strict')
        try:
            raw = json.loads(completed.stdout)
            diagnostics = raw if tool == 'ruff' else raw['generalDiagnostics']
            current = fingerprints(tool, diagnostics)
        except (ValueError, KeyError, TypeError) as exc:
            checks[tool] = {'status': 'NOT VERIFIED', 'reason': str(exc)}
            continue
        previous = baseline.get('tools', {}).get(tool, {})
        comparable = (version == previous.get('version') and
                      baseline.get('python_minor') == list(sys.version_info[:2]))
        delta = compare(current, previous.get('fingerprints', {}))
        checks[tool] = {'status': 'VERIFIED', 'version': version,
                        'returncode': completed.returncode, 'diagnostic_count': len(diagnostics),
                        'static_clean': completed.returncode == 0, 'comparable': comparable,
                        'no_regressions': comparable and delta['new_count'] == 0 and
                                          completed.returncode in (0, 1),
                        'fingerprints': current, **delta}
    dependency_check = dependencies()
    report = {'python': sys.version, 'platform': platform.platform(),
              'executable': sys.executable, 'source_files': len(files),
              'baseline': {'path': str(baseline_path),
                           'sha256': hashlib.sha256(baseline_bytes).hexdigest()
                                     if baseline_bytes is not None else None,
                           'python_minor': baseline.get('python_minor'),
                           'tools': {name: {'version': value.get('version'),
                                            'diagnostic_count': value.get('diagnostic_count')}
                                     for name, value in baseline.get('tools', {}).items()}},
              'syntax_errors': syntax_errors, 'imports_returncode': imports.returncode,
              'dependencies': dependency_check, 'static': checks,
              'static_clean': all(c.get('static_clean', False) for c in checks.values()),
              'note': 'No regression is not a clean static result. Every finding remains in raw reports; '
                      'CPU/static checks do not verify GPU inference, visual quality or Resolve.'}
    report['success'] = (not syntax_errors and imports.returncode == 0 and
                         dependency_check['status'] == 'PASS' and
                         all(c.get('no_regressions', False) for c in checks.values()))
    (output_dir / 'health.json').write_text(json.dumps(report, indent=2) + '\n',
                                             encoding='utf-8', errors='strict')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmark-results/code-health')
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    args = parser.parse_args()
    report = run(args.output_dir, args.baseline)
    print(json.dumps({k: v for k, v in report.items() if k not in ('dependencies', 'static')}, indent=2))
    for tool, check in report['static'].items():
        print(tool, {k: v for k, v in check.items() if k not in ('fingerprints', 'new', 'resolved')})
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
