import ast
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.desktop_bundle import ROOT, build, data_files, preflight, spec_text
from corridorkey.recovery_identity import canonical, implementation_identity


class DesktopBundleTests(unittest.TestCase):
    def test_recipe_collects_executing_engine_packages_as_source_only(self):
        tree = ast.parse(spec_text(ROOT))
        analysis = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'Analysis')
        options = {keyword.arg: keyword.value for keyword in analysis.keywords}
        self.assertIn('module_collection_mode', options)
        self.assertEqual(ast.literal_eval(options['module_collection_mode']),
                         {'corridorkey': 'py', 'moge': 'py'})

    def test_source_only_bundle_identity_covers_both_moge_locations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = ('corridorkey/__init__.py', 'corridorkey/runtime.py',
                       'moge/__init__.py', 'moge/model.py',
                       'third_party/MoGe/moge/__init__.py', 'third_party/MoGe/moge/model.py')
            for name in sources:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# bundled source fixture\n')
            initial = implementation_identity(root)
            self.assertEqual({row['name'] for row in initial['files']}, set(sources))
            for name in ('moge/model.py', 'third_party/MoGe/moge/model.py', 'corridorkey/runtime.py'):
                path = root / name
                before = path.read_text()
                path.write_text(before + '# changed executing implementation\n')
                self.assertNotEqual(implementation_identity(root)['sha256'], initial['sha256'])
                path.write_text(before)
            (root / 'corridorkey/runtime.py').unlink()
            with self.assertRaisesRegex(ValueError, 'shared runtime source'):
                implementation_identity(root)

    def test_development_identity_without_collected_moge_keeps_original_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = ('corridorkey/runtime.py', 'third_party/MoGe/moge/model.py')
            records = []
            for name in sources:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'# existing development source\n')
                records.append({'name': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            expected = hashlib.sha256(canonical(records).encode()).hexdigest()
            self.assertEqual(implementation_identity(root), {'files': records, 'sha256': expected})

    def test_windows_target_rejects_cpu_torch_without_claiming_gpu_inference(self):
        versions = {'PyInstaller': '6.22.3', 'PySide6-Essentials': '6.10.2', 'shiboken6': '6.10.2'}
        with patch('sys.platform', 'win32'), patch('platform.machine', return_value='AMD64'), patch(
                'importlib.metadata.version', side_effect=lambda name: versions.get(name, '1.0')), patch(
                'corridorkey.desktop_bundle.inspect_backend', return_value={'cuda_build': None, 'cuda_available': False}):
            report = preflight()
            self.assertFalse(report['ready'])
            self.assertTrue(any('CUDA-enabled' in error for error in report['errors']))

    def test_recipe_keeps_notices_and_excludes_models_and_media(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vendored = root / 'third_party/MoGe'
            vendored.mkdir(parents=True)
            for name in ('LICENSE', 'model.py', 'config.json', 'private.pt', 'private.exr'):
                (vendored / name).write_text('fixture')
            selected = {Path(source).name for source, _ in data_files(root)}
            self.assertIn('THIRD_PARTY_NOTICES.md', selected)
            self.assertIn('LICENSE', selected)
            self.assertIn('model.py', selected)
            self.assertNotIn('private.pt', selected)
            self.assertNotIn('private.exr', selected)
            compile(spec_text(root), 'CorridorKey.spec', 'exec')

    def test_wrong_platform_check_creates_nothing_and_does_not_install(self):
        with tempfile.TemporaryDirectory() as folder, patch('sys.platform', 'darwin'):
            before = list(Path(folder).iterdir())
            self.assertFalse(preflight()['ready'])
            with patch('subprocess.run') as process, self.assertRaises(ValueError):
                build(Path(folder))
            process.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), before)

    def test_failed_builder_preserves_report_and_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            important = parent / 'important.txt'
            important.write_text('preserve')
            with patch('corridorkey.desktop_bundle.preflight', return_value={'ready': True, 'errors': []}), patch(
                    'subprocess.run', side_effect=[subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 7)]), patch(
                    'corridorkey.benchmark.repository_identity', return_value={'commit': 'explicit-test-fixture'}), self.assertRaises(RuntimeError):
                build(parent)
            report = json.loads(next(parent.glob('desktop-build-*/build-report.json')).read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['returncode'], 7)
            self.assertEqual(report['repository']['commit'], 'explicit-test-fixture')
            self.assertEqual(important.read_text(), 'preserve')
            self.assertIn(str(ROOT / 'bundle_entry.py'), next(parent.glob('desktop-build-*/*.spec')).read_text())
