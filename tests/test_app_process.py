import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from corridorkey.app_process import main, module_command


class ApplicationProcessTests(unittest.TestCase):
    def test_asset_download_catalog_dispatch_is_offline_and_frozen_explicit(self):
        module = 'corridorkey.asset_download'
        self.assertEqual(module_command(module, 'catalog'), [sys.executable, '-m', module, 'catalog'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command(module, 'catalog'),
                             [sys.executable, '--corridorkey-module', module, 'catalog'])
        with tempfile.TemporaryDirectory() as folder:
            before = list(Path(folder).iterdir())
            script = '''import runpy,sys
from unittest.mock import patch
sys.argv=['app_process','--corridorkey-module','corridorkey.asset_download','catalog']
with patch('socket.socket.connect', side_effect=AssertionError('Offline catalog must not connect')):
    runpy.run_module('corridorkey.app_process',run_name='__main__')
'''
            result = subprocess.run([sys.executable, '-c', script], cwd=folder,
                                    env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])},
                                    capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            catalog = json.loads(result.stdout)
            self.assertEqual(len(catalog['publishers']), 4)
            self.assertIn('scope', catalog)
            self.assertEqual(list(Path(folder).iterdir()), before)

    def test_asset_audit_dispatch_and_exact_report_export_preserve_assets(self):
        module = 'corridorkey.asset_audit'
        self.assertEqual(module_command(module, '--help'), [sys.executable, '-m', module, '--help'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command(module, '--help'),
                             [sys.executable, '--corridorkey-module', module, '--help'])
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder).resolve()
            root = parent / 'assets'
            weights = root / 'weights' / 'corridorkey_v2.safetensors'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'deliberately malformed fixture, never deserialize')
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            output = parent / 'audit.json'
            result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process',
                                     '--corridorkey-module', module, 'audit', '--asset-root', str(root),
                                     '--output', str(output)], capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report['kind'], 'corridorkey_asset_audit')
            self.assertEqual(report['model_compatibility'], 'NOT VERIFIED')
            self.assertFalse(report['assets_present'])
            entry = next(row for row in report['entries'] if row['asset'] == 'weights')
            self.assertEqual(entry['format_status'], 'FAIL')
            self.assertEqual(json.loads(output.read_text()), report)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes()
                                      for p in root.rglob('*') if p.is_file()})
            original = output.read_bytes()
            repeat = subprocess.run([sys.executable, '-m', 'corridorkey.app_process',
                                     '--corridorkey-module', module, 'audit', '--asset-root', str(root),
                                     '--output', str(output)], capture_output=True, text=True, timeout=20, check=False)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_queue_transfer_dispatch_preserves_history_after_source_move(self):
        from corridorkey.queue_store import QueueStore

        module = 'corridorkey.queue_transfer'
        self.assertEqual(module_command(module, '--help'), [sys.executable, '-m', module, '--help'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command(module, '--help'),
                             [sys.executable, '--corridorkey-module', module, '--help'])
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder).resolve()
            source = parent / 'source.json'
            source.write_text('{}')
            batch = parent / 'batch.json'
            batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': str(source), 'output_dir': str(parent / 'output'),
                 'settings': {'output_format': 'exr'}}]}))
            workspace = QueueStore.create(parent)
            with closing(QueueStore(workspace)) as store:
                store.start_session()
                saved = store.prepare('historical', 'recover', {
                    'batch': str(batch), 'workspace': str(parent / 'cache'), 'max_cache_bytes': 0})
            QueueStore.seal(workspace)
            before = {str(p.relative_to(workspace)): p.read_bytes()
                      for p in workspace.rglob('*') if p.is_file()}

            def command(*args):
                result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process',
                                         '--corridorkey-module', module, *args],
                                        capture_output=True, text=True, timeout=20, check=False)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                data = json.loads(result.stdout)
                self.assertTrue(data['usable'], data)
                return data

            archive = command('export', '--workspace', str(workspace), '--output-parent', str(parent))
            self.assertEqual(before, {str(p.relative_to(workspace)): p.read_bytes()
                                      for p in workspace.rglob('*') if p.is_file()})
            command('inspect', '--archive', archive['destination'])
            moved = workspace.with_name('original-history-moved')
            workspace.rename(moved)
            imported = command('import', '--archive', archive['destination'], '--output-parent', str(parent))
            restored = Path(imported['destination'])
            detail = QueueStore.inspect(restored, request_id='historical')
            self.assertEqual(detail['request']['payload']['batch'], saved['batch'])
            local = Path(detail['load_payload']['batch'])
            self.assertTrue(local.is_relative_to(restored))
            self.assertEqual(local.read_bytes(), (moved / 'batches' / 'historical.json').read_bytes())
            self.assertEqual((restored / 'queue.sqlite').read_bytes(), before['queue.sqlite'])
            with self.assertRaises(ValueError):
                QueueStore(restored)
            self.assertEqual(source.read_text(), '{}')
            self.assertFalse((parent / 'output').exists())

    def test_source_and_frozen_command_contract(self):
        self.assertEqual(module_command('corridorkey.evaluation_dossier', '--help'),
                         [sys.executable, '-m', 'corridorkey.evaluation_dossier', '--help'])
        self.assertEqual(module_command('corridorkey.worker', 'serve'),
                         [sys.executable, '-m', 'corridorkey.worker', 'serve'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command('corridorkey.evaluation_dossier', '--help'),
                             [sys.executable, '--corridorkey-module', 'corridorkey.evaluation_dossier', '--help'])
            self.assertEqual(module_command('corridorkey.worker', 'serve'),
                             [sys.executable, '--corridorkey-module', 'corridorkey.worker', 'serve'])
            with self.assertRaisesRegex(ValueError, 'source checkout'):
                module_command('unittest')
        with self.assertRaises(ValueError):
            module_command('os')

    def test_dispatch_rejects_unknown_modules_without_loading_gui(self):
        with self.assertRaises(SystemExit):
            main(['--corridorkey-module', 'os'])

    def test_real_dispatch_backend_and_worker_help(self):
        for module in ('corridorkey.worker', 'corridorkey.benchmark', 'corridorkey.batch', 'corridorkey.ofx_cache'):
            result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process',
                                     '--corridorkey-module', module, '--help'], capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
        result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process', '--corridorkey-module',
                                 'corridorkey.backend_probe'], capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(type(json.loads(result.stdout)['cuda_available']), bool)

    def test_backend_probe_uses_code_root_after_caller_changes_directory(self):
        from corridorkey.asset_inventory import inspect_backend

        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                result = inspect_backend()
                self.assertIs(type(result['cuda_available']), bool, result)
                self.assertTrue(Path.cwd().samefile(folder))
            finally:
                os.chdir(original)

    def test_pair_cli_uses_explicit_source_and_frozen_dispatch(self):
        self.assertEqual(module_command('corridorkey.benchmark_pair', '--help'),
                         [sys.executable, '-m', 'corridorkey.benchmark_pair', '--help'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command('corridorkey.benchmark_pair', '--help'),
                             [sys.executable, '--corridorkey-module', 'corridorkey.benchmark_pair', '--help'])
        result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process',
                                 '--corridorkey-module', 'corridorkey.benchmark_pair', '--help'],
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--manifest', result.stdout)
        self.assertIn('--output', result.stdout)

    def test_cache_cli_uses_explicit_dispatch_and_preserves_workspace(self):
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        self.assertEqual(module_command('corridorkey.cache_maintenance', '--help'),
                         [sys.executable, '-m', 'corridorkey.cache_maintenance', '--help'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command('corridorkey.cache_maintenance', '--help'),
                             [sys.executable, '--corridorkey-module', 'corridorkey.cache_maintenance', '--help'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, workspace = root / 'original.bin', root / 'cache'
            source.write_bytes(b'original output bytes')
            key = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'dispatch'}))
            with BundleStore(workspace, 10000) as store:
                store.publish(key, {'alpha.exr': source}, {})
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process', '--corridorkey-module',
                                     'corridorkey.cache_maintenance', 'inspect', '--workspace', str(workspace)],
                                    capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data['entry_count'], 1)
            self.assertEqual(data['entries'][0]['key'], key.key)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_backup_cli_source_frozen_and_actual_restore_dispatch(self):
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        self.assertEqual(module_command('corridorkey.cache_backup', '--help'),
                         [sys.executable, '-m', 'corridorkey.cache_backup', '--help'])
        with patch.object(sys, 'frozen', True, create=True):
            self.assertEqual(module_command('corridorkey.cache_backup', '--help'),
                             [sys.executable, '--corridorkey-module', 'corridorkey.cache_backup', '--help'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, workspace = root / 'original.bin', root / 'cache'
            source.write_bytes(b'preserved backup bytes')
            key = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'backup-cli'}))
            with BundleStore(workspace, 10000) as store:
                store.publish(key, {'alpha.exr': source}, {})
            before = {str(p): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}

            def invoke(*args):
                result = subprocess.run([sys.executable, '-m', 'corridorkey.app_process', '--corridorkey-module',
                                         'corridorkey.cache_backup', *args], capture_output=True, text=True,
                                        timeout=20, check=False)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                return json.loads(result.stdout)

            backup = invoke('backup', '--workspace', str(workspace), '--output-parent', str(root))
            self.assertTrue(backup['usable'])
            invoke('inspect', '--backup', backup['destination'])
            restored = invoke('restore', '--backup', backup['destination'], '--output-parent', str(root))
            self.assertTrue(restored['usable'])
            with BundleStore(restored['destination'], 10000) as store:
                self.assertIsNotNone(store.lookup(key))
            self.assertEqual(source.read_bytes(), b'preserved backup bytes')
            self.assertEqual(before, {str(p): p.read_bytes() for p in workspace.rglob('*') if p.is_file()})
