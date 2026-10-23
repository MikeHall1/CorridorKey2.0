"""Explicit EXR writer-prefix policy admission; no inference/quality claims."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class WriterPrefixPlumbingTests(unittest.TestCase):
    def test_old_wire_defaults_false_and_explicit_resume_policy_is_immutable(self):
        from corridorkey.worker_protocol import (
            decode_frame,
            parse_request,
            request_frame,
        )
        payload = {'batch': '/tmp/batch.json', 'workspace': '/tmp/cache', 'max_cache_bytes': 0}
        self.assertFalse(parse_request(decode_frame(request_frame('old', 'recover', payload))).payload()['durable_writer_prefix'])
        explicit = {**payload, 'durable_writer_prefix': True, 'resume': True}
        request = parse_request(decode_frame(request_frame('new', 'recover', explicit)))
        explicit['durable_writer_prefix'] = False
        self.assertTrue(request.payload()['durable_writer_prefix'])
        self.assertTrue(request.payload()['resume'])
        self.assertFalse(request.payload()['durable_stage2'])
        for value in (1, 'true', None):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                request_frame('bad', 'recover', {**payload, 'durable_writer_prefix': value})

    def test_worker_cli_requires_workspace_and_forwards_flag_without_auto_resume(self):
        from corridorkey.worker import main
        with patch('corridorkey.worker_client.run_worker_batch', return_value={'status': 'complete'}) as runner:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['run', '--batch', '/tmp/batch.json', '--report', '/tmp/report.json', '--durable-writer-prefix'])
            runner.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run', '--batch', '/tmp/batch.json', '--report', '/tmp/report.json',
                                       '--workspace', '/tmp/cache', '--durable-writer-prefix']), 0)
            self.assertTrue(runner.call_args.kwargs['durable_writer_prefix'])
            self.assertFalse(runner.call_args.kwargs['resume'])

    def test_direct_recovery_cli_forwards_prefix_and_resume(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.recovery import main
        from corridorkey.session import JobRequest
        job = JobRequest.create('/tmp/source', '/tmp/output', InferenceConfig(output_format='exr'))
        paths = SimpleNamespace(validate=lambda: None, **dict.fromkeys(
            ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir'), Path('/tmp/fixture')))
        with patch('corridorkey.recovery.load_jobs', return_value=([job], None)), \
             patch('corridorkey.runtime.RuntimePaths.from_package', return_value=paths), \
             patch('corridorkey.recovery.cuda_environment', return_value={}), \
             patch('corridorkey.recovery.run_recoverable', return_value={'status': 'complete', 'report_path': '/tmp/report', 'jobs': []}) as runner, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['run', '--batch', '/tmp/batch.json', '--workspace', '/tmp/cache',
                                   '--max-cache-bytes', '1024', '--resume', '--durable-writer-prefix']), 0)
        self.assertTrue(runner.call_args.kwargs['durable_writer_prefix'])
        self.assertTrue(runner.call_args.kwargs['resume'])

    def test_worker_batch_client_passes_policy_and_closes_same_owner(self):
        from corridorkey.worker_client import run_worker_batch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.mov'
            source.write_bytes(b'fixture; no decode')
            batch = root / 'batch.json'
            batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': str(source), 'output_dir': str(root / 'out'), 'settings': {'output_format': 'exr'}}]}))
            sent = []
            client = Mock(generation='fixture', capacity=1, cleanup_error=None, log_tail='')
            client.process.pid = 123
            client.send.side_effect = lambda action, payload: sent.append((action, payload)) or 'request'
            client.next_event.side_effect = [{'request_id': 'request', 'kind': 'result', 'payload': {'record': {'status': 'complete'}}}, EOFError()]
            client.close.return_value = True
            result = run_worker_batch(batch, root / 'report.json', workspace=root / 'cache',
                                      durable_writer_prefix=True, resume=True, client_factory=lambda **kw: client)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(sent[0][0], 'recover')
            self.assertTrue(sent[0][1]['durable_writer_prefix'])
            self.assertTrue(sent[0][1]['resume'])
            client.close.assert_called_once()

    def test_stress_policy_accepts_only_boolean_and_defaults_off(self):
        from corridorkey.stress_suite import load_stress_manifest
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stress.json'
            data: dict[str, Any] = {'schema_version': 1, 'kind': 'production_stress', 'cases': [
                {'id': 'prefix', 'scenario': 'interruption_recovery', 'batch_manifest': 'batch.json'}]}
            path.write_text(json.dumps(data))
            self.assertFalse(load_stress_manifest(path)[0]['cases'][0]['recovery']['durable_writer_prefix'])
            data['cases'][0]['recovery'] = {'durable_writer_prefix': True}
            path.write_text(json.dumps(data))
            self.assertTrue(load_stress_manifest(path)[0]['cases'][0]['recovery']['durable_writer_prefix'])
            data['cases'][0]['recovery']['durable_writer_prefix'] = 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_stress_manifest(path)

    def test_controller_omits_default_kwarg_and_captures_explicit_adapter_policy(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.recovery import run_recoverable
        from corridorkey.session import JobRequest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.bin'
            source.write_bytes(b'input preserved')
            implementation = root / 'implementation' / 'corridorkey'
            implementation.mkdir(parents=True)
            (implementation / 'runtime.py').write_text('# fixture')
            assets = {}
            for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
                assets[name] = root / name
                assets[name].write_bytes(name.encode())
            assets['cradio_dir'] = root / 'cradio'
            assets['cradio_dir'].mkdir()
            (assets['cradio_dir'] / 'config.json').write_text('{}')
            environment = {'backend': 'fixture', 'precision': 'float32', 'dependencies': {'fixture': '1'}, 'runtime': {'fixture': True}}
            job = JobRequest.create(source, root / 'output', InferenceConfig(output_format='exr', outputs={'alpha'}))
            calls = []

            class SessionOwner:
                last_result = None

                def __init__(self, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            class LegacySession(SessionOwner):
                def run(self, job, *, cancellation, progress, durable_foundations, durable_rvm, durable_hann, durable_stage2):
                    calls.append('legacy signature accepted')
                    raise RuntimeError('fixture stops before rendering')

            class PrefixSession(SessionOwner):
                def run(self, job, *, durable_writer_prefix, **kwargs):
                    calls.append(durable_writer_prefix)
                    raise RuntimeError('fixture stops before rendering')

            for enabled, resume in ((False, False), (True, False), (True, True)):
                with self.subTest(enabled=enabled, resume=resume), \
                     patch('corridorkey.durable_writer_prefix.DurableWriterPrefix') as adapter:
                    def factory(session_type=PrefixSession if enabled else LegacySession, **kwargs) -> Any:
                        return session_type(**kwargs)

                    with self.assertRaisesRegex(RuntimeError, 'fixture stops'):
                        run_recoverable([job], workspace=root / f'cache-{enabled}-{resume}', max_cache_bytes=1024,
                            assets=assets, environment=environment, implementation_root=implementation.parent,
                            durable_writer_prefix=enabled, resume=resume,
                            session_factory=factory)
                    if enabled:
                        self.assertIs(calls[-1], adapter.return_value)
                        self.assertIs(adapter.call_args.kwargs['resume'], resume)
                        identity = adapter.call_args.args[1].to_dict()
                        self.assertTrue(identity['environment']['recovery_policy']['durable_writer_prefix'])
                    else:
                        self.assertEqual(calls[-1], 'legacy signature accepted')
                        adapter.assert_not_called()
            self.assertEqual(source.read_bytes(), b'input preserved')


    def test_controller_rechecks_restored_prefix_before_publication_and_journal_seal(self):
        from corridorkey import recovery
        from corridorkey.recovery_store import BundleStore, CacheInvalid
        from tests.test_recovery_runner import RecoveryRunnerTests

        def run_phase(phase):
            fixture = RecoveryRunnerTests('test_second_identical_job_restores_bytes_and_resume_never_loads_model')
            fixture.setUp()
            try:
                expected = {}
                mutation_path = []

                def mutate():
                    mutation_path[0].write_bytes(b'changed native prefix after rendering')

                def verify_prefix():
                    if any(path.read_bytes() != content for path, content in expected.items()):
                        raise CacheInvalid('fixture verified prefix changed')
                    return {str(path.resolve().relative_to(Path(fixture.renders[-1]).resolve())):
                            {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                            for path, content in expected.items()}

                def factory(mutation_phase=phase, **kwargs):
                    session = fixture.factory(**kwargs)
                    original_render = session._runner

                    def render(*args, **kw):
                        original_render(*args, **kw)
                        expected.update({path: path.read_bytes() for path in fixture.root.rglob('*.exr')})
                        mutation_path.extend(expected)
                        if mutation_phase == 'before_publication':
                            mutate()

                    session._runner = render
                    return session

                original_verify_inventory = recovery.verify_inventory

                def inventory_then_mutate(*args, mutation_phase=phase, **kwargs):
                    valid = original_verify_inventory(*args, **kwargs)
                    if valid and mutation_phase == 'before_journal':
                        mutate()
                    return valid

                original_publish = BundleStore.publish

                def publish_with_mutation(store, *args, **kwargs):
                    if phase == 'during_publication':
                        mutate()
                    return original_publish(store, *args, **kwargs)

                adapter = Mock()
                adapter.snapshot.return_value = {'enabled': True}
                adapter.verify_restored_prefix.side_effect = verify_prefix
                with patch('corridorkey.durable_writer_prefix.DurableWriterPrefix', return_value=adapter), \
                     patch('corridorkey.recovery.verify_inventory', side_effect=inventory_then_mutate), \
                     patch.object(BundleStore, 'publish', new=publish_with_mutation), \
                     self.assertRaisesRegex(CacheInvalid, 'receipt' if phase == 'during_publication' else 'verified prefix changed'):
                    recovery.run_recoverable(fixture.jobs[:1], workspace=fixture.workspace,
                        max_cache_bytes=100000, assets=fixture.assets, environment=fixture.environment,
                        implementation_root=fixture.root / 'implementation', durable_writer_prefix=True,
                        session_factory=factory)
                self.assertEqual(adapter.verify_restored_prefix.call_count, 2 if phase == 'before_journal' else 1)
                if phase == 'during_publication':
                    self.assertEqual(list((fixture.workspace / 'entries').iterdir()), [])
                report = json.loads(next((fixture.workspace / 'reports').glob('run-*.json')).read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['jobs'], [])
                journal = json.loads(next((fixture.workspace / 'journals').glob('*.json')).read_text())
                self.assertEqual(journal['payload']['jobs'][0][-1]['status'], 'failed')
                self.assertEqual(fixture.source.read_bytes(), b'synthetic identity input; renderer does not perform neural inference')
            finally:
                fixture.doCleanups()

        for phase in ('before_publication', 'during_publication', 'before_journal'):
            with self.subTest(phase=phase):
                run_phase(phase)

    def test_zero_quota_near_capacity_refuses_before_new_journal_or_report(self):
        from corridorkey import cache_maintenance
        from corridorkey.config import InferenceConfig
        from corridorkey.recovery import run_recoverable
        from corridorkey.recovery_store import BundleStore, CacheCapacity
        from corridorkey.session import JobRequest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / 'workspace'
            with BundleStore(workspace, 0):
                pass
            reports = workspace / 'reports'
            reports.mkdir()
            for index in range(3):
                (reports / f'run-{index:032x}.json').write_text('{}')
            original = {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
            source = root / 'source.bin'
            source.write_bytes(b'preserved local source')
            job = JobRequest.create(source, root / 'output', InferenceConfig(output_format='exr'))
            factory = Mock(side_effect=AssertionError('No session may start without controller headroom'))
            with patch.object(cache_maintenance, 'MAX_FILES', len(original)):
                self.assertEqual(cache_maintenance.inventory(workspace)['report_count'], 3)
                with self.assertRaisesRegex(CacheCapacity, 'journal/report structural headroom'):
                    run_recoverable([job], workspace=workspace, max_cache_bytes=0, assets={}, environment={},
                                    durable_writer_prefix=True, session_factory=factory)
                self.assertEqual(cache_maintenance.inventory(workspace)['report_count'], 3)
            factory.assert_not_called()
            self.assertFalse((workspace / 'journals').exists())
            self.assertEqual(original, {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob('*') if p.is_file()})
            self.assertEqual(source.read_bytes(), b'preserved local source')

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
    def test_current_prefix_summary_excludes_historical_counts_and_never_invents_missing_native_writes(self):
        from corridorkey.desktop_recovery import writer_prefix_summary
        snapshot = {'restored_prefix_frames': 2, 'restored_files': 4, 'restored_bytes': 200,
                    'stored_frames': 3, 'stored_files': 6, 'stored_bytes': 300,
                    'newly_submitted_frames': 3, 'native_new_written_frames': 3, 'native_new_written_files': 6,
                    'native_completion_status': 'complete', 'budget_rejected_frames': 0, 'corrupt_entries': 0}
        record = {'diagnostics': {'durable_writer_prefix': snapshot}}
        report = {'jobs': [{'mode': 'render', 'new_session_run': True, 'origin_record': record},
                           {'mode': 'verified_cache_restore', 'new_session_run': False, 'origin_record': record}]}
        text = writer_prefix_summary(report)
        self.assertIn('Verified prefix restored: 2 frames; 4 files; 200 bytes', text)
        self.assertIn('Confirmed new native writes: 3 frames / 6 files', text)
        self.assertIn('Native completion: complete', text)
        self.assertIn('global cache retention is not guaranteed', text)
        self.assertIn('No current-attempt', writer_prefix_summary({'jobs': report['jobs'][1:]}))
        text = writer_prefix_summary({'jobs': [], 'failed_attempt': {'diagnostics': {'durable_writer_prefix': {
            'restored_prefix_frames': 2, 'native_completion_status': [], 'checkpoint_disabled': True,
            'last_error': 'checkpoint failed'}}}})
        self.assertIn('Confirmed new native writes: unavailable', text)
        self.assertIn('Native completion: not verified', text)
        self.assertIn('checkpoint failed', text)
        self.assertIn('Prefix checkpointing was disabled during this attempt', text)


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class WriterPrefixPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['writer-prefix-test', '-platform', 'offscreen'])

    def test_checkbox_is_opt_in_loaded_without_submission_and_request_is_frozen(self):
        from corridorkey.desktop_recovery import RecoveryPanel
        from tests.qt_cleanup import dispose_widget
        panel = RecoveryPanel()
        try:
            received = []
            panel.run_requested.connect(received.append)
            payload = {'batch': '/tmp/batch.json', 'workspace': '/tmp/cache', 'max_cache_bytes': 1024}
            panel.load_payload(payload)
            self.assertFalse(panel.writer_prefix.isChecked())
            panel.load_payload({**payload, 'durable_writer_prefix': True})
            self.assertTrue(panel.writer_prefix.isChecked())
            self.assertEqual(received, [])
            panel.request(True)
            self.assertTrue(received[0]['durable_writer_prefix'])
            self.assertTrue(received[0]['resume'])
            panel.writer_prefix.setChecked(False)
            self.assertTrue(received[0]['durable_writer_prefix'])
            panel.request(False)
            self.assertFalse(received[1]['durable_writer_prefix'])
            self.assertFalse(received[1]['resume'])
            self.assertFalse(received[1]['durable_stage2'])
        finally:
            dispose_widget(panel)
