"""New upstream policy is explicit across public, persisted and desktop admission."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from corridorkey.worker_protocol import decode_frame, parse_request, request_frame
from tests import test_desktop_recovery as desktop


class UpstreamTransportTests(unittest.TestCase):
    def test_wire_and_saved_queue_are_immutable_and_old_missing_means_off(self):
        from corridorkey.queue_store import QueueStore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'source.bin').write_bytes(b'fixture')
            batch = root / 'batch.json'
            batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                'jobs': [{'input': 'source.bin', 'output_dir': 'output', 'settings': {'output_format': 'exr'}}]}))
            payload = {'batch': str(batch), 'workspace': str(root / 'cache'), 'max_cache_bytes': 0}
            self.assertFalse(parse_request(decode_frame(request_frame('old', 'recover', payload)))
                             .payload()['reuse_upstream_finishing_edits'])
            with self.assertRaises(ValueError):
                request_frame('bad', 'recover', {**payload, 'reuse_upstream_finishing_edits': True})
            for value in (1, None, 'true'):
                with self.assertRaises(ValueError):
                    request_frame('bad', 'recover', {**payload, 'durable_hann': True,
                        'reuse_upstream_finishing_edits': value})
            queue = QueueStore.create(root)
            store = QueueStore(queue)
            try:
                store.start_session()
                store.prepare('old', 'recover', payload)
                payload.update(durable_hann=True, reuse_upstream_finishing_edits=True)
                store.prepare('new', 'recover', payload)
                payload['reuse_upstream_finishing_edits'] = False
            finally:
                store.close()
            for name, enabled in (('old', False), ('new', True)):
                saved = parse_request(QueueStore.inspect(queue, request_id=name)['request']).payload()
                self.assertEqual(saved['reuse_upstream_finishing_edits'], enabled)
                self.assertFalse(saved['durable_stage2'])
                self.assertFalse(saved['reuse_stage2_finishing_edits'])

    def test_direct_helper_requires_real_boolean_upstream_flag_and_workspace_before_io(self):
        from corridorkey.worker_client import run_worker_batch
        invalid: tuple[dict[str, Any], ...] = ({'durable_foundations': True}, {'workspace': '/cache'},
                        {'durable_hann': 1, 'workspace': '/cache'},
                        {'durable_rvm': 1, 'durable_hann': True, 'workspace': '/cache'},
                        {'durable_rvm': True}, {'durable_stage2': True, 'workspace': '/cache'})
        for options in invalid:
            with patch('corridorkey.batch.load_jobs') as read, patch('corridorkey.worker_client.WorkerClient') as worker:
                with self.assertRaises(ValueError):
                    run_worker_batch('/batch', '/report', reuse_upstream_finishing_edits=True, **options)
                read.assert_not_called()
                worker.assert_not_called()

    def test_worker_and_recovery_cli_guard_before_batch_io_and_forward_explicit_policy(self):
        from corridorkey.recovery import main as recovery_main
        from corridorkey.worker import main
        with patch('corridorkey.worker_client.run_worker_batch', return_value={'status': 'complete'}) as runner:
            for tail in ([], ['--workspace', '/cache'], ['--workspace', '/cache', '--durable-stage2']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['run', '--batch', '/batch', '--report', '/report', '--reuse-upstream-finishing-edits', *tail])
                runner.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run', '--batch', '/batch', '--report', '/report', '--workspace', '/cache',
                    '--durable-hann', '--reuse-upstream-finishing-edits']), 0)
            self.assertTrue(runner.call_args.kwargs['reuse_upstream_finishing_edits'])
            self.assertFalse(runner.call_args.kwargs['reuse_stage2_finishing_edits'])
        with patch('corridorkey.recovery.load_jobs') as read:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                recovery_main(['run', '--batch', '/batch', '--workspace', '/cache', '--reuse-upstream-finishing-edits'])
            read.assert_not_called()


@unittest.skipUnless(desktop.importlib.util.find_spec('PySide6'), 'optional Qt unavailable')
class UpstreamDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        desktop.DesktopRecoveryTests.setUpClass.__func__(cls)
    setUp = desktop.DesktopRecoveryTests.setUp
    tearDown = desktop.DesktopRecoveryTests.tearDown

    def test_explicit_upstream_control_requires_cache_captures_and_reopens_without_mutation(self: Any):
        panel = self.panel
        self.assertFalse(panel.upstream_finishing_edits.isChecked())
        panel.upstream_finishing_edits.setChecked(True)
        panel.request(False)
        self.assertEqual(self.received, [])
        self.assertIn('requires', panel.status.text())
        panel.hann.setChecked(True)
        panel.request(False)
        captured = self.received[0]
        self.assertTrue(captured['reuse_upstream_finishing_edits'])
        self.assertFalse(captured['reuse_stage2_finishing_edits'])
        panel.upstream_finishing_edits.setChecked(False)
        self.assertTrue(captured['reuse_upstream_finishing_edits'])
        panel.load_payload(captured)
        self.assertTrue(panel.upstream_finishing_edits.isChecked())
        panel.load_payload({key: value for key, value in captured.items() if key != 'reuse_upstream_finishing_edits'})
        self.assertFalse(panel.upstream_finishing_edits.isChecked())
        self.assertEqual(len(self.received), 1)
