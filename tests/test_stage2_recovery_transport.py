"""Stage2 recovery is explicit, strict and preserved across CLI/wire admission."""
import contextlib
import io
import unittest
from unittest.mock import patch

from corridorkey.worker_protocol import decode_frame, parse_request, request_frame


class Stage2RecoveryTransportTests(unittest.TestCase):
    def test_old_requests_default_off_and_new_boolean_is_frozen(self):
        payload = {'batch': '/tmp/batch.json', 'workspace': '/tmp/cache', 'max_cache_bytes': 0}
        legacy = parse_request(decode_frame(request_frame('old', 'recover', payload)))
        self.assertFalse(legacy.payload()['durable_stage2'])
        self.assertFalse(legacy.payload()['reuse_stage2_finishing_edits'])
        explicit = {**payload, 'durable_stage2': True}
        request = parse_request(decode_frame(request_frame('new', 'recover', explicit)))
        explicit['durable_stage2'] = False
        self.assertTrue(request.payload()['durable_stage2'])
        self.assertFalse(request.payload()['resume'])
        self.assertFalse(request.payload()['durable_hann'])
        for invalid in (1, 'true', None):
            with self.subTest(value=invalid), self.assertRaises((ValueError, TypeError)):
                request_frame('bad', 'recover', {**payload, 'durable_stage2': invalid})

    def test_finishing_option_requires_stage2_and_survives_wire(self):
        payload = {'batch': '/tmp/batch.json', 'workspace': '/tmp/cache', 'max_cache_bytes': 0,
                   'reuse_stage2_finishing_edits': True}
        with self.assertRaises(ValueError):
            request_frame('bad', 'recover', payload)
        for invalid in (1, 'true', None):
            with self.assertRaises(ValueError):
                request_frame('bad', 'recover', {**payload, 'durable_stage2': True,
                                               'reuse_stage2_finishing_edits': invalid})
        request = parse_request(decode_frame(request_frame('new', 'recover',
            {**payload, 'durable_stage2': True})))
        self.assertTrue(request.payload()['reuse_stage2_finishing_edits'])

    def test_queue_reopen_keeps_opt_in_and_legacy_defaults(self):
        import json
        import tempfile
        from pathlib import Path

        from corridorkey.queue_store import QueueStore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'source.bin').write_bytes(b'identity fixture')
            batch = root / 'batch.json'
            batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                'jobs': [{'input': 'source.bin', 'output_dir': 'output',
                          'settings': {'output_format': 'exr', 'outputs': ['alpha']}}]}))
            queue = QueueStore.create(root)
            store = QueueStore(queue)
            try:
                store.start_session()
                payload = {'batch': str(batch), 'workspace': str(root / 'cache'), 'max_cache_bytes': 0}
                store.prepare('legacy', 'recover', payload)
                store.prepare('edited', 'recover', {**payload, 'durable_stage2': True,
                                                    'reuse_stage2_finishing_edits': True})
            finally:
                store.close()
            for name, expected in (('legacy', False), ('edited', True)):
                record = QueueStore.inspect(queue, request_id=name)
                normalized = parse_request(record['request']).payload()
                self.assertEqual(normalized['reuse_stage2_finishing_edits'], expected)
                self.assertEqual(normalized['durable_stage2'], expected)
                self.assertNotEqual(normalized['batch'], str(batch))

    def test_direct_batch_helper_rejects_invalid_finishing_policy_before_io(self):
        from typing import Any

        from corridorkey.worker_client import run_worker_batch
        invalid_options: list[dict[str, Any]] = [
            {'reuse_stage2_finishing_edits': value, 'durable_stage2': True, 'workspace': '/tmp/cache'}
            for value in (1, 'true', None)]
        invalid_options.extend([
            {'reuse_stage2_finishing_edits': True},
            {'reuse_stage2_finishing_edits': True, 'durable_stage2': True},
            {'reuse_stage2_finishing_edits': True, 'workspace': '/tmp/cache'},
            {'reuse_stage2_finishing_edits': True, 'durable_stage2': 1, 'workspace': '/tmp/cache'}])
        for options in invalid_options:
            with self.subTest(options=options), patch('corridorkey.batch.load_jobs') as load_jobs:
                load_jobs.side_effect = AssertionError('must reject before loading a manifest')
                with patch('corridorkey.worker_client.WorkerClient') as client:
                    client.side_effect = AssertionError('must not create a worker')
                    with self.assertRaises(ValueError):
                        run_worker_batch('/unused/batch', '/unused/report', client_factory=client, **options)
                    load_jobs.assert_not_called()
                    client.assert_not_called()

    def test_worker_cli_requires_workspace_and_transmits_explicit_flag(self):
        from corridorkey.worker import main

        with patch('corridorkey.worker_client.run_worker_batch', return_value={'status': 'complete'}) as runner:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['run', '--batch', '/tmp/batch.json', '--report', '/tmp/report.json', '--durable-stage2'])
            runner.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(['run', '--batch', '/tmp/batch.json', '--report', '/tmp/report.json',
                               '--workspace', '/tmp/cache', '--durable-stage2', '--reuse-stage2-finishing-edits'])
            self.assertEqual(result, 0)
            self.assertTrue(runner.call_args.kwargs['durable_stage2'])
            self.assertTrue(runner.call_args.kwargs['reuse_stage2_finishing_edits'])
            self.assertFalse(runner.call_args.kwargs['resume'])
            self.assertFalse(runner.call_args.kwargs['durable_hann'])
