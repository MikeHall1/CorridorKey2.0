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
        explicit = {**payload, 'durable_stage2': True}
        request = parse_request(decode_frame(request_frame('new', 'recover', explicit)))
        explicit['durable_stage2'] = False
        self.assertTrue(request.payload()['durable_stage2'])
        self.assertFalse(request.payload()['resume'])
        self.assertFalse(request.payload()['durable_hann'])
        for invalid in (1, 'true', None):
            with self.subTest(value=invalid), self.assertRaises((ValueError, TypeError)):
                request_frame('bad', 'recover', {**payload, 'durable_stage2': invalid})

    def test_worker_cli_requires_workspace_and_transmits_explicit_flag(self):
        from corridorkey.worker import main

        with patch('corridorkey.worker_client.run_worker_batch', return_value={'status': 'complete'}) as runner:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['run', '--batch', '/tmp/batch.json', '--report', '/tmp/report.json', '--durable-stage2'])
            runner.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(['run', '--batch', '/tmp/batch.json', '--report', '/tmp/report.json',
                               '--workspace', '/tmp/cache', '--durable-stage2'])
            self.assertEqual(result, 0)
            self.assertTrue(runner.call_args.kwargs['durable_stage2'])
            self.assertFalse(runner.call_args.kwargs['resume'])
            self.assertFalse(runner.call_args.kwargs['durable_hann'])
