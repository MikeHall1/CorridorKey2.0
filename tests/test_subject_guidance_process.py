"""Actual owned CPU fixture children; no SAM2 model/provider quality evidence."""
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from corridorkey.control import CancelledError
from corridorkey.subject_guidance import (
    Click,
    GuidanceFrame,
    GuidanceUnavailable,
    ProviderPrediction,
)
from corridorkey.subject_guidance_process import (
    MAX_MESSAGE_BYTES,
    SubjectGuidanceProcess,
)


class FixtureProvider:
    device = 'cpu'

    def __init__(self, mode='normal'):
        self.mode = mode
        self.embeddings = 0
        self.frame = None

    def set_frame(self, frame, control):
        self.frame = frame
        self.embeddings += 1
        print('fixture Python stdout must not enter protocol')
        os.write(1, b'fixture native stdout must not enter protocol\n')
        return 32

    def predict(self, clicks, previous_logits, control):
        if self.mode == 'crash':
            os._exit(7)
        if self.mode == 'ignore_term':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if self.mode in ('block', 'ignore_term'):
            print('fixture native prediction entered', flush=True)
            while True:
                time.sleep(.01)
        assert self.frame is not None
        mask = bytearray(self.frame.width * self.frame.height)
        for click in clicks:
            mask[click.y * self.frame.width + click.x] = 255 if click.positive else 0
        return ProviderPrediction(bytes(mask), 'binary_u8', struct.pack('<f', 1),
            json.dumps({'fixture': True, 'embedding_evaluations': self.embeddings,
                        'previous_logits': previous_logits is not None}), fixture=True)

    def reset(self):
        self.frame = None

    def close(self):
        self.reset()


def fixture_command(mode='normal'):
    return [sys.executable, '-u', '-c',
        ('from corridorkey.subject_guidance_process import _serve; '
        'from tests.test_subject_guidance_process import FixtureProvider; '
        f'_serve(provider=FixtureProvider({mode!r}))')]


def corrupt_provenance_child(field):
    import corridorkey.subject_guidance_process as transport
    encode = transport._encode
    def changed(value):
        if value.get('type') == 'result' and 'provenance_json' in value.get('payload', {}):
            provenance = json.loads(value['payload']['provenance_json'])
            if field == 'clicks':
                provenance['clicks'] = [{'x': 0, 'y': 0, 'positive': True}]
            elif field == 'numeric_type':
                provenance['clicks'][0]['x'] = float(provenance['clicks'][0]['x'])
            else:
                provenance['mask_format'] = 'coverage_f32'
            value['payload']['provenance_json'] = json.dumps(provenance)
        return encode(value)
    transport._encode = changed
    transport._serve(provider=FixtureProvider())


class GuidanceProcessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.frame = GuidanceFrame('a' * 64, 12, 'assumed-sRGB-u8-fixture', 16, 12, bytes(16 * 12 * 3))
        self.clicks = (Click(3, 4, True),)

    def process(self, mode='normal', **kwargs):
        process = SubjectGuidanceProcess(self.root / 'config.yaml', self.root / 'weights.pt', workspace=self.root,
            fixture_command=fixture_command(mode), grace_seconds=.05, **kwargs)
        self.addCleanup(process.close)
        return process

    def test_parent_import_is_stdlib_only(self):
        output = subprocess.check_output([sys.executable, '-S', '-c',
            'import sys;import corridorkey.subject_guidance_process;assert not any(n in sys.modules for n in ("torch","numpy","sam2","PySide6"));print("PASS")'],
            cwd=Path(__file__).resolve().parents[1], text=True)
        self.assertEqual(output.strip(), 'PASS')

    def test_matching_wire_ids_cannot_hide_changed_click_or_mask_format_provenance(self):
        for field in ('clicks', 'numeric_type', 'mask_format'):
            with self.subTest(field=field):
                command = [sys.executable, '-u', '-c',
                    ('from tests.test_subject_guidance_process import corrupt_provenance_child; '
                     f'corrupt_provenance_child({field!r})')]
                process = SubjectGuidanceProcess(self.root / 'config', self.root / 'weights', workspace=self.root,
                    fixture_command=command, grace_seconds=.05)
                try:
                    with self.assertRaisesRegex(ValueError, 'provenance'):
                        process.predict(self.frame, self.clicks)
                    self.assertIsNone(process.pid)
                    self.assertFalse(any(thread.is_alive() for thread in process._threads))
                finally:
                    process.close()

    def test_lazy_persistent_refinement_clear_and_source_rebind(self):
        process = self.process()
        self.assertIsNone(process.pid)
        first = process.predict(self.frame, self.clicks)
        pid = process.pid
        self.assertIsNotNone(pid)
        self.assertEqual(first.mask_bytes[67], 255)
        second = process.predict(self.frame, (*self.clicks, Click(3, 4, False), Click(7, 4, True)))
        self.assertEqual(process.pid, pid)
        self.assertEqual(second.mask_bytes[67], 0)
        metadata = json.loads(second.provenance_json)
        self.assertEqual(metadata['provider']['embedding_evaluations'], 1)
        self.assertTrue(metadata['provider']['previous_logits'])
        self.assertIn('no SAM2 inference', metadata['evidence'])
        process.clear()
        cleared = process.predict(self.frame, self.clicks)
        self.assertEqual(process.pid, pid)
        self.assertFalse(json.loads(cleared.provenance_json)['provider']['previous_logits'])
        self.assertEqual(json.loads(cleared.provenance_json)['provider']['embedding_evaluations'], 1)
        rebound = process.predict(replace(self.frame, source_frame=13), self.clicks)
        self.assertEqual(json.loads(rebound.provenance_json)['provider']['embedding_evaluations'], 2)
        owned_threads = list(process._threads)
        process.close()
        self.assertIsNone(process.pid)
        self.assertTrue(owned_threads)
        self.assertFalse(any(thread.is_alive() for thread in owned_threads))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_callback_cancellation_and_timeout_kill_reap_transport(self):
        for mode, timeout in (('block', 2), ('ignore_term', .15)):
            with self.subTest(mode=mode):
                process = self.process(mode, operation_timeout=timeout)
                deadline = time.monotonic() + .3
                def check(mode=mode, deadline=deadline):
                    if mode == 'block' and time.monotonic() >= deadline:
                        raise CancelledError('fixture user cancellation')
                with self.assertRaises(CancelledError if mode == 'block' else TimeoutError):
                    process.predict(self.frame, self.clicks, check=check)
                self.assertIsNone(process.pid)
                self.assertIsNone(process.cleanup_error)
                self.assertFalse(any(thread.is_alive() for thread in process._threads))

    def test_clear_busy_fences_result_and_does_not_start_competing_owner(self):
        process = self.process('block')
        errors = []
        def run():
            try:
                process.predict(self.frame, self.clicks)
            except CancelledError as exc:
                errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 3
        while not any(b'fixture native prediction entered' in chunk for chunk in list(process._logs)) and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertIsNotNone(process.pid)
        self.assertTrue(any(b'fixture native prediction entered' in chunk for chunk in process._logs))
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            process.predict(self.frame, self.clicks)
        process.clear()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsNone(process.pid)

    def test_crash_malformed_and_oversized_child_streams_are_retired(self):
        process = self.process('crash')
        with self.assertRaisesRegex(RuntimeError, 'protocol|closed'):
            process.predict(self.frame, self.clicks)
        self.assertIsNone(process.pid)
        for suffix, expected in (("out.write(struct.pack('>I', 33554433));out.flush()", 'byte bound'),
                                 ("out.write(_encode({'bad':'schema'}));out.flush()", 'identity/schema')):
            with self.subTest(suffix=suffix):
                script = ('import sys,struct;from corridorkey.subject_guidance_process import _encode,_read;'
                    'out=sys.stdout.buffer;out.write(_encode({"type":"ready","version":1,"device":"cpu","precision":"float32"}));out.flush();'
                    '_read(sys.stdin.buffer);' + suffix)
                process = SubjectGuidanceProcess(self.root / 'config', self.root / 'weights', workspace=self.root,
                    fixture_command=[sys.executable, '-u', '-c', script], grace_seconds=.05)
                self.addCleanup(process.close)
                with self.assertRaisesRegex((RuntimeError, ValueError), expected):
                    process.predict(self.frame, self.clicks)
                self.assertIsNone(process.pid)

    def test_admission_precedes_child_start_or_base64_materialization(self):
        process = self.process()
        invalid = replace(self.frame, width=4096, height=4096)
        with patch('corridorkey.subject_guidance_process.base64.b64encode') as encoded, self.assertRaises(ValueError):
            process.predict(invalid, self.clicks)
        encoded.assert_not_called()
        self.assertIsNone(process.pid)
        self.assertEqual(MAX_MESSAGE_BYTES, 32 * 1024**2)

    def test_startup_timeout_and_closed_owner_never_launch_another_child(self):
        process = SubjectGuidanceProcess(self.root / 'config', self.root / 'weights', workspace=self.root,
            fixture_command=[sys.executable, '-c', 'import time;time.sleep(5)'], startup_timeout=.1, grace_seconds=.05)
        self.addCleanup(process.close)
        with self.assertRaises(TimeoutError):
            process.predict(self.frame, self.clicks)
        self.assertIsNone(process.pid)
        process.close()
        with patch('corridorkey.subject_guidance_process.subprocess.Popen') as launch, self.assertRaises(RuntimeError):
            process.predict(self.frame, self.clicks)
        launch.assert_not_called()

    @unittest.skipIf(os.name == 'nt', 'POSIX group observation safety fixture')
    def test_cleanup_uncertainty_preserves_cancel_error_and_prevents_new_owner(self):
        process = self.process('block')
        deadline = time.monotonic() + .2
        def cancel():
            if time.monotonic() >= deadline:
                raise CancelledError('fixture cancellation stays primary')
        actual_signal = os.killpg
        def denied_query(pid, sig):
            if sig == 0:
                raise PermissionError('fixture group observation denied')
            return actual_signal(pid, sig)
        with patch('corridorkey.subject_guidance_process.os.killpg', side_effect=denied_query), \
                self.assertRaisesRegex(CancelledError, 'stays primary') as raised:
            process.predict(self.frame, self.clicks, check=cancel)
        self.assertIn('cleanup NOT VERIFIED', raised.exception.__notes__[0])
        self.assertIsNotNone(process.cleanup_error)
        with patch('corridorkey.subject_guidance_process.subprocess.Popen') as launch, self.assertRaises(GuidanceUnavailable):
            process.predict(self.frame, self.clicks)
        launch.assert_not_called()
        process.close()

    def test_actual_product_child_missing_assets_is_actionable_and_reaped(self):
        process = SubjectGuidanceProcess(self.root / 'missing.yaml', self.root / 'missing.pt', workspace=self.root)
        self.addCleanup(process.close)
        with self.assertRaisesRegex(GuidanceUnavailable, 'No download'):
            process.predict(self.frame, self.clicks)
        self.assertIsNone(process.pid)
        self.assertIsNone(process.cleanup_error)


if __name__ == '__main__':
    unittest.main()
