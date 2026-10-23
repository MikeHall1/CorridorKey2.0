"""Explicit capability probes use bounded owned CPU children, never models."""
import json
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.control import CancellationToken, CancelledError
from corridorkey.readiness_capabilities import _capture, check_capabilities
from corridorkey.session import JobRequest


class CaptureTests(unittest.TestCase):
    def capture(self, program, token=None, **kwargs):
        return _capture([sys.executable, '-c', program], token or CancellationToken(), **kwargs)

    def test_actual_bounded_stdout_stderr_and_exit_status(self):
        raw = self.capture('import sys; print("stdout"); print("stderr",file=sys.stderr)')
        self.assertIn('stdout', raw)
        self.assertIn('stderr', raw)
        with self.assertRaisesRegex(RuntimeError, 'exited 3'):
            self.capture('import sys; print("failed"); sys.exit(3)')

    def test_timeout_overflow_and_cancellation_reap_real_owned_children(self):
        original = subprocess.Popen
        children = []

        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child

        with patch('corridorkey.readiness_capabilities.subprocess.Popen', side_effect=launch):
            with self.assertRaises(TimeoutError):
                self.capture('import time; time.sleep(20)', timeout=.1)
            with patch('corridorkey.readiness_capabilities.MAX_OUTPUT_BYTES', 64), \
                    self.assertRaisesRegex(ValueError, '256 KiB'):
                self.capture('import sys,time; sys.stdout.write("x"*65); sys.stdout.flush(); time.sleep(20)')
            token = CancellationToken()
            timer = threading.Timer(.1, token.cancel)
            timer.start()
            try:
                with self.assertRaises(CancelledError):
                    self.capture('import time; time.sleep(20)', token)
            finally:
                timer.join(2)
        self.assertEqual(len(children), 3)
        self.assertTrue(all(child.returncode is not None for child in children))


class CapabilityTests(unittest.TestCase):
    def job(self, **changes):
        return JobRequest.create('/tmp/local-source', '/tmp/local-output',
                                 InferenceConfig(**changes), WriterConfig())

    def backend(self):
        return json.dumps({'cuda_available': False, 'torch': 'fixture', 'cuda_build': None})

    def test_exr_needs_no_ffmpeg_and_unavailable_cuda_has_no_fallback(self):
        with patch('corridorkey.readiness_capabilities._capture', return_value=self.backend()) as capture:
            report = check_capabilities(self.job(output_format='exr'))
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(report['stages'][0]['status'], 'FAIL')
        self.assertEqual(report['stages'][1]['status'], 'NOT REQUIRED')
        self.assertIsNone(report['effective_backend'])

    def test_video_probes_actual_requested_encoder_and_cutout_override(self):
        def respond(command, token):
            if 'corridorkey.backend_probe' in command:
                return self.backend()
            codec = command[-1].split('=', 1)[1]
            return f'Encoder {codec} [fixture]:\n Supported pixel formats: yuv444p10le yuva444p10le\n'

        with patch('corridorkey.readiness_capabilities._capture', side_effect=respond) as capture, \
                patch('corridorkey.readiness_capabilities.shutil.which', return_value='/fixture/ffmpeg'):
            report = check_capabilities(self.job(outputs={'alpha', 'cutout'}))
        encoders = [stage for stage in report['stages'] if stage['stage'] == 'ffmpeg_capabilities']
        self.assertEqual({stage['details']['encoder'] for stage in encoders}, {'libx264', 'prores_ks'})
        self.assertEqual([stage['status'] for stage in encoders], ['PASS', 'PASS'])
        self.assertEqual(capture.call_count, 3)
        self.assertTrue(all('actual encode/decode NOT VERIFIED' in stage['limits'] for stage in encoders))

    def test_unknown_encoder_missing_pixel_format_and_missing_binary(self):
        for text, expected in [('Codec not recognized', 'FAIL'),
                               ('Encoder libx264 [fixture]:\n Supported pixel formats: yuv420p', 'FAIL'),
                               ('Encoder libx264 [fixture]:', 'NOT VERIFIED')]:
            with self.subTest(text=text), \
                    patch('corridorkey.readiness_capabilities._capture', side_effect=[self.backend(), text]), \
                    patch('corridorkey.readiness_capabilities.shutil.which', return_value='/fixture/ffmpeg'):
                result = check_capabilities(self.job(outputs={'alpha'}))
                self.assertEqual(result['stages'][-1]['status'], expected)
        with patch('corridorkey.readiness_capabilities._capture', return_value=self.backend()), \
                patch('corridorkey.readiness_capabilities.shutil.which', return_value=None):
            result = check_capabilities(self.job())
        self.assertIn('not found', result['stages'][-1]['details'])

    def test_cancellation_and_bad_backend_response_are_not_success(self):
        token = CancellationToken()
        token.cancel()
        with patch('corridorkey.readiness_capabilities._capture') as capture:
            result = check_capabilities(self.job(), cancellation=token)
        self.assertFalse(capture.called)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['interrupted_stage'], 'backend')
        token = CancellationToken()

        def stop(*args):
            token.cancel()
            token.check()

        with patch('corridorkey.readiness_capabilities._capture', side_effect=stop):
            result = check_capabilities(self.job(), cancellation=token)
        self.assertEqual(result['interrupted_stage'], 'backend')
        with patch('corridorkey.readiness_capabilities._capture', return_value='[]'):
            result = check_capabilities(self.job(output_format='exr'))
        self.assertEqual(result['stages'][0]['status'], 'FAIL')

    def test_alias_and_multiple_encoder_blocks_never_infer_selection_or_mix_formats(self):
        raw = ('Encoder h264_hardware [fixture]:\n Supported pixel formats: nv12\n'
               'Encoder libx264 [fixture]:\n Supported pixel formats: yuv444p10le\n')
        for codec, expected in [('h264', 'NOT VERIFIED'), ('libx264', 'PASS'), ('h264_hardware', 'FAIL')]:
            job = JobRequest.create('/tmp/local-source', '/tmp/local-output',
                                    InferenceConfig(outputs={'alpha'}), WriterConfig(codec=codec))
            with self.subTest(codec=codec), \
                    patch('corridorkey.readiness_capabilities._capture', side_effect=[self.backend(), raw]), \
                    patch('corridorkey.readiness_capabilities.shutil.which', return_value='/fixture/ffmpeg'):
                result = check_capabilities(job)
            stage = result['stages'][-1]
            self.assertEqual(stage['status'], expected)
            self.assertEqual(stage['details']['encoder'], codec)
            self.assertEqual(len(stage['details']['encoder_candidates']), 2)
            self.assertEqual(job.configurations()[1].codec, codec)


if __name__ == '__main__':
    unittest.main()
