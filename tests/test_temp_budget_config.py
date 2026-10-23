"""Scratch ceilings survive the real CLI, batch and immutable job boundaries."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from corridorkey.batch import load_jobs
from corridorkey.config import InferenceConfig
from corridorkey.session import InferenceSession, JobRequest


class TempBudgetConfigTests(unittest.TestCase):
    def test_budget_validation_and_immutable_request_roundtrip(self):
        self.assertIsNone(InferenceConfig().temp_map_budget_bytes)
        for value in (0, 1, 2**63 - 1):
            with self.subTest(value=value):
                config = InferenceConfig(temp_map_budget_bytes=value)
                request = JobRequest.create('.', '.', config)
                config.temp_map_budget_bytes = None
                self.assertEqual(request.configurations()[0].temp_map_budget_bytes, value)
        for value in (True, False, -1, 2**63, 1.0, '1024'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'temp_map_budget_bytes'):
                invalid = InferenceConfig()
                vars(invalid)['temp_map_budget_bytes'] = value
                invalid.validate()

    def test_cli_captures_ceiling_without_changing_inference_choices(self):
        import infer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'local request transport fixture')
            runner = Mock()
            with InferenceSession(loader=lambda: object(), runner=runner, disposer=lambda model: None) as session:
                arguments = ['--input', str(source), '--output_dir', str(root / 'out'),
                             '--hann_chunk', '8', '--hann_stride', '4', '--output_format', 'exr',
                             '--temp_map_budget_bytes', str(2**63 - 1)]
                infer.main(arguments, session=session)
                config = runner.call_args.args[2]
                self.assertEqual(config.temp_map_budget_bytes, 2**63 - 1)
                self.assertEqual((config.hann_chunk, config.hann_stride), (8, 4))
                self.assertFalse(config.low_vram)
                self.assertFalse(config.quick_key)
                runner.reset_mock()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    infer.main(arguments[:-1] + ['-1'], session=session)
                runner.assert_not_called()
            self.assertEqual(source.read_bytes(), b'local request transport fixture')

    def test_batch_retains_independent_ceilings_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source.mp4').write_bytes(b'local request transport fixture')
            manifest = root / 'batch.json'
            data = {'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': 'source.mp4', 'output_dir': 'legacy'},
                {'input': 'source.mp4', 'output_dir': 'zero', 'settings': {'temp_map_budget_bytes': 0}},
                {'input': 'source.mp4', 'output_dir': 'large', 'settings': {'temp_map_budget_bytes': 2**63 - 1}},
            ]}
            manifest.write_text(json.dumps(data))
            jobs, _ = load_jobs(manifest)
            self.assertEqual([job.configurations()[0].temp_map_budget_bytes for job in jobs],
                             [None, 0, 2**63 - 1])
            data['jobs'][1]['settings']['temp_map_budget_bytes'] = -1
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'temp_map_budget_bytes'):
                load_jobs(manifest)
