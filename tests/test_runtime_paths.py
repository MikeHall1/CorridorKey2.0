"""Asset selection binds real engine paths without relocating bundled code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.control import CancellationToken
from corridorkey.runtime_paths import RuntimePaths
from corridorkey.session import JobRequest
from corridorkey.worker_core import SharedEngine
from corridorkey.worker_protocol import (
    decode_frame,
    job_to_wire,
    parse_request,
    request_frame,
)


class RuntimePathsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.code = self.root / 'code'
        self.assets = self.root / 'assets'
        self.addCleanup(self.folder.cleanup)

    def test_separate_roots_and_legacy_preference_preserve_code_location(self):
        (self.assets / 'weights').mkdir(parents=True)
        legacy = self.assets / 'weights/corridorkey_v2.pt'
        legacy.write_bytes(b'fixture')
        paths = RuntimePaths.from_package(self.code, asset_root=self.assets)
        self.assertEqual(paths.weights, legacy)
        self.assertEqual(paths.package_root, self.code)
        preferred = self.assets / 'weights/corridorkey_v2.safetensors'
        preferred.write_bytes(b'fixture')
        self.assertEqual(RuntimePaths.from_package(self.code, asset_root=self.assets).weights, preferred)
        self.assertEqual(RuntimePaths.from_package(self.code).moge_checkpoint, self.code / 'vfms/moge/model.pt')

    def test_normal_session_uses_selected_assets(self):
        source = self.root / 'source'
        source.write_text('{}')
        job = JobRequest.create(source, self.root / 'outputs')
        request = parse_request(decode_frame(request_frame('normal', 'render', {'job': job_to_wire(job)})))
        engine = SharedEngine(package_root=self.code, asset_root=self.assets, weights='selected.safetensors')
        with patch('corridorkey.worker_core.InferenceSession') as session:
            session.return_value.run.return_value.to_dict.return_value = {'status': 'complete'}
            self.assertEqual(engine.run(request, CancellationToken(), lambda event: None)['status'], 'complete')
            session.assert_called_once_with(weights='selected.safetensors', package_root=self.code, asset_root=self.assets)
        engine.close()

    def test_recovery_binds_default_weights_assets_and_fixed_implementation(self):
        source = self.root / 'source'
        source.write_text('{}')
        manifest = self.root / 'batch.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': str(source), 'output_dir': str(self.root / 'outputs'), 'settings': {'output_format': 'exr'}}]}))
        override = self.root / 'override.safetensors'
        override.write_bytes(b'fixture')
        paths = RuntimePaths.from_package(self.code, override, asset_root=self.assets)
        for name in ('foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
            path = getattr(paths, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
        paths.cradio_dir.mkdir(parents=True)
        payload = {'batch': str(manifest), 'workspace': str(self.root / 'cache'), 'max_cache_bytes': 0}
        request = parse_request(decode_frame(request_frame('recovery', 'recover', payload)))
        engine = SharedEngine(package_root=self.code, asset_root=self.assets, weights=override)
        with patch('corridorkey.recovery.cuda_environment', return_value={'fixture': True}), patch('corridorkey.recovery.run_recoverable') as run:
            run.return_value = {'status': 'complete', 'report_path': 'fixture', 'journal': 'fixture', 'jobs': []}
            engine.run(request, CancellationToken(), lambda event: None)
            args = run.call_args.kwargs
            self.assertEqual(args['assets']['weights'], override)
            self.assertEqual(args['assets']['moge_checkpoint'], self.assets / 'vfms/moge/model.pt')
            self.assertEqual(args['implementation_root'], self.code)
            self.assertEqual(args['session_factory'].keywords['asset_root'], self.assets)

    def test_actual_worker_receives_selected_assets_and_reports_missing_paths(self):
        from corridorkey.worker_client import WorkerClient
        source = self.root / 'source'
        source.write_text('{}')
        batch = self.root / 'batch.json'
        batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': str(source), 'output_dir': str(self.root / 'outputs'), 'settings': {'output_format': 'exr'}}]}))
        client = WorkerClient(asset_root=self.assets, weights=self.root / 'selected.safetensors')
        try:
            self.assertEqual(client.command[0], sys.executable)
            identifier = client.send('recover', {'batch': str(batch), 'workspace': str(self.root / 'cache'), 'max_cache_bytes': 0})
            while True:
                event = client.next_event(10)
                if event.get('request_id') == identifier and event['kind'] in {'failed', 'lost', 'rejected', 'result'}:
                    break
            self.assertEqual(event['kind'], 'failed')
            message = event['payload']['error']['message']
            self.assertIn(str(self.assets / 'vfms/moge/model.pt'), message)
            self.assertIn(str(self.root / 'selected.safetensors'), message)
            self.assertIn('Local assets → Download model assets', message)
            self.assertIn('python -m corridorkey.asset_download plan', message)
            self.assertIn('scripts/download_weights.py', message)
            self.assertTrue(client.close())
        finally:
            client.close()
