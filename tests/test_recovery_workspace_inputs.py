"""Pre-prune recovery admission preserves authored documents and their assets."""
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.corrections import CorrectionDocument, save_corrections
from corridorkey.identities import source_identity
from corridorkey.recovery import _separate_workspace, run_recoverable
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore
from corridorkey.session import JobRequest

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2'))


def png_mask(width=1):
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, 1, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\x00' + b'\xff' * width)) + chunk(b'IEND', b''))


class RecoveryWorkspaceInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'local source identity')
        self.mask = self.root / 'mask.png'
        self.mask.write_bytes(png_mask())
        self.workspace = self.root / 'workspace'
        self.identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'mask'}))
        with BundleStore(self.workspace, 100000) as store:
            store.publish(self.identity, {'mask.png': self.mask}, {})
            self.owner = store.owner
        self.payload = self.workspace / 'entries' / self.identity.key / 'payload-000000.bin'

    def document(self, mask, path):
        data = {'schema_version': 1, 'revision': 1,
                'source': {'sha256': source_identity(self.source)['sha256'], 'start_frame': 0,
                           'frame_count': 1, 'fps': 24., 'width': 1, 'height': 1},
                'shots': [[0, 1]], 'entries': [
                    {'id': 'mask', 'revision': 1, 'frame': 0, 'kind': 'matte',
                     'asset': {'path': str(mask), 'sha256': hashlib.sha256(mask.read_bytes()).hexdigest()},
                     'strength': 1., 'direction': 'both', 'range': [0, 1]}]}
        save_corrections(path, CorrectionDocument.parse(data, path.parent))
        return path

    def job(self, document):
        return JobRequest.create(self.source, self.root / 'render', InferenceConfig(
            outputs={'alpha'}, output_format='exr', auto_vram=False, corrections_path=str(document)))

    def snapshot(self):
        return {str(path.relative_to(self.workspace)): path.read_bytes() if path.is_file() else None
                for path in self.workspace.rglob('*')}

    def refuse_before_store(self, job):
        before = self.snapshot()
        with patch('corridorkey.recovery.BundleStore', side_effect=AssertionError('store constructed before admission')), self.assertRaisesRegex(ValueError, 'correction dependencies'):
            run_recoverable([job], workspace=self.workspace, max_cache_bytes=0, assets={}, environment={},
                            session_factory=lambda **kwargs: self.fail('engine/session started'))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.workspace / 'journals').exists())
        self.assertFalse((self.workspace / 'reports').exists())

    def test_sealed_mask_and_owned_staging_document_refuse_before_prune(self):
        document = self.document(self.payload, self.root / 'external.json')
        self.refuse_before_store(self.job(document))
        stage = self.workspace / 'staging' / f'entry-{self.owner}-protected'
        stage.mkdir()
        staged_document = self.document(self.mask, stage / 'payload-000000.bin')
        self.refuse_before_store(self.job(staged_document))
        self.assertEqual(self.payload.read_bytes(), png_mask())
        self.assertTrue(staged_document.exists())

    def test_asset_and_workspace_aliases_refuse_without_replacing_originals(self):
        alias = self.root / 'mask-alias.png'
        try:
            alias.symlink_to(self.payload)
            root_alias = self.root / 'workspace-alias'
            root_alias.symlink_to(self.workspace, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f'local symlink creation unavailable: {exc}')
        document = self.document(alias, self.root / 'aliases.json')
        self.refuse_before_store(self.job(document))
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'correction dependencies'):
            _separate_workspace(root_alias, [self.job(document)], {})
        self.assertEqual(self.snapshot(), before)

    def test_cancel_and_oversized_document_fail_without_any_workspace_change(self):
        document = self.document(self.mask, self.root / 'external.json')
        before = self.snapshot()
        def cancel():
            raise CancelledError('stop during admission')
        with self.assertRaises(CancelledError):
            _separate_workspace(self.workspace, [self.job(document)], {}, check=cancel)
        with patch('corridorkey.corrections.MAX_CORRECTION_BYTES', 8), self.assertRaisesRegex(ValueError, 'byte limit'):
            _separate_workspace(self.workspace, [self.job(document)], {})
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(STACK, 'requires CPU EXR fixture; no pretrained inference')
    def test_unrelated_corrections_still_render_and_explicitly_resume(self):
        from tests.test_recovery_runner import RecoveryRunnerTests

        fixture = RecoveryRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.folder.cleanup)
        # Its independent CPU renderer deliberately does not interpret masks;
        # this proves admission/preflight/recovery identity remain coherent.
        document = self.root / 'valid-external.json'
        valid_mask = self.root / 'two-pixel-mask.png'
        valid_mask.write_bytes(png_mask(width=2))
        data = json.loads(self.document(valid_mask, document).read_text())
        data['source'].update(sha256=source_identity(fixture.source)['sha256'], width=2, height=1, frame_count=2)
        data['shots'] = [[0, 2]]
        data['entries'][0]['range'] = [0, 2]
        save_corrections(document, CorrectionDocument.parse(data, self.root))
        cfg, writer = fixture.jobs[0].configurations()
        cfg.corrections_path = str(document)
        job = JobRequest.create(fixture.source, fixture.root / 'corrected-output', cfg, writer)
        first = fixture.run_jobs(jobs=[job])
        self.assertEqual(first['status'], 'complete')
        second = fixture.run_jobs(jobs=[job], resume=True)
        self.assertEqual(second['status'], 'complete')
        self.assertEqual(second['jobs'][0]['mode'], 'verified_existing')
        self.assertEqual(fixture.loads, 1)
