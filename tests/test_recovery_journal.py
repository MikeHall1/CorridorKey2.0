import json
import tempfile
import unittest
from pathlib import Path

from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_journal import AttemptJournal, export_paths, inventory
from corridorkey.recovery_store import BundleStore, CacheInvalid


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.cache = self.root / 'cache'
        self.identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'journal'}))
        self.output = self.root / 'render'
        folder = self.output / 'alpha'
        folder.mkdir(parents=True)
        (folder / 'alpha_000000.exr').write_bytes(b'export integrity fixture, no neural inference')
        self.record = {'status': 'complete', 'output_dir': str(self.output), 'diagnostics': {
            'frames_submitted': 1, 'outputs': {'alpha': str(folder)}, 'output_contracts': {'alpha': {'format': 'exr'}}}}

    def test_only_verified_completed_attempts_are_recovered(self):
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'])
            attempt = journal.start(0, self.identity, self.output, 'render')
            self.assertIsNone(journal.completed(0, self.identity))
            artifacts = inventory(export_paths(self.record))
            journal.finish(0, attempt, status='complete', artifacts=artifacts, origin_record=self.record)
            completed = journal.completed(0, self.identity)
            assert completed is not None
            self.assertEqual(completed['output_dir'], str(self.output.resolve()))
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'], resume=True)
            completed = journal.completed(0, self.identity)
            assert completed is not None
            self.assertEqual(completed['status'], 'complete')
            (self.output / 'alpha/alpha_000000.exr').write_bytes(b'corrupt')
            self.assertIsNone(journal.completed(0, self.identity))

    def test_interrupted_attempt_is_reported_without_trusting_partial_outputs(self):
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'])
            journal.start(0, self.identity, self.output, 'render')
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'], resume=True)
            self.assertEqual(journal.snapshot()['jobs'][0][-1]['status'], 'interrupted')
            self.assertIsNone(journal.completed(0, self.identity))
            self.assertTrue((self.output / 'alpha/alpha_000000.exr').exists())

    def test_existing_journal_requires_resume_and_metadata_corruption_is_rejected(self):
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'])
            journal.start(0, self.identity, self.output, 'render')
            with self.assertRaisesRegex(ValueError, 'explicit resume'):
                AttemptJournal(store, ['request'])
            data = json.loads(journal.path.read_text())
            data['payload']['jobs'][0][0]['status'] = 'complete'
            journal.path.write_text(json.dumps(data))
            with self.assertRaisesRegex(CacheInvalid, 'checksum'):
                AttemptJournal(store, ['request'], resume=True)

    def test_snapshot_ownership_and_output_completeness(self):
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'])
            journal.snapshot()['jobs'][0].append({'status': 'complete'})
            self.assertEqual(journal.snapshot()['jobs'], [[]])
        self.record['diagnostics']['frames_submitted'] = 2
        with self.assertRaisesRegex(ValueError, 'every submitted frame'):
            export_paths(self.record)
        self.record['diagnostics']['frames_submitted'] = 1
        (self.output / 'alpha/user-footage.png').write_bytes(b'preserve')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            export_paths(self.record)

    def test_completion_cannot_seal_missing_or_changed_artifacts(self):
        with BundleStore(self.cache, 10000) as store:
            journal = AttemptJournal(store, ['request'])
            attempt = journal.start(0, self.identity, self.output, 'render')
            artifacts = inventory(export_paths(self.record))
            (self.output / 'alpha/alpha_000000.exr').write_bytes(b'changed')
            with self.assertRaisesRegex(CacheInvalid, 'unverified'):
                journal.finish(0, attempt, status='complete', artifacts=artifacts, origin_record=self.record)
            self.assertEqual(journal.snapshot()['jobs'][0][-1]['status'], 'running')
