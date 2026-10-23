"""Actual process/disk sampling and explicit noninitializing GPU contracts."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from corridorkey.resource_snapshot import disk_snapshot, process_snapshot
from corridorkey.worker_client import WorkerClient


class ResourceSnapshotTests(unittest.TestCase):
    def test_no_torch_import_or_cuda_initialization_for_status(self):
        with patch.dict(sys.modules, {'torch': None}):
            snapshot = process_snapshot()
        self.assertEqual(snapshot['pid'], os.getpid())
        self.assertIsNone(snapshot['gpu']['allocated_bytes'])
        self.assertIn('NOT VERIFIED', snapshot['gpu']['scope'])
        cuda = Mock()
        cuda.is_initialized.return_value = False
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            self.assertEqual(process_snapshot()['gpu']['state'], 'not initialized')
        cuda.current_device.assert_not_called()
        cuda.memory_allocated.assert_not_called()

    def test_initialized_allocator_sampling_and_error_preserve_scope(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.current_device.return_value = 0
        cuda.memory_allocated.return_value = 17
        cuda.memory_reserved.return_value = 32
        cuda.max_memory_allocated.return_value = 23
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            gpu = process_snapshot()['gpu']
            self.assertEqual((gpu['allocated_bytes'], gpu['reserved_bytes'], gpu['peak_allocated_bytes']), (17, 32, 23))
            self.assertIn('not total GPU', gpu['scope'])
            cuda.memory_reserved.side_effect = RuntimeError('fixture failure')
            self.assertIsNone(process_snapshot()['gpu']['reserved_bytes'])

    def test_disk_measurement_uses_existing_ancestor_without_creating_requested_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder).resolve()
            requested = parent / 'not-created' / 'output'
            snapshot = disk_snapshot(requested)
            self.assertEqual(snapshot['measured_at'], str(parent))
            self.assertGreater(snapshot['total_bytes'], 0)
            self.assertGreaterEqual(snapshot['free_bytes'], 0)
            self.assertFalse(requested.parent.exists())

    def test_actual_owned_worker_status_contains_its_process_snapshot_without_rendering(self):
        client = WorkerClient()
        try:
            request_id = client.send('status', {})
            while True:
                event = client.next_event(10)
                if event['kind'] == 'status' and event.get('request_id') == request_id:
                    break
            resources = event['payload']['resources']
            self.assertEqual(resources['pid'], client.process.pid)
            self.assertIsNone(event['payload']['active'])
            self.assertEqual(event['payload']['pending'], [])
            self.assertIsNone(resources['gpu']['allocated_bytes'])
            self.assertTrue(client.close())
        finally:
            client.close()

    def test_partial_torch_import_and_failed_initialization_probe_are_unavailable(self):
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace())}):
            self.assertEqual(process_snapshot()['gpu']['state'], 'unavailable')
        cuda = Mock()
        cuda.is_initialized.side_effect = RuntimeError('probe unavailable')
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            self.assertIsNone(process_snapshot()['gpu']['allocated_bytes'])
        cuda.current_device.assert_not_called()
