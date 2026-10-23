import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.ofx_cache_task import CachePreparation
from tests.test_ofx_seal import exports


class CachePreparationTests(unittest.TestCase):
    def finish(self, task):
        self.assertTrue(task.done.wait(15))
        task.thread.join(5)
        self.assertFalse(task.thread.is_alive())
        result = task.results.get_nowait()
        self.assertFalse(task.busy)
        return result

    def test_actual_immutable_record_and_bounded_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, record = exports(root)
            task = CachePreparation(record, root)
            record['status'] = 'failed'
            result = self.finish(task)
            self.assertEqual(result.status, 'complete', result.error)
            assert result.cache is not None and result.workspace is not None
            self.assertTrue(result.cache.is_file())
            saved = json.loads((result.workspace / 'completed-record.json').read_text())
            self.assertEqual(saved['status'], 'complete')
            progress = task.progress()
            assert progress is not None
            self.assertEqual(progress.stage, 'ofx_publish')
            self.assertEqual(task.results.maxsize, 1)

    def test_cancel_while_worker_owned_never_publishes(self):
        from corridorkey.ofx_cache import seal
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture gate')
            return seal(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, record = exports(root)
            with patch('corridorkey.ofx_cache.seal', side_effect=delayed):
                task = CachePreparation(record, root)
                try:
                    self.assertTrue(entered.wait(5))
                    task.cancel()
                    self.assertTrue(task.busy)
                finally:
                    release.set()
                result = self.finish(task)
            self.assertEqual(result.status, 'cancelled')
            self.assertIsNone(result.cache)
            self.assertFalse(list(root.rglob('*.ckcache')))

    def test_import_does_not_load_numeric_engine_or_start_workers(self):
        code = "import sys,threading; import corridorkey.ofx_cache_task; print('torch' in sys.modules, len(threading.enumerate()))"
        result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.strip(), 'False 1')

    def test_unexpected_worker_exit_still_delivers_terminal_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, record = exports(root)
            with patch('corridorkey.ofx_cache.seal', side_effect=SystemExit('injected worker exit')), self.assertLogs(
                    'corridorkey.ofx_cache_task', level='ERROR'):
                result = self.finish(CachePreparation(record, root))
            self.assertEqual(result.status, 'failed')
            self.assertIn('SystemExit', result.error or '')
            self.assertFalse(list(root.rglob('*.ckcache')))

    def test_destination_resolution_occurs_on_owned_thread(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, record = exports(root)
            original = Path.resolve
            owners = []

            def observed(path, *args, **kwargs):
                if path == root:
                    owners.append(threading.get_ident())
                return original(path, *args, **kwargs)

            with patch.object(Path, 'resolve', observed):
                result = self.finish(CachePreparation(record, root))
            self.assertEqual(result.status, 'complete', result.error)
            self.assertTrue(owners)
            self.assertNotIn(threading.get_ident(), owners)
