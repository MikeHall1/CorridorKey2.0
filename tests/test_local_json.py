import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from corridorkey.local_json import read_object, read_regular_bytes
from tests.test_project_collection import windows_file_clocks


class RegularJsonTests(unittest.TestCase):
    def test_windows_clock_domains_regular_json_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selected.json'
            path.write_bytes(b'{"value":1}')
            os.utime(path, ns=(1000000000, 1000000000))
            with windows_file_clocks():
                self.assertEqual(read_object(path, max_bytes=128, regular_file=True), ({'value': 1}, b'{"value":1}'))

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_regular_file_admission_refuses_pipe_before_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selected.json'
            mkfifo = getattr(os, 'mkfifo', None)
            assert callable(mkfifo)
            mkfifo(path)
            with self.assertRaisesRegex(ValueError, 'regular file'):
                read_object(path, max_bytes=128, regular_file=True)

    def test_replacement_of_selected_path_during_open_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selected.json'
            replacement = Path(folder) / 'replacement.json'
            path.write_bytes(b'{"value":1}')
            replacement.write_bytes(path.read_bytes())
            original = os.fdopen
            @contextmanager
            def replace(fd, *args):
                with original(fd, *args) as stream:
                    os.replace(replacement, path)
                    yield stream
            with patch('corridorkey.local_json.os.fdopen', side_effect=replace), self.assertRaisesRegex(ValueError, 'changed'):
                read_regular_bytes(path, max_bytes=128)
            self.assertEqual(path.read_bytes(), b'{"value":1}')
