"""Path and descriptor timestamps are distinct clock domains on Windows."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from corridorkey.identities import verify_file_observation


class FileObservationTests(unittest.TestCase):
    def test_windows_path_birthtime_does_not_replace_descriptor_change_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'abc')
            os.utime(path, ns=(1000000000, 1000000000))
            original = Path.stat
            real = original(path)
            values = {name: getattr(real, name) for name in dir(real) if name.startswith('st_')}
            values['st_ctime_ns'] -= 1000000000
            path_info = SimpleNamespace(**values)
            with path.open('rb') as stream:
                descriptor = os.fstat(stream.fileno())
                with patch.object(sys, 'platform', 'win32'), patch.object(Path, 'stat', return_value=path_info):
                    self.assertIs(verify_file_observation(path, path_info, descriptor, descriptor), path_info)
                    for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'):
                        changed = {name: getattr(descriptor, name) for name in dir(descriptor) if name.startswith('st_')}
                        changed[field] += 1
                        with self.subTest(field=field), self.assertRaises(ValueError):
                            verify_file_observation(path, path_info, descriptor, SimpleNamespace(**changed))

    def test_path_changes_remain_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'abc')
            before = path.stat()
            with path.open('rb') as stream:
                descriptor = os.fstat(stream.fileno())
                for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'):
                    changed = {name: getattr(before, name) for name in dir(before) if name.startswith('st_')}
                    changed[field] += 1
                    with self.subTest(field=field), patch.object(Path, 'stat', return_value=SimpleNamespace(**changed)), self.assertRaises(ValueError):
                        verify_file_observation(path, before, descriptor, descriptor)

    def test_windows_reopen_keeps_every_descriptor_field_and_final_path_fence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'abc')
            before = path.stat()
            with path.open('rb') as stream:
                descriptor = os.fstat(stream.fileno())
                for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink'):
                    changed = {name: getattr(descriptor, name) for name in dir(descriptor) if name.startswith('st_')}
                    changed[field] += 1
                    with (self.subTest(field=field), patch.object(sys, 'platform', 'win32'),
                          patch('corridorkey.identities.os.fstat', return_value=SimpleNamespace(**changed)),
                          self.assertRaises(ValueError)):
                        verify_file_observation(path, before, descriptor, descriptor)
                changed = {name: getattr(before, name) for name in dir(before) if name.startswith('st_')}
                changed['st_ctime_ns'] += 1
                with (patch.object(sys, 'platform', 'win32'),
                      patch.object(Path, 'stat', side_effect=[before, SimpleNamespace(**changed)]),
                      self.assertRaises(ValueError)):
                    verify_file_observation(path, before, descriptor, descriptor)

    def test_helper_preserves_callers_hardlink_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'abc')
            os.link(path, Path(folder) / 'other-name')
            before = path.stat()
            self.assertEqual(before.st_nlink, 2)
            with path.open('rb') as stream, patch.object(sys, 'platform', 'win32'):
                descriptor = os.fstat(stream.fileno())
                self.assertEqual(verify_file_observation(path, before, descriptor, descriptor), before)
