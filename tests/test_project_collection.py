"""Pinned bounded local I/O; synthetic bytes only."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.control import CancelledError
from corridorkey.project_collection import (
    admit_file,
    assert_pin,
    read_file,
    safe_name,
)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'frame.png'
        self.path.write_bytes(b'abc' * 100)

    def tearDown(self):
        self.temp.cleanup()

    def test_bounded_copy_exclusive_and_source_preserved(self):
        pin = admit_file(self.path)
        target = self.root / 'copy.png'
        digest, data = read_file(pin, capture=True, destination=target)
        self.assertEqual(data, self.path.read_bytes())
        self.assertEqual(target.read_bytes(), data)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(FileExistsError):
            read_file(pin, destination=target)
        self.assertEqual(self.path.read_bytes(), b'abc' * 100)

    def test_symlink_hardlink_and_ancestor_alias_refused(self):
        link = self.root / 'link.png'
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            admit_file(link)
        folder = self.root / 'alias'
        folder.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            admit_file(folder / self.path.name)
        os.link(self.path, self.root / 'hard.png')
        with self.assertRaises(ValueError):
            admit_file(self.path)

    def test_same_size_mutation_and_inode_replacement_fenced(self):
        pin = admit_file(self.path)
        self.path.write_bytes(b'x' * pin.size)
        with self.assertRaises(ValueError):
            assert_pin(pin)
        pin = admit_file(self.path)
        replacement = self.root / 'replace'
        replacement.write_bytes(b'x' * pin.size)
        replacement.replace(self.path)
        with self.assertRaises(ValueError):
            read_file(pin)

    def test_growth_during_fd_read_and_cancel_are_refused(self):
        pin = admit_file(self.path)
        original = os.read
        grown = False
        def grow(fd, size):
            nonlocal grown
            block = original(fd, size)
            if not grown:
                grown = True
                with self.path.open('ab') as stream:
                    stream.write(b'new')
            return block
        with patch('corridorkey.project_collection.os.read', side_effect=grow), self.assertRaises(ValueError):
            read_file(pin)
        def cancel():
            raise CancelledError('test')
        with self.assertRaises(CancelledError):
            read_file(admit_file(self.path), check=cancel)

    def test_nonportable_names(self):
        for name in ('../escape', 'x/y', r'x\y', 'C:evil', '//UNC', '.', '..', 'NUL.png', 'file.', 'trailing '):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)
