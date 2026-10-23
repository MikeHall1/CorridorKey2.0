"""Launcher inventories must not reinterpret interrupted scans as no hints."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_interactive as launcher


class InteractiveInventoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.sequence = self.root / 'b-sequence'
        self.sequence.mkdir()
        self.inputs = self.sequence / 'input'
        self.inputs.mkdir()
        self.hints = self.sequence / 'AlphaHint'
        self.hints.mkdir()
        self.frame = self.inputs / 'frame001.PNG'
        self.frame.write_bytes(b'inventory fixture')

    def interrupted_scan(self, target, first):
        original = Path.iterdir

        def scan(path):
            if path == target:
                yield first
                raise PermissionError('directory inventory interrupted')
            yield from original(path)

        return patch.object(Path, 'iterdir', scan)

    def test_partial_sequence_folder_scan_cannot_silently_drop_companion_hints(self):
        with self.interrupted_scan(self.sequence, self.inputs), \
                self.assertRaisesRegex(PermissionError, 'inventory interrupted'):
            launcher.resolve_image_seq_dirs(self.sequence)

    def test_partial_frame_scan_is_not_accepted_after_first_image(self):
        with self.interrupted_scan(self.inputs, self.frame), \
                self.assertRaisesRegex(PermissionError, 'inventory interrupted'):
            launcher.check_image_sequence_dir(self.sequence)

    def test_batch_refuses_partial_clip_list_after_nested_scan_error(self):
        (self.root / 'a-clip.mov').write_bytes(b'video inventory fixture')
        with self.interrupted_scan(self.sequence, self.inputs), \
                self.assertRaisesRegex(PermissionError, 'inventory interrupted'):
            launcher.get_batch_clips(self.root)

    def test_complete_inventory_preserves_case_insensitive_hints_and_batch_order(self):
        clip = self.root / 'a-clip.mov'
        clip.write_bytes(b'video inventory fixture')
        (self.root / 'a-clip_alphahint.mov').write_bytes(b'companion excluded from batch')
        self.assertEqual(launcher.resolve_image_seq_dirs(self.sequence), (self.inputs, self.hints))
        self.assertEqual(launcher.check_image_sequence_dir(self.sequence), (self.inputs, self.hints))
        self.assertEqual(launcher.get_batch_clips(self.root), [clip, self.sequence])
        self.assertEqual(launcher.resolve_image_seq_dirs(clip), (None, None))
        self.assertEqual(launcher.get_batch_clips(clip), [])
        self.assertEqual(launcher.get_batch_clips(self.root / 'missing'), [])

    def test_complete_empty_sequence_and_directory_are_not_errors(self):
        self.frame.unlink()
        self.assertEqual(launcher.check_image_sequence_dir(self.sequence), (None, None))
        self.assertEqual(launcher.get_batch_clips(self.inputs), [])
