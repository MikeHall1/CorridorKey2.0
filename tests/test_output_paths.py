"""Read-only output safety without image, model or Qt dependencies."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.output_paths import output_layout, raw_clip_name, validate_output_paths


class OutputPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'plate.mov'
        self.source.write_bytes(b'preserve source')
        self.output = self.root / 'render'
        self.cfg = InferenceConfig(outputs={'alpha', 'fg', 'checker', 'cutout'}, output_format='exr')

    def leaf(self, role='alpha', index=0):
        planned = output_layout(self.source, self.cfg)['outputs'][role]
        directory = self.output / planned['path']
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{planned['frame_name']}_{index:06d}.exr"

    def guard(self, **kwargs):
        validate_output_paths(self.source, self.output, self.cfg, **kwargs)

    def test_layout_all_suffixes_metadata_and_native_passthrough(self):
        self.assertEqual(raw_clip_name('/shot/input'), 'shot')
        self.assertEqual(raw_clip_name('/shot/FRAMES'), 'shot')
        self.assertEqual(raw_clip_name('/shot/rgb'), 'shot')
        self.cfg.quick_key, self.cfg.quick_interval = True, 3
        self.cfg.quick_auto_cuts = True
        self.cfg.model_scale = .5
        self.cfg.cutout_alpha_mode = 'premultiplied'
        self.cfg.despill_strength = .53
        result = output_layout(self.source, self.cfg)
        self.assertEqual(result['outputs']['cutout'], {
            'path': 'plate_cutout_premultiplied_source_rgb_draft_spatial_draft_exr',
            'frame_name': 'cutout_premultiplied_source_rgb_draft_spatial_draft', 'is_directory': True})
        self.assertEqual(result['outputs']['checker']['frame_name'], 'checker_comp_despill053_source_rgb_draft_spatial_draft')
        self.assertEqual(result['metadata'], ('plate_settings.txt', 'plate_quick_key_frames.jsonl', 'plate_shot_analysis.jsonl'))
        self.cfg.quick_interval = 1
        self.cfg.model_scale = 1
        self.cfg.output_format = 'video'
        result = output_layout(self.source, self.cfg)
        self.assertEqual(result['outputs']['alpha']['path'], 'plate_alpha.mp4')
        self.assertEqual(result['outputs']['cutout']['path'], 'plate_cutout_premultiplied.mov')
        self.assertEqual(result['metadata'], ('plate_settings.txt',))

    def test_generated_hint_file_and_same_inode_alias_refused_without_changes(self):
        hint = self.leaf()
        hint.write_bytes(b'preserve hint')
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.guard(initial_hint_path=hint)
        self.assertEqual(hint.read_bytes(), b'preserve hint')
        hint.unlink()
        os.link(self.source, hint)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            self.guard()
        self.assertEqual(self.source.read_bytes(), b'preserve source')
        self.assertEqual(hint.read_bytes(), b'preserve source')

    def test_case_variant_generated_leaf_hardlink_is_protected(self):
        lower = self.leaf()
        upper = lower.with_name(lower.name.upper())
        os.link(self.source, upper)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            self.guard()
        self.assertEqual(self.source.read_bytes(), b'preserve source')
        self.assertEqual(upper.read_bytes(), b'preserve source')

    def test_protected_directory_member_hardlink_and_nested_output_refused(self):
        frames = self.root / 'hints'
        frames.mkdir()
        member = frames / 'mask.exr'
        member.write_bytes(b'preserve hint member')
        alias = self.leaf(index=1000000)
        os.link(member, alias)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            self.guard(hint_video_path=frames)
        with self.assertRaisesRegex(ValueError, 'protected'):
            validate_output_paths(frames, frames / 'new-output', self.cfg)
        self.assertFalse((frames / 'new-output').exists())
        self.assertEqual(member.read_bytes(), b'preserve hint member')

    def test_source_directory_hardlinks_protected_but_unrelated_rerender_allowed(self):
        source = self.root / 'frames'
        source.mkdir()
        member = source / '0001.png'
        member.write_bytes(b'frame')
        layout = output_layout(source, self.cfg)['outputs']['alpha']
        leaf = self.output / layout['path']
        leaf.mkdir(parents=True)
        target = leaf / (layout['frame_name'] + '_000000.exr')
        os.link(member, target)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            validate_output_paths(source, self.output, self.cfg)
        target.unlink()
        target.write_bytes(b'old output')
        before = {path: path.read_bytes() for path in (member, target, self.source)}
        validate_output_paths(source, self.output, self.cfg)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_existing_directory_identity_closes_lexical_alias_containment(self):
        frames = self.root / 'native-frames'
        frames.mkdir()
        nested = frames / 'new-render'
        # Simulate a spelling/normalization mismatch at the lexical boundary;
        # real filesystem directory identity must still close the write path.
        with patch('corridorkey.output_paths._within', return_value=False), self.assertRaisesRegex(ValueError, 'ancestry'):
            validate_output_paths(frames, nested, self.cfg)
        self.assertFalse(nested.exists())
        unrelated = self.root / 'unrelated-render'
        validate_output_paths(frames, unrelated, self.cfg)
        self.assertFalse(unrelated.exists())

    def test_video_and_metadata_targets_protected(self):
        self.cfg.output_format = 'video'
        self.output.mkdir()
        target = self.output / 'plate_alpha.mp4'
        target.write_bytes(b'video hint')
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.guard(hint_video_path=target)
        settings = self.output / 'plate_settings.txt'
        settings.write_bytes(b'first hint')
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.guard(initial_hint_path=settings)

    def test_correction_document_and_assets_protected_without_decoding(self):
        asset = self.leaf()
        asset.write_bytes(b'authored asset bytes need no decoder')
        document = self.root / 'corrections.json'
        data = {'schema_version': 1, 'revision': 1,
                'source': {'sha256': '0' * 64, 'start_frame': 0, 'frame_count': 1,
                           'fps': 24., 'width': 1, 'height': 1}, 'shots': [[0, 1]],
                'entries': [{'id': 'mask', 'revision': 1, 'frame': 0, 'kind': 'matte',
                             'asset': {'path': str(asset), 'sha256': '1' * 64}, 'strength': 1.,
                             'direction': 'both', 'range': [0, 1]}]}
        document.write_text(json.dumps(data))
        self.cfg.corrections_path = str(document)
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.guard()
        self.assertEqual(asset.read_bytes(), b'authored asset bytes need no decoder')
        settings = self.output / 'plate_settings.txt'
        settings.write_text(json.dumps(data))
        self.cfg.corrections_path = str(settings)
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.guard()
        self.assertEqual(json.loads(settings.read_text()), data)

    def test_symlink_leaf_and_root_refused_and_bounds_cancellation_leave_files(self):
        target = self.leaf()
        try:
            target.symlink_to(self.source)
        except OSError as exc:
            self.skipTest(f'local symlink creation unavailable: {exc}')
        with self.assertRaises(ValueError):
            self.guard()
        target.unlink()
        target.write_bytes(b'old export')
        with patch('corridorkey.output_paths.MAX_SCAN_ENTRIES', 0), self.assertRaisesRegex(ValueError, 'entry limit'):
            self.guard()
        def cancel():
            raise CancelledError('explicit stop')
        with self.assertRaises(CancelledError):
            self.guard(check=cancel)
        self.assertEqual(target.read_bytes(), b'old export')
        linked = self.root / 'linked-output'
        linked.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real directory'):
            validate_output_paths(self.source, linked, self.cfg)
