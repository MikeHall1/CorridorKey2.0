"""Standalone diagnostic recipe tests; no model/GPU/Qt claims."""
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.diagnostic_ranges import (
    candidates,
    inspect_diagnostics,
    prepare_range,
    revalidate,
)
from corridorkey.diagnostics import json_value
from corridorkey.session import JobRequest


def make_diagnostic_fixture(root):
    root = Path(root).resolve()
    source, output, parent = root / 'source', root / 'old-output', root / 'new-parent'
    for path in (source, output, parent):
        path.mkdir()
    for frame in range(7):
        (source / f'{frame:04}.png').write_bytes(bytes([frame]) * 3)
    cfg = InferenceConfig(start_frame=2, num_frames=5, quick_key=True, quick_interval=2,
                          quick_cut_frames=[1, 3], quick_reset_frames=[2, 4])
    job = JobRequest.create(source, output, cfg)
    quick = output / 'quick.jsonl'
    quick.write_text(''.join(json.dumps({'frame': frame, 'event': 'rendered_before_writer_submission',
                                        'source_rgb_draft': True, 'anchors': [0, 4],
                                        'look_ahead_frames': 4 - frame, 'fallback': 'neural' if frame == 1 else None,
                                        'uncertain_fraction': 0.25 if frame == 1 else 0.0,
                                        'reliability_calibrated': False}) + '\n' for frame in range(3)))
    shot = output / 'shot.jsonl'
    shot.write_text(''.join(json.dumps({'frame': frame, 'cut': frame == 3, 'score': .5,
                                       'mean_rgb_change': .2, 'histogram_distance': .3, 'calibrated': False}) + '\n'
                            for frame in range(5)))
    cfg, writer = job.configurations()
    record = {'status': 'cancelled', 'input': job.input_path, 'output_dir': job.output_dir,
              'requested_settings': json_value(cfg), 'writer_settings': json_value(writer),
              'hints': {'first_frame': None, 'full': None},
              'diagnostics': {'source': {'start_frame': 2, 'num_frames': 5, 'native_hw': [64, 64],
                                         'native_fps': 24.0, 'exr_linear': False},
                              'quick_key': {'frame_diagnostics_path': str(quick), 'fallback_frames': [1, 4],
                                            'extra_neural_frames': [1], 'analysis': {'trace_path': str(shot),
                                                                                   'detected_cuts': [3]}}}}
    return job, record, parent


class DiagnosticRangesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.job, self.record, self.parent = make_diagnostic_fixture(self.root)

    def snapshot(self):
        return inspect_diagnostics(self.job, self.record)

    def test_absolute_events_and_truthful_partial_summary_coverage(self):
        snapshot = self.snapshot()
        data = snapshot.to_dict()
        self.assertEqual(data['coverage'], {'expected_frames': 5, 'quick_observed_frames': 3,
                                          'shot_observed_frames': 5, 'summary_only_events': 1})
        self.assertFalse(data['historical_source_identity_verified'])
        self.assertEqual([row['frame'] for row in data['events']], [3, 5, 6])
        self.assertFalse(data['events'][-1]['observed'])
        ranges = candidates(snapshot)
        self.assertEqual([(v.start_frame, v.end_frame) for v in ranges], [(3, 4), (5, 7)])
        self.assertEqual([(v.start_frame, v.end_frame) for v in candidates(snapshot, padding_frames=1)], [(2, 7)])
        self.assertEqual(ranges[-1].observed_frames, 1)
        data['source']['start_frame'] = 999
        self.assertEqual(snapshot.to_dict()['source']['start_frame'], 2)

    def test_prepares_fresh_full_quality_no_submission_or_prior_changes(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        events = []
        snapshot = inspect_diagnostics(self.job, self.record, progress=events.append)
        prepared = prepare_range(snapshot, 3, 6, self.parent)
        cfg, _writer = prepared.configurations()
        self.assertEqual((cfg.start_frame, cfg.num_frames, cfg.start_mode), (3, 3, 'begin'))
        self.assertFalse(cfg.quick_key)
        self.assertFalse(cfg.quick_auto_cuts)
        self.assertEqual(cfg.quick_cut_frames, [])
        self.assertEqual(cfg.quick_reset_frames, [])
        self.assertFalse(Path(prepared.output_dir).exists())
        self.assertFalse(cfg.temp_dir.exists())
        self.assertTrue(Path(prepared.output_dir).parent.is_dir())
        self.assertEqual(len(list(self.parent.iterdir())), 1)
        self.assertEqual(prepared.writer_json, self.job.writer_json)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(events[0].stage, 'diagnostic_ranges')
        other = prepare_range(snapshot, 3, 6, self.parent)
        self.assertNotEqual(other.output_dir, prepared.output_dir)

    def test_preserve_mode_rebases_selected_relative_cuts_resets(self):
        prepared = prepare_range(self.snapshot(), 3, 6, self.parent, full_quality=False)
        cfg, _ = prepared.configurations()
        self.assertTrue(cfg.quick_key)
        self.assertEqual(cfg.quick_cut_frames, [0, 2])
        self.assertEqual(cfg.quick_reset_frames, [1])

    def test_initial_hint_explicit_drop_fullhint_preserved(self):
        first, full = self.root / 'first.png', self.root / 'full.mp4'
        first.write_bytes(b'first')
        full.write_bytes(b'full')
        self.job = replace(self.job, initial_hint_path=str(first), hint_video_path=str(full))
        self.record['hints'] = {'first_frame': str(first), 'full': str(full)}
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'First-frame'):
            prepare_range(snapshot, 3, 6, self.parent)
        same_start = prepare_range(snapshot, 2, 6, self.parent)
        self.assertEqual(same_start.initial_hint_path, str(first))
        prepared = prepare_range(snapshot, 3, 6, self.parent, initial_hint_policy='drop')
        self.assertIsNone(prepared.initial_hint_path)
        self.assertEqual(prepared.hint_video_path, str(full))

    def test_correction_document_requires_explicit_drop_on_range_change(self):
        from corridorkey.corrections import canonical
        document = {'schema_version': 1, 'revision': 1,
                    'source': {'sha256': '0' * 64, 'start_frame': 2, 'frame_count': 5,
                               'fps': 24.0, 'width': 64, 'height': 64},
                    'shots': [[0, 5]],
                    'entries': [{'id': 'reset', 'revision': 1, 'frame': 1, 'kind': 'reset'}]}
        path = self.root / 'corrections.json'
        path.write_text(canonical(document))
        cfg, writer = self.job.configurations()
        cfg = replace(cfg, corrections_path=str(path))
        self.job = JobRequest.create(self.job.input_path, self.job.output_dir, cfg, writer)
        self.record['requested_settings'] = json_value(cfg)
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'Correction document'):
            prepare_range(snapshot, 3, 6, self.parent)
        same = prepare_range(snapshot, 2, 7, self.parent)
        self.assertEqual(same.configurations()[0].corrections_path, str(path))
        changed = prepare_range(snapshot, 3, 6, self.parent, correction_policy='drop')
        self.assertIsNone(changed.configurations()[0].corrections_path)
        self.assertEqual(json.loads(path.read_text()), document)

    def test_late_cancel_does_not_hide_reserved_job(self):
        snapshot = self.snapshot()
        original = Path.mkdir
        committed = False
        def reserve(path, *args, **kwargs):
            nonlocal committed
            result = original(path, *args, **kwargs)
            committed = True
            return result
        def check():
            if committed:
                raise CancelledError('late cancellation')
        with patch('pathlib.Path.mkdir', reserve):
            job = prepare_range(snapshot, 3, 6, self.parent, check=check)
        self.assertTrue(Path(job.output_dir).parent.is_dir())
        self.assertTrue(committed)

    def test_snapshot_token_and_rehashed_shape_refuse(self):
        import hashlib

        from corridorkey.diagnostic_ranges import DiagnosticSnapshot
        snapshot = self.snapshot()
        data = snapshot.to_dict()
        data['source']['start_frame'] = 4
        with self.assertRaisesRegex(ValueError, 'token'):
            candidates(DiagnosticSnapshot(json.dumps(data)))
        unsigned = dict(data)
        unsigned.pop('token')
        data['token'] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'event schema'):
            candidates(DiagnosticSnapshot(json.dumps(data)))

    def test_source_colour_profile_preserved(self):
        cfg, writer = self.job.configurations()
        cfg = replace(cfg, frame_dir_linear=True, source_colour_profile='linear_bt2020_d65')
        self.job = JobRequest.create(self.job.input_path, self.job.output_dir, cfg, writer)
        self.record['requested_settings'] = json_value(cfg)
        self.record['diagnostics']['source']['exr_linear'] = True
        snapshot = self.snapshot()
        self.assertTrue(snapshot.to_dict()['source']['scene_linear'])
        prepared = prepare_range(snapshot, 3, 6, self.parent)
        self.assertEqual(prepared.configurations()[0].source_colour_profile, 'linear_bt2020_d65')

    def test_changed_inputs_or_trace_refuse_before_reservation(self):
        snapshot = self.snapshot()
        source = Path(self.job.input_path) / '0001.png'
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            prepare_range(snapshot, 3, 6, self.parent)
        self.assertEqual(list(self.parent.iterdir()), [])
        snapshot = self.snapshot()
        trace = Path(self.record['diagnostics']['quick_key']['frame_diagnostics_path'])
        trace.write_text(trace.read_text().replace('0.25', '0.35'))
        with self.assertRaisesRegex(ValueError, 'Trace changed|trace changed'):
            revalidate(snapshot)

    def test_invalid_bounds_parent_protection_and_cancel(self):
        snapshot = self.snapshot()
        for lo, hi in ((True, 4), (1, 5), (3, 8), (5, 5)):
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):
                prepare_range(snapshot, lo, hi, self.parent)
        for parent in (self.job.input_path, self.job.output_dir):
            with self.assertRaisesRegex(ValueError, 'protected'):
                prepare_range(snapshot, 3, 6, parent)
        with self.assertRaisesRegex(ValueError, 'protected'):
            prepare_range(snapshot, 3, 6, self.parent, protected_paths=[self.parent])
        def cancel():
            raise CancelledError('cancelled')
        with self.assertRaises(CancelledError):
            prepare_range(snapshot, 3, 6, self.parent, check=cancel)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_observed_selection_must_match_explicit_and_implicit_begin_bounds(self):
        source = self.record['diagnostics']['source']
        for start, count in ((3, 5), (2, 6)):
            source['start_frame'], source['num_frames'] = start, count
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                self.snapshot()
        cfg, writer = self.job.configurations()
        cfg = replace(cfg, start_frame=None)
        self.job = JobRequest.create(self.job.input_path, self.job.output_dir, cfg, writer)
        self.record['requested_settings'] = json_value(cfg)
        source['start_frame'], source['num_frames'] = 2, 5
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.snapshot()
        source['start_frame'] = 0
        self.assertEqual(self.snapshot().to_dict()['source']['start_frame'], 0)

    def test_wrong_terminal_binding_and_strict_trace_rows(self):
        wrong = dict(self.record, output_dir=str(self.parent))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            inspect_diagnostics(self.job, wrong)
        path = Path(self.record['diagnostics']['quick_key']['frame_diagnostics_path'])
        original = path.read_bytes()
        for raw in (b'{"frame":0,"frame":1}\n', b'{"frame":NaN}\n', b'[]\n', original.splitlines(keepends=True)[0] * 2,
                    original.rstrip(b'\n')):
            path.write_bytes(raw)
            with self.subTest(raw=raw[:40]), self.assertRaises((ValueError, TypeError)):
                self.snapshot()
        path.write_bytes(original)
        with patch('corridorkey.diagnostic_ranges.MAX_LINE_BYTES', 20), self.assertRaises(ValueError):
            self.snapshot()

    def test_trace_outside_alias_and_hardlink_refuse(self):
        quick = self.record['diagnostics']['quick_key']
        original = Path(quick['frame_diagnostics_path'])
        external = self.root / 'external.jsonl'
        external.write_bytes(original.read_bytes())
        quick['frame_diagnostics_path'] = str(external)
        with self.assertRaisesRegex(ValueError, 'belong'):
            self.snapshot()
        alias = Path(self.job.output_dir) / 'alias.jsonl'
        alias.symlink_to(original)
        quick['frame_diagnostics_path'] = str(alias)
        with self.assertRaisesRegex(ValueError, 'alias'):
            self.snapshot()
        alias.unlink()
        os.link(original, alias)
        quick['frame_diagnostics_path'] = str(original)
        with self.assertRaisesRegex(ValueError, 'single-link'):
            self.snapshot()

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_fifo_refuses_without_blocking(self):
        path = Path(self.job.output_dir) / 'pipe.jsonl'
        os.mkfifo(path)
        self.record['diagnostics']['quick_key']['frame_diagnostics_path'] = str(path)
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.snapshot()

    def test_no_trace_is_unavailable_not_zero_quality_and_frozen_job(self):
        self.record['diagnostics']['quick_key'] = None
        snapshot = self.snapshot()
        self.assertEqual(candidates(snapshot), ())
        self.assertEqual(snapshot.to_dict()['coverage']['quick_observed_frames'], 0)
        self.assertEqual(snapshot.to_dict()['job'], asdict(self.job))


if __name__ == '__main__':
    unittest.main()
