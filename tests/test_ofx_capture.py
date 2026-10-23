"""Native-float capture intake and real cache bytes; toy evidence, no host/GPU."""
import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.diagnostics import json_value
from corridorkey.ofx_cache import seal
from corridorkey.ofx_capture import (
    DOMAIN,
    MAX_PAYLOAD_BYTES,
    CapturePreparationError,
    PreparedCapture,
    attachment_receipt,
    capture_job_config,
    inventory_capture,
    materialize_capture,
    validate_capture_destination,
    verify_capture,
    verify_completed_capture,
)
from corridorkey.output_paths import output_layout
from corridorkey.project_collection import FilePin
from corridorkey.source_witness import SourceWitness
from corridorkey.video_io import read_image_rgb


def capture_fixture(root: Path, *, host_start=-17, frames=None):
    """Create the exact native-producer wire without claiming a Resolve capture."""
    directory = root / 'capture'
    directory.mkdir()
    if frames is None:
        bits = np.array([0x80000000, 1, 0x3f800000, 0xbf400000, 0x40900000, 0], np.uint32)
        frame = np.tile(bits.view(np.float32), 3).reshape(2, 3, 3)
        frames = (frame, frame[:, ::-1].copy())
    height, width = frames[0].shape[:2]
    payload = b''.join(np.ascontiguousarray(frame, dtype='<f4').tobytes() for frame in frames)
    data = {'schema_version': 1, 'type': 'corridorkey_ofx_capture', 'capture_id': 'a' * 32,
            'source_scope': 'ofx_instance_upstream', 'host_start': host_start, 'frame_count': len(frames),
            'width': width, 'height': height, 'fps': 24., 'pixel_aspect': 1.,
            'colour_contract': 'scene_linear_bt709_d65_opaque_rgb',
            'payload': {'path': 'frames.rgb32', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()},
            'source_digests': [hashlib.sha256(DOMAIN + struct.pack('<III', index, width, height)
                                            + np.ascontiguousarray(frame, dtype='<f4').tobytes()).hexdigest()
                               for index, frame in enumerate(frames)]}
    manifest = directory / 'capture.json'
    manifest.write_text(json.dumps(data, indent=2))
    (directory / 'frames.rgb32').write_bytes(payload)
    (directory / 'COMPLETE').touch()
    return manifest, data, frames


def completed_fixture(prepared: PreparedCapture):
    """Actual input decoding/witness and FLOAT EXRs, explicitly nonneural fixture."""
    config, writer = prepared.job.configurations()
    root = Path(prepared.job.output_dir)
    root.mkdir()
    data = prepared.plan.summary
    paths = [pin.path for pin in prepared.inputs]
    witness = SourceWitness({'native_hw': (data['height'], data['width']), 'start_frame': 0,
                             'num_frames': len(paths), 'native_fps': data['fps'], 'frame_paths': paths}, JobControl())
    outputs = {}
    for role, row in output_layout(prepared.job.input_path, config)['outputs'].items():
        outputs[role] = str(root / row['path'])
        Path(outputs[role]).mkdir()
    for index, path in enumerate(paths):
        rgb = read_image_rgb(path, expected_channels=3)
        witness.observe_path(path, rgb)
        for role, directory in outputs.items():
            pixels = np.full(rgb.shape[:2], .5, np.float32) if role == 'alpha' else rgb[:, :, ::-1]
            if not cv2.imwrite(str(Path(directory) / f'{role}_{index:06d}.exr'), pixels,
                               [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
                raise RuntimeError('Fixture EXR write failed')
    return {'schema_version': 1, 'status': 'complete', 'input': prepared.job.input_path,
            'output_dir': prepared.job.output_dir, 'hints': {'first_frame': None, 'full': None},
            'requested_settings': json_value(config), 'writer_settings': json_value(writer),
            'diagnostics': {'source': {'native_hw': [data['height'], data['width']], 'native_fps': data['fps'],
                                       'start_frame': 0, 'num_frames': len(paths)},
                            'frames_submitted': len(paths), 'effective_settings': json_value(config),
                            'execution': {'neural_inference': False, 'scope': 'CPU fixture only'},
                            'outputs': outputs, 'output_contracts': {
                                'fg': {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear'},
                                'alpha': {'format': 'exr', 'channels': 1}},
                            'source_witness': witness.finish(root)}}


class OfxCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest, self.data, self.frames = capture_fixture(self.root)
        self.scratch = self.root / 'scratch'
        self.scratch.mkdir()
        self.receipts = self.root / 'receipts'
        self.receipts.mkdir()
        self.output = self.root / 'output'
        self.originals = {path: path.read_bytes() for path in self.manifest.parent.iterdir()}

    def tearDown(self):
        # Tests that intentionally change captures restore them before returning.
        for path, raw in self.originals.items():
            self.assertEqual(path.read_bytes(), raw)

    def plan(self):
        return verify_capture(self.manifest, read_budget=MAX_PAYLOAD_BYTES)

    def prepared(self, **options):
        return materialize_capture(self.plan(), scratch_parent=self.scratch, output_dir=self.output,
                                   config=InferenceConfig(exr_dtype='float'), **options)

    def test_metadata_inventory_and_zero_budget_never_reads_payload(self):
        with patch('corridorkey.ofx_capture._frames', side_effect=AssertionError('metadata only')):
            report = inventory_capture(self.manifest)
            self.assertFalse(report['verified'])
            self.assertEqual(report['host_start'], -17)
            with self.assertRaisesRegex(ValueError, 'read budget'):
                verify_capture(self.manifest, read_budget=0)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_actual_float_exr_roundtrip_signed_zero_subnormal_hdr_and_local_job(self):
        prepared = self.prepared()
        self.assertIsInstance(prepared.workspace, Path)
        self.assertEqual(prepared.host_start, -17)
        config, _ = prepared.job.configurations()
        self.assertEqual((config.start_frame, config.num_frames, config.frame_dir_fps), (0, 2, 24.))
        self.assertTrue(config.source_witness and config.frame_dir_linear)
        self.assertEqual(config.source_colour_profile, 'legacy')
        for frame, pin in zip(self.frames, prepared.inputs, strict=True):
            decoded = read_image_rgb(pin.path, expected_channels=3)
            np.testing.assert_array_equal(decoded.view(np.uint32), frame.view(np.uint32))
            self.assertEqual(cv2.imread(pin.path, -1).dtype, np.float32)
        self.assertTrue((prepared.workspace / 'COMPLETE').exists())
        self.assertFalse(self.output.exists())

    def test_derived_config_is_pure_preserves_engine_choices_and_refuses_context_loss(self):
        plan = self.plan()
        config = InferenceConfig(model_scale=.5, hann_chunk=9, hann_stride=4, low_vram=True,
                                 outputs={'checker'}, exr_dtype='half', start_frame=30)
        original = json_value(config)
        with patch('corridorkey.ofx_capture.assert_pin', side_effect=AssertionError('pure config helper')):
            candidate = capture_job_config(plan, config)
        self.assertEqual(json_value(config), original)
        self.assertEqual((candidate.model_scale, candidate.hann_chunk, candidate.hann_stride, candidate.exr_dtype), (.5, 9, 4, 'half'))
        self.assertEqual(candidate.outputs, {'fg', 'alpha', 'checker'})
        for bad in (replace(config, corrections_path='/private/other.json'), replace(config, source_colour_profile='linear_bt709_d65'),
                    replace(config, start_mode='middle'), replace(config, quick_cut_frames=[3]),
                    replace(config, linear_brightness=1.1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                capture_job_config(plan, bad)
        invalid_configs: tuple[Any, ...] = (InferenceConfig, replace(config, outputs={'fg', 2}))
        for bad in invalid_configs:
            with self.assertRaises((TypeError, ValueError)):
                capture_job_config(plan, bad)

    def test_strict_wire_bounds_traversal_duplicate_fields_and_nonfinite_pixels(self):
        for field, value in (('host_start', 2**31), ('frame_count', True), ('width', 16 * 1024**2 + 1),
                             ('fps', 10**1000), ('pixel_aspect', True), ('extra', 1)):
            changed = copy.deepcopy(self.data)
            changed[field] = value
            self.manifest.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                inventory_capture(self.manifest)
        changed = copy.deepcopy(self.data)
        changed['payload']['path'] = '../frames.rgb32'
        self.manifest.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            inventory_capture(self.manifest)
        self.manifest.write_bytes(self.originals[self.manifest].replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'))
        with self.assertRaises(ValueError):
            inventory_capture(self.manifest)
        self.manifest.write_bytes(self.originals[self.manifest])
        payload = self.manifest.parent / 'frames.rgb32'
        raw = bytearray(self.originals[payload])
        raw[:4] = struct.pack('<I', 0x7fc00000)
        changed = copy.deepcopy(self.data)
        changed['payload']['sha256'] = hashlib.sha256(raw).hexdigest()
        frame_size = self.frames[0].nbytes
        changed['source_digests'][0] = hashlib.sha256(DOMAIN + struct.pack('<III', 0, 3, 2) + raw[:frame_size]).hexdigest()
        self.manifest.write_text(json.dumps(changed))
        payload.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.plan()
        self.manifest.write_bytes(self.originals[self.manifest])
        payload.write_bytes(self.originals[payload])

    def test_public_plan_and_prepared_instances_require_exact_physical_binding(self):
        plan = self.plan()
        for changed in (replace(plan, manifest_pin=plan.manifest_pin),):
            self.assertEqual(changed.identity, plan.identity)
        with self.assertRaises(TypeError):
            invalid_prepared: Any = PreparedCapture
            verify_completed_capture(invalid_prepared, {})
        with self.assertRaises((TypeError, ValueError)):
            replace(plan, payload_pin=FilePin)
        with self.assertRaises((TypeError, ValueError)):
            replace(plan, payload_pin=replace(plan.payload_pin, path=r'C:\foreign\frames.rgb32'))
        data = json.loads(plan.manifest_json)
        data['fps'] = 25.
        forged = replace(plan, manifest_json=json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
        with self.assertRaisesRegex(ValueError, 'physical manifest'):
            materialize_capture(forged, scratch_parent=self.scratch, output_dir=self.output, config=InferenceConfig())
        prepared = self.prepared()
        job_data = asdict(prepared.job)
        job_data['config_json'] = job_data['config_json'][:-1] + ',"model_scale":0.5}'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            replace(prepared, job=type(prepared.job)(**job_data))
        tampered = replace(prepared.job, config_json=prepared.job.config_json.replace('"model_scale": 1.0', '"model_scale": 0.5'))
        forged_prepared = replace(prepared, job=tampered)
        with self.assertRaisesRegex(ValueError, 'binding metadata'):
            verify_completed_capture(forged_prepared, {})

    def test_cancel_and_decode_drift_no_job_or_committed_workspace(self):
        plan = self.plan()
        cancellation = CancellationToken()
        cancellation.cancel('test explicit cancellation')
        with self.assertRaises(CancelledError):
            materialize_capture(plan, scratch_parent=self.scratch, output_dir=self.output, config=InferenceConfig(), check=cancellation.check)
        self.assertEqual(list(self.scratch.iterdir()), [])
        native = read_image_rgb

        def drift(path, **kwargs):
            pixels = native(path, **kwargs)
            pixels[0, 0, 0] = 0.5
            return pixels

        with patch('corridorkey.video_io.read_image_rgb', side_effect=drift), self.assertRaisesRegex(ValueError, 'float32 bits'):
            self.prepared()
        self.assertEqual(list(self.scratch.iterdir()), [])
        with patch('corridorkey.video_io.read_image_rgb', side_effect=RuntimeError('native failure')), patch('corridorkey.ofx_capture.cleanup_owned', side_effect=OSError('cleanup blocked')), self.assertRaises(CapturePreparationError) as caught:
            self.prepared()
        self.assertTrue(caught.exception.workspace.is_dir())
        self.assertFalse((caught.exception.workspace / 'COMPLETE').exists())

    def test_actual_record_source_witness_seal_and_receipt_map_signed_host_time(self):
        prepared = self.prepared()
        record = completed_fixture(prepared)
        report = verify_completed_capture(prepared, record)
        self.assertTrue(report['completed_job_verified'])
        self.assertIs(report['execution']['neural_inference'], False)
        self.assertEqual(validate_capture_destination(prepared, self.root), self.root)
        with self.assertRaisesRegex(ValueError, 'protected'):
            validate_capture_destination(prepared, prepared.workspace)
        self.assertIn('NOT VERIFIED', report['scope'])
        record_path = self.root / 'record.json'
        record_path.write_text(json.dumps(record))
        cache = seal(record_path, self.root)
        old_cache = self.root / 'previous-valid.ckcache'
        old_cache.write_bytes(b'previous selection untouched')
        receipt_path = attachment_receipt(prepared, record, cache, self.receipts)
        receipt = json.loads(receipt_path.read_bytes())
        self.assertEqual(receipt['cache_start'], -17)
        self.assertEqual(receipt['local_start'], 0)
        self.assertEqual(receipt['cache_sha256'], hashlib.sha256(cache.read_bytes()).hexdigest())
        self.assertEqual(receipt['cache_file'], str(cache))
        self.assertEqual(old_cache.read_bytes(), b'previous selection untouched')
        raw = bytearray(cache.read_bytes())
        raw[-1] ^= 1
        cache.write_bytes(raw)
        before = set(self.receipts.iterdir())
        with self.assertRaises(ValueError):
            attachment_receipt(prepared, record, cache, self.receipts)
        self.assertEqual(set(self.receipts.iterdir()), before)

    def test_failed_mismatched_record_and_foreign_paths_refuse_before_witness_io(self):
        prepared = self.prepared()
        record = completed_fixture(prepared)
        for field, value in (('status', 'failed'), ('input', '/different'), ('output_dir', '/different')):
            changed = copy.deepcopy(record)
            changed[field] = value
            with self.subTest(field=field), patch('corridorkey.source_witness.load_witness', side_effect=AssertionError('must refuse before witness IO')), self.assertRaises(ValueError):
                verify_completed_capture(prepared, changed)
        for mutate in ('output', 'witness', 'count', 'settings', 'effective'):
            changed = copy.deepcopy(record)
            if mutate == 'output':
                changed['diagnostics']['outputs']['fg'] = r'C:\foreign\fg'
            elif mutate == 'witness':
                changed['diagnostics']['source_witness']['path'] = r'C:\foreign\source.bin'
            elif mutate == 'count':
                changed['diagnostics']['frames_submitted'] -= 1
            elif mutate == 'settings':
                changed['requested_settings']['model_scale'] = .5
            else:
                changed['diagnostics']['effective_settings']['model_scale'] = .5
            with self.subTest(mutate=mutate), patch('corridorkey.source_witness.load_witness', side_effect=AssertionError('must refuse before witness IO')), self.assertRaises(ValueError):
                verify_completed_capture(prepared, changed)

    def test_alias_truncation_content_drift_and_fresh_output_refused(self):
        payload = self.manifest.parent / 'frames.rgb32'
        original = payload.read_bytes()
        plan = self.plan()
        payload.write_bytes(original[:-1])
        with self.assertRaises(ValueError):
            inventory_capture(self.manifest)
        payload.write_bytes(original)
        with self.assertRaises(ValueError):
            materialize_capture(plan, scratch_parent=self.scratch, output_dir=self.output, config=InferenceConfig())
        alias = self.root / 'alias.json'
        alias.symlink_to(self.manifest)
        with self.assertRaises(ValueError):
            inventory_capture(alias)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            self.prepared()
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_signed_host_range_requires_representable_out_exclusive(self):
        try:
            for start, accepted in ((-(2**31), True), (2**31 - 3, True),
                                    (2**31 - 2, False), (2**31 - 1, False)):
                changed = dict(self.data, host_start=start)
                self.manifest.write_text(json.dumps(changed))
                with self.subTest(start=start):
                    if accepted:
                        self.assertEqual(inventory_capture(self.manifest)['host_end_exclusive'], start + 2)
                    else:
                        with self.assertRaises(ValueError):
                            inventory_capture(self.manifest)
        finally:
            self.manifest.write_bytes(self.originals[self.manifest])

    def test_protected_scratch_parent_refuses_before_creation_even_without_cleanup(self):
        child = self.manifest.parent / 'existing-child'
        child.mkdir()
        try:
            for parent in (self.manifest.parent, child):
                before = set(self.manifest.parent.rglob('*'))
                with self.subTest(parent=parent), patch('corridorkey.ofx_capture.fresh_directory') as create, patch('corridorkey.ofx_capture.cleanup_owned', side_effect=OSError('cleanup unavailable')):
                    with self.assertRaisesRegex(ValueError, 'Scratch'):
                        materialize_capture(self.plan(), scratch_parent=parent, output_dir=self.output,
                                            config=InferenceConfig())
                    create.assert_not_called()
                self.assertEqual(set(self.manifest.parent.rglob('*')), before)
        finally:
            child.rmdir()

    def test_result_construction_failure_cannot_publish_ready_workspace(self):
        with patch('corridorkey.ofx_capture.PreparedCapture', side_effect=RuntimeError('construction failed')), patch('corridorkey.ofx_capture.cleanup_owned', side_effect=OSError('cleanup unavailable')), self.assertRaises(CapturePreparationError) as caught:
            self.prepared()
        root = caught.exception.workspace
        self.assertFalse((root / 'COMPLETE').exists())
        self.assertFalse((root / 'ready').exists())
        self.assertFalse(self.output.exists())

    def test_marker_close_failure_cannot_publish_receipt(self):
        prepared = self.prepared()
        record = completed_fixture(prepared)
        record_path = self.root / 'record.json'
        record_path.write_text(json.dumps(record))
        cache = seal(record_path, self.root)
        real_open = Path.open

        class FailingClose:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self.stream

            def __exit__(self, *args):
                self.stream.close()
                raise OSError('marker close failed')

        def open_marker(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            return FailingClose(stream) if path.name == 'COMPLETE' and args == ('xb',) else stream

        with patch.object(Path, 'open', open_marker), patch('corridorkey.ofx_capture.cleanup_owned', side_effect=OSError('cleanup unavailable')), self.assertRaises(CapturePreparationError) as caught:
            attachment_receipt(prepared, record, cache, self.receipts)
        self.assertFalse((caught.exception.workspace / 'COMPLETE').exists())
        self.assertFalse((caught.exception.workspace / 'ready').exists())
        self.assertTrue(cache.is_file())

    def test_auto_low_vram_accepts_only_documented_cache_zero_transition(self):
        config = InferenceConfig(auto_vram=True, foundation_cache_mb=16, exr_dtype='float')
        prepared = materialize_capture(self.plan(), scratch_parent=self.scratch, output_dir=self.output,
                                       config=config)
        record = completed_fixture(prepared)
        effective = record['diagnostics']['effective_settings']
        effective.update(low_vram=True, foundation_cache_mb=0, hann_chunk=8, hann_stride=4)
        self.assertTrue(verify_completed_capture(prepared, record)['completed_job_verified'])
        for changes in ({'foundation_cache_mb': 8}, {'foundation_cache_mb': 0, 'low_vram': False},
                        {'foundation_cache_mb': 0, 'model_scale': .5}):
            changed = copy.deepcopy(record)
            changed['diagnostics']['effective_settings'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verify_completed_capture(prepared, changed)
