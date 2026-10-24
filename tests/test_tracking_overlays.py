"""Actual tracking candidates and bounded proxy IO; no quality confidence claim."""
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from corridorkey.control import CancelledError
from corridorkey.correction_tracking import (
    TrackingAnchor,
    TrackingCandidate,
    TrackingFrame,
    TrackingRequest,
    track_corrections,
)
from corridorkey.tracking_overlays import (
    OverlayBinding,
    OverlayBundle,
    OverlayMap,
    OverlayReceipt,
    build_tracking_overlays,
    read_bundle,
    render_layer,
    verify_binding,
    write_bundle,
)
from tests.test_project_collection import windows_file_clocks


@contextmanager
def finalize_overlay_writes(finalize):
    original = Path.open

    @contextmanager
    def opened(path, *args, **kwargs):
        overlay = path.name.startswith('overlay-') and path.suffix == '.u8' and args == ('xb',)
        before = None
        with original(path, *args, **kwargs) as stream:
            yield stream
            if overlay:
                before = os.fstat(stream.fileno())
        if overlay:
            assert before is not None
            finalize(path, before)

    with patch.object(Path, 'open', opened):
        yield


def candidate():
    frames = [np.full((8,16,3), .4, np.float32) for _ in range(2)]
    mask = np.zeros((8,16), np.float32)
    mask[:,4:8] = 1
    anchor = TrackingAnchor.create('artist',0,0,'keep',1.,mask)
    request = TrackingRequest('a'*64,0,2,source_start=10, source_decode_identity='b'*64)
    def flow(previous, following):
        return np.zeros((*previous.shape,2),np.float32)
    return list(track_corrections(request,(anchor,),lambda absolute:TrackingFrame('a'*64,absolute,frames[absolute-10]),estimator=flow,estimator_identity='zero-flow-fixture'))


def build(value=None, **options):
    result = build_tracking_overlays(value if value is not None else candidate()[1], document_identity='c'*64,decode_identity='b'*64,**options)
    assert result is not None
    return result


class TrackingOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_candidates_bind_offsets_proxy_and_heuristic_meaning(self):
        anchor, propagated = candidate()
        before = (propagated.mask_bytes,propagated.reliability_bytes,propagated.uncertain_bytes)
        bundle = build(propagated,max_side=8)
        self.assertEqual((bundle.binding.local_frame,bundle.binding.source_frame),(1,11))
        self.assertEqual((bundle.binding.native_width,bundle.binding.native_height),(16,8))
        self.assertEqual(bundle.binding.source_proxy_sha256,json.loads(propagated.provenance_json)['source_proxy_sha256']['1'])
        self.assertEqual([(layer.kind,layer.width,layer.height) for layer in bundle.maps],[('reliability',8,4),('uncertainty',8,4)])
        self.assertEqual(set(bundle.maps[0].payload),{255})
        self.assertEqual(set(bundle.maps[1].payload),{0})
        self.assertIsNone(build_tracking_overlays(anchor,document_identity='c'*64,decode_identity='b'*64))
        overlay = render_layer(bundle,'reliability')
        self.assertFalse(overlay.legend['calibrated'])
        self.assertIn('initialized conventions',overlay.legend['anchor_note'])
        self.assertIn('Not quality confidence',overlay.legend['limitation'])
        self.assertEqual(before,(propagated.mask_bytes,propagated.reliability_bytes,propagated.uncertain_bytes))

    def test_fractional_area_average_and_rgba_does_not_change_any_original(self):
        value = candidate()[1]
        uncertainty = np.zeros((8,16),np.uint8)
        uncertainty[:,::2] = 1
        reliability = np.tile(np.array([0.,1.],np.float32),64).reshape(8,16)
        value = replace(value,uncertain_bytes=uncertainty.tobytes(),reliability_bytes=reliability.astype('<f4').tobytes())
        bundle = build(value,max_side=8)
        self.assertEqual(set(bundle.maps[0].payload),{128})
        self.assertEqual(set(bundle.maps[1].payload),{128})
        before = bundle.maps[0].payload
        rendered = render_layer(bundle,'uncertainty',opacity=.5)
        rgba = np.frombuffer(rendered.pixels,np.uint8).reshape(4,8,4)
        self.assertTrue(np.all(rgba[:,:,3] == 64))
        self.assertEqual(before,bundle.maps[0].payload)
        legend = rendered.legend
        legend['kind'] = 'changed outside owner'
        self.assertEqual(rendered.legend['kind'],'uncertainty')
        with self.assertRaises(ValueError):
            render_layer(bundle,'occlusion_ground_truth')

    def test_windows_clock_domains_write_path_pins_and_read_bundle(self):
        bundle = build()
        with windows_file_clocks():
            receipt = write_bundle(bundle, self.root, remaining_budget=64 * 1024**2)
            self.assertEqual(read_bundle(receipt, self.root, expected_binding=bundle.binding), bundle)

    def test_windows_close_finalizes_mtime_before_overlay_pin_admission(self):
        bundle = build()
        finalized = []

        def finalize(path, before):
            raw = path.read_bytes()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10000000))
            self.assertEqual((path.stat().st_dev, path.stat().st_ino, path.stat().st_size),
                             (before.st_dev, before.st_ino, before.st_size))
            self.assertEqual(path.read_bytes(), raw)
            finalized.append(path)

        with windows_file_clocks(), finalize_overlay_writes(finalize):
            receipt = write_bundle(bundle, self.root, remaining_budget=64 * 1024**2)
            self.assertEqual(read_bundle(receipt, self.root, expected_binding=bundle.binding), bundle)
        self.assertEqual(len(finalized), len(bundle.maps))

    def test_postclose_overlay_replacement_and_wrong_payload_remain_refused(self):
        for kind in ('replacement', 'payload'):
            with self.subTest(kind=kind):
                root = self.root / kind
                root.mkdir()
                retained = []

                def finalize(path, before, kind=kind, root=root, retained=retained):
                    if kind == 'replacement':
                        replacement = root / 'replacement'
                        replacement.write_bytes(path.read_bytes())
                        replacement.replace(path)
                        retained.append(path)
                    else:
                        path.write_bytes(b'x' * before.st_size)

                with windows_file_clocks(), finalize_overlay_writes(finalize), self.assertRaises(ValueError):
                    write_bundle(build(), root, remaining_budget=64 * 1024**2)
                if kind == 'replacement':
                    self.assertEqual(list(root.iterdir()), retained)
                    self.assertEqual(retained[0].read_bytes(), build().maps[0].payload)
                else:
                    self.assertEqual(list(root.iterdir()), [])

    def test_private_receipt_roundtrip_exact_and_aggregate_budget_admission(self):
        bundle = build()
        with patch('corridorkey.tracking_overlays.read_file',side_effect=AssertionError('no default IO')):
            render_layer(bundle,'reliability')
        receipt = write_bundle(bundle,self.root,remaining_budget=64*1024**2)
        self.assertEqual(len(list(self.root.iterdir())),2)
        self.assertGreater(receipt.bytes_charged,sum(len(layer.payload) for layer in bundle.maps))
        self.assertEqual(read_bundle(receipt.to_dict(),self.root,expected_binding=bundle.binding),bundle)
        self.assertEqual(receipt.binding,bundle.binding)
        self.assertEqual(receipt.identity,bundle.identity)
        files = set(self.root.iterdir())
        with self.assertRaisesRegex(ValueError,'budget'):
            write_bundle(bundle,self.root,remaining_budget=receipt.bytes_charged-1)
        self.assertEqual(set(self.root.iterdir()),files)
        detached = receipt.to_dict()
        detached['maps'].clear()
        self.assertEqual(len(receipt.to_dict()['maps']),2)

    def test_source_correction_frame_decode_candidate_proxy_all_fence_stale(self):
        bundle = build()
        for field,value in (('source_identity','d'*64),('document_identity','d'*64),('candidate_identity','d'*64),('decode_identity','d'*64),('source_frame',12),('local_frame',0),('source_proxy_sha256','d'*64)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify_binding(bundle,replace(bundle.binding,**{field:value}))
        receipt = write_bundle(bundle,self.root,remaining_budget=64*1024**2)
        with patch('corridorkey.tracking_overlays.read_file',side_effect=AssertionError('binding before file read')),self.assertRaises(ValueError):
            read_bundle(receipt,self.root,expected_binding=replace(bundle.binding,source_identity='d'*64))

    def test_receipt_traversal_symlink_hardlink_corruption_and_size_refused(self):
        import os
        bundle = build()
        receipt = write_bundle(bundle,self.root,remaining_budget=64*1024**2)
        data = receipt.to_dict()
        for name in ('../escape','C:evil',r'folder\leaf','CON.u8'):
            corrupted = receipt.to_dict()
            corrupted['maps'][0]['name'] = name
            with self.subTest(name=name),self.assertRaises(ValueError):
                OverlayReceipt(json.dumps(corrupted).encode())
        path = self.root/data['maps'][0]['name']
        raw = path.read_bytes()
        path.write_bytes(b'x'*len(raw))
        with self.assertRaisesRegex(ValueError,'identity'):
            read_bundle(receipt,self.root)
        path.unlink()
        elsewhere = self.root/'original'
        elsewhere.write_bytes(raw)
        path.symlink_to(elsewhere)
        with self.assertRaises(ValueError):
            read_bundle(receipt,self.root)
        path.unlink()
        os.link(elsewhere,path)
        with self.assertRaises(ValueError):
            read_bundle(receipt,self.root)
        path.unlink()
        path.write_bytes(raw+b'grew')
        with self.assertRaisesRegex(ValueError,'size'):
            read_bundle(receipt,self.root)

    def test_cancelled_partial_write_cleans_only_created_leaves_and_preserves_primary(self):
        bundle = build()
        unrelated = self.root/'unrelated'
        unrelated.write_bytes(b'keep')
        def cancel_after_first():
            if len(list(self.root.glob('overlay-*'))) == 1:
                raise CancelledError('cancel after first owned map')
        with self.assertRaises(CancelledError):
            write_bundle(bundle,self.root,remaining_budget=64*1024**2,check=cancel_after_first)
        self.assertEqual(list(self.root.iterdir()),[unrelated])
        with patch.object(Path,'unlink',side_effect=OSError('cleanup denied')),self.assertRaises(CancelledError) as caught:
            write_bundle(bundle,self.root,remaining_budget=64*1024**2,check=cancel_after_first)
        self.assertIn('cleanup denied',str(caught.exception.__notes__))
        self.assertEqual(unrelated.read_bytes(),b'keep')

    def test_missing_maps_available_without_mask_and_invalid_native_bytes_values(self):
        value = candidate()[1]
        self.assertIsNone(build_tracking_overlays(replace(value,reliability_bytes=None,uncertain_bytes=None),document_identity='c'*64,decode_identity='b'*64))
        self.assertIsNotNone(build(replace(value,mask_bytes=None)))
        for changes in ({'reliability_bytes':None},{'uncertain_bytes':b'x'},
                        {'reliability_bytes':np.full((8,16),np.nan,dtype='<f4').tobytes()},
                        {'uncertain_bytes':bytes([2])*128},{'source_frame':12}):
            with self.subTest(changes=list(changes)),self.assertRaises(ValueError):
                build(replace(value,**changes))

    def test_first_failed_propagation_maps_remain_inspectable_later_blocked_unavailable(self):
        frames = [np.full((8,16,3),.4,np.float32),np.full((8,16,3),1.,np.float32),np.full((8,16,3),.4,np.float32)]
        anchor = TrackingAnchor.create('artist',0,0,'keep',1.,np.ones((8,16),np.float32))
        request = TrackingRequest('a'*64,0,3,source_start=10,source_decode_identity='b'*64,unreliable_policy='stop')
        def flow(previous, following):
            return np.zeros((*previous.shape,2),np.float32)
        values = list(track_corrections(request,(anchor,),lambda index:TrackingFrame('a'*64,index,frames[index-10]),estimator=flow,estimator_identity='zero-flow-fixture'))
        self.assertIsNone(values[1].mask_bytes)
        self.assertEqual(values[1].status,'stopped')
        bundle = build(values[1])
        self.assertEqual(set(bundle.maps[1].payload),{255})
        self.assertIsNone(build_tracking_overlays(values[2],document_identity='c'*64,decode_identity='b'*64))

    def test_native_map_byte_bound_before_array_work_and_cancelled_read(self):
        value = candidate()[1]
        with patch('numpy.frombuffer',side_effect=AssertionError('admit first')),self.assertRaises(ValueError):
            build(replace(value,width=4*1024*1024,height=2))
        receipt = write_bundle(build(),self.root,remaining_budget=64*1024**2)
        def cancel():
            raise CancelledError('read cancelled')
        with self.assertRaises(CancelledError):
            read_bundle(receipt,self.root,check=cancel)
        self.assertEqual(len(list(self.root.iterdir())),2)

    def test_failed_exclusive_collision_preserves_existing_leaf(self):
        bundle = build()
        fixed = 'a'*32
        first = self.root/f'overlay-{fixed}-reliability.u8'
        first.write_bytes(b'unrelated existing leaf')
        class FixedUUID:
            hex = fixed
        with patch('corridorkey.tracking_overlays.uuid.uuid4',return_value=FixedUUID()),self.assertRaises(FileExistsError):
            write_bundle(bundle,self.root,remaining_budget=64*1024**2)
        self.assertEqual(first.read_bytes(),b'unrelated existing leaf')
        self.assertEqual(list(self.root.iterdir()),[first])

    def test_final_write_fence_rejects_map_changed_after_verified_copy(self):
        from corridorkey.project_collection import read_file as actual_read
        changed = False
        def mutate(pin, **options):
            nonlocal changed
            result = actual_read(pin,**options)
            if not changed:
                changed = True
                Path(pin.path).write_bytes(b'x'*pin.size)
            return result
        with patch('corridorkey.tracking_overlays.read_file',side_effect=mutate),self.assertRaises(ValueError):
            write_bundle(build(),self.root,remaining_budget=64*1024**2)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_immutable_classes_sets_provenance_geometry_and_boundaries(self):
        bundle = build()
        mutable: Any = bundle.maps[0]
        with self.assertRaises(FrozenInstanceError):
            mutable.payload = b'changed'
        invalid: Any = TrackingCandidate
        with self.assertRaises(TypeError):
            build(invalid)
        with self.assertRaises(TypeError):
            invalid_binding: Any = OverlayBinding
            OverlayBundle(invalid_binding,bundle.maps,bundle.provenance_json)
        with self.assertRaises(ValueError):
            build(max_side=257)
        with self.assertRaises(ValueError):
            replace(bundle,maps=(OverlayMap('reliability',2,2,bytes(4)),OverlayMap('uncertainty',2,2,bytes(4))))
        from corridorkey.tracking_overlays import _canonical
        for invalid_metadata in ({'class':OverlayBinding},{'instance':bundle.binding},{'set':{1,'x'}}):
            with self.assertRaises(TypeError):
                _canonical(invalid_metadata)
        # Caller-supplied document/decode identities are strict SHA values.
        with self.assertRaises(ValueError):
            build_tracking_overlays(candidate()[1],document_identity='c'*64,decode_identity='unsupported')
