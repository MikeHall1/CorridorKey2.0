"""Real sequence indexing and metadata fences; no inference/speed claim."""
import json
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')

import cv2
import numpy as np

from corridorkey.control import CancellationToken, CancelledError
from corridorkey.desktop_comparison import RenderView
from corridorkey.preview_background import BackgroundRecipe, prepare_background
from corridorkey.preview_exports import (
    ExportIndexLimits,
    PreparedExportIndexes,
    capture_export_request,
    prepare_export_indexes,
)
from corridorkey.preview_sequence import SequenceIndex


class PreviewExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root/'source'
        self.source.mkdir()
        for index in range(8):
            self.assertTrue(cv2.imwrite(str(self.source/f'frame{index}.png'),np.full((2,3,3),index*20,np.uint8)))
        self.a = self.make_view('a',2,5)
        self.b = self.make_view('b',5,3)

    def tearDown(self):
        self.temp.cleanup()

    def make_view(self,name,start,count):
        paths = {}
        for kind in ('fg','alpha'):
            path = self.root/f'{name}-{kind}'
            path.mkdir()
            paths[kind] = str(path)
            for index in range(count):
                values = np.full((2,3,3) if kind=='fg' else (2,3),.1*(index+1) if kind=='fg' else .5,np.float32)
                self.assertTrue(cv2.imwrite(str(path/f'frame{index}.exr'),values))
        return RenderView(str(self.source),start,count,2,3,24.,tuple(paths.items()),True,False,'ignore')

    def prepare(self,**options):
        return prepare_export_indexes(capture_export_request(self.b,mode='Checker composite',**options))

    def test_no_io_immutable_capture_cached_identity_and_public_instance_contracts(self):
        with (patch.object(Path,'resolve',side_effect=AssertionError('no resolve')),
              patch.object(Path,'stat',side_effect=AssertionError('no stat')),
              patch('os.scandir',side_effect=AssertionError('no scan'))):
            request = capture_export_request(self.b,mode='Photo composite',reference=self.a,reference_mode='Alpha')
            self.assertEqual(request.view,self.b)
            self.assertEqual(request.reference_mode,'Alpha')
        self.assertEqual(request.identity,capture_export_request(self.b,mode='Photo composite',reference=self.a,reference_mode='Alpha').identity)
        self.assertEqual(request.__dict__['identity'],request.identity)
        json.dumps(asdict(request),allow_nan=False)
        invalid:Any = request
        with self.assertRaises(FrozenInstanceError):
            invalid.mode = 'Source'
        for value in (RenderView,ExportIndexLimits,{1,'mixed'}):
            bad:Any = value
            with self.assertRaises(TypeError):
                capture_export_request(bad,mode='Source')
            with self.assertRaises(TypeError):
                prepare_export_indexes(bad)
            with self.assertRaises(TypeError):
                json.dumps(value)
        bad:Any = replace(self.b,outputs=[('alpha',dict(self.b.outputs)['alpha'])])
        with self.assertRaises(ValueError):
            capture_export_request(bad,mode='Alpha')
        with self.assertRaises(ValueError):
            capture_export_request(replace(self.b,source='relative'),mode='Source')
        invalid_limits:Any = ExportIndexLimits
        with self.assertRaises(TypeError):
            capture_export_request(self.b,mode='Alpha',limits=invalid_limits)

    def test_composite_ab_source_offsets_and_distinct_matte_indexes(self):
        bundle = self.prepare(reference=self.a,reference_mode='Checker composite')
        self.assertEqual(len(bundle.streams),4)
        pair = bundle.options_for(6)['pair']
        self.assertEqual((pair['a']['index'],pair['b']['index']),(4,1))
        for key,view in (('a',self.a),('b',self.b)):
            options = pair[key]
            self.assertIsInstance(options['sequence'],SequenceIndex)
            self.assertIsInstance(options['alpha_sequence'],SequenceIndex)
            self.assertEqual(len(options['sequence'].paths),view.count)
            self.assertEqual(options['sequence'].read_rgb(options['index']).shape,(2,3,3))
            np.testing.assert_allclose(options['alpha_sequence'].read_rgb(options['index']),.5)
        with self.assertRaises(ValueError):
            bundle.options_for(7)
        with patch('corridorkey.preview_exports.hashlib.sha256',side_effect=AssertionError('no rehash on frame/UI')):
            self.assertEqual(bundle.options_for(6)['pair']['b']['index'],1)
            self.assertEqual(bundle.identity,bundle.__dict__['identity'])

    def test_source_global_membership_natural_order_and_refresh_additions(self):
        bundle = prepare_export_indexes(capture_export_request(self.b,mode='Source'))
        options = bundle.options_for(6)
        self.assertEqual(options['index'],6)
        self.assertEqual(len(options['sequence'].paths),8)
        self.assertTrue(cv2.imwrite(str(self.source/'frame10.png'),np.zeros((2,3,3),np.uint8)))
        self.assertEqual(len(options['sequence'].paths),8)
        refreshed = prepare_export_indexes(bundle.request)
        self.assertEqual(len(refreshed.streams[0].paths),9)
        self.assertTrue(refreshed.streams[0].paths[-1].endswith('frame10.png'))
        self.assertNotEqual(bundle.identity,refreshed.identity)
        self.assertEqual(refreshed.request.identity,bundle.request.identity)
        with self.assertRaisesRegex(ValueError,'coverage'):
            prepare_export_indexes(capture_export_request(replace(self.b,start=20),mode='Source'))

    def test_deduplicates_physical_directories_preserves_readonly_alias_and_hardlink(self):
        alpha = Path(dict(self.b.outputs)['alpha'])
        alias = self.root/'alpha-alias'
        alias.symlink_to(alpha,target_is_directory=True)
        reference = replace(self.b,outputs=(('alpha',str(alias)),))
        with patch('corridorkey.preview_exports.scan_sequence',wraps=__import__('corridorkey.preview_exports',fromlist=['scan_sequence']).scan_sequence) as scan:
            bundle = prepare_export_indexes(capture_export_request(self.b,mode='Alpha',reference=reference,reference_mode='Alpha'))
        self.assertEqual(scan.call_count,1)
        self.assertEqual(len(bundle.streams),1)
        pair = bundle.options_for(6)['pair']
        self.assertIs(pair['a']['sequence'],pair['b']['sequence'])
        np.testing.assert_array_equal(pair['a']['sequence'].read_rgb(1),pair['b']['sequence'].read_rgb(1))
        source_file = self.source/'frame0.png'
        linked = self.source/'frame8.png'
        os.link(source_file,linked)
        leaf = self.source/'frame9.png'
        leaf.symlink_to(source_file)
        source = prepare_export_indexes(capture_export_request(self.b,mode='Source'))
        self.assertEqual(len(source.streams[0].paths),10)
        np.testing.assert_array_equal(source.streams[0].read_rgb(8),source.streams[0].read_rgb(9))

    def test_no_per_frame_scandir_or_full_frame_paths_and_preserves_source_bytes(self):
        originals = {path:path.read_bytes() for stream in (self.source,*(Path(path) for _,path in self.b.outputs)) for path in stream.iterdir()}
        bundle = self.prepare()
        with (patch('os.scandir',side_effect=AssertionError('no per-frame directory scan')),
              patch('corridorkey.desktop_preview.frame_paths',side_effect=AssertionError('no legacy enumeration'))):
            for frame in range(self.b.start,self.b.end):
                options = bundle.options_for(frame)
                options['sequence'].read_rgb(options['index'])
                options['alpha_sequence'].read_rgb(options['index'])
        self.assertEqual(originals,{path:path.read_bytes() for path in originals})

    def test_aggregate_entry_frame_path_directory_metadata_limits_and_count_refusal(self):
        for limits in (ExportIndexLimits(max_directories=1),ExportIndexLimits(max_entries=4),
                       ExportIndexLimits(max_frames=4),ExportIndexLimits(max_path_bytes=20),
                       ExportIndexLimits(max_metadata_bytes=8192)):
            with self.subTest(limits=limits),self.assertRaises(ValueError):
                self.prepare(limits=limits)
        with self.assertRaisesRegex(ValueError,'coverage'):
            prepare_export_indexes(capture_export_request(replace(self.b,count=2),mode='Alpha'))
        with self.assertRaises(ValueError):
            capture_export_request(self.b,mode='unknown')
        with self.assertRaises(ValueError):
            capture_export_request(replace(self.b,linear=False),mode='Alpha')
        with self.assertRaises(ValueError):
            capture_export_request(replace(self.b,outputs=()),mode='Checker composite')

    def test_ordinary_same_size_restored_mtime_removal_and_mid_decode_mutations_refuse(self):
        bundle = self.prepare()
        index = bundle.streams[0]
        path = Path(index.paths[0])
        raw = path.read_bytes()
        info = path.stat()
        path.write_bytes(raw)
        os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'changed'):
            index.read_rgb(0)
        bundle = self.prepare()
        index = bundle.streams[0]
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            index.read_rgb(0)
        path.write_bytes(raw)
        index = self.prepare().streams[0]
        from corridorkey.video_io import read_image_rgb
        def changed(*args,**kwargs):
            pixels = read_image_rgb(*args,**kwargs)
            path.write_bytes(raw)
            return pixels
        with patch('corridorkey.video_io.read_image_rgb',side_effect=changed),self.assertRaisesRegex(ValueError,'changed'):
            index.read_rgb(0)

    def test_preparation_cancellation_and_public_bundle_index_forgery_boundaries(self):
        token = CancellationToken()
        token.cancel('before scan')
        with patch('os.scandir',side_effect=AssertionError('no scan')),self.assertRaises(CancelledError):
            prepare_export_indexes(capture_export_request(self.b,mode='Alpha'),check=token.check)
        calls = [0]
        def check():
            calls[0] += 1
            if calls[0]==5:
                raise CancelledError('during metadata scan')
        with self.assertRaises(CancelledError):
            prepare_export_indexes(capture_export_request(self.b,mode='Alpha'),check=check)
        bundle = self.prepare()
        with self.assertRaises(ValueError):
            replace(bundle,identity='0'*64)
        with self.assertRaises(ValueError):
            replace(bundle,metadata_bytes=1)
        with self.assertRaises(ValueError):
            replace(bundle,bindings=(('/outside',0),))
        first = bundle.streams[0]
        with self.assertRaises(ValueError):
            replace(first,paths=('/outside/frame.exr',*first.paths[1:]))
        invalid:Any = PreparedExportIndexes
        with self.assertRaises(TypeError):
            PreparedExportIndexes(invalid,bundle.streams,bundle.bindings,bundle.identity,bundle.metadata_bytes)

    def test_photo_snapshot_passes_same_global_object_through_both_views(self):
        photo = self.root/'photo.png'
        self.assertTrue(cv2.imwrite(str(photo),np.full((2,3,3),128,np.uint8)))
        background = prepare_background(BackgroundRecipe.capture(photo,profile='assumed_srgb',expected_hw=(2,3)))
        bundle = prepare_export_indexes(capture_export_request(self.b,mode='Photo composite',reference=self.a,reference_mode='Photo composite'))
        pair = bundle.options_for(6,background_image=background)['pair']
        self.assertIs(pair['a']['background_image'],background)
        self.assertIs(pair['b']['background_image'],background)
        self.assertEqual((pair['a']['index'],pair['b']['index']),(4,1))

    def test_actual_4k_exr_read_keeps_existing_decoder_geometry_without_new_pixel_cap(self):
        root = self.root/'4k-alpha'
        root.mkdir()
        path = root/'frame0.exr'
        self.assertTrue(cv2.imwrite(str(path),np.full((2160,3840),.5,np.float32)))
        view = replace(self.b,start=0,count=1,height=2160,width=3840,outputs=(('alpha',str(root)),))
        bundle = prepare_export_indexes(capture_export_request(view,mode='Alpha'))
        pixels = bundle.options_for(0)['sequence'].read_rgb(0)
        self.assertEqual(pixels.shape,(2160,3840,3))
        self.assertEqual(float(pixels[100,200,0]),.5)
