"""Real bounded image fixtures and analytic preview math; no display/GPU claim."""
import hashlib
import json
import os
import struct
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')

import cv2
import numpy as np

from corridorkey.control import CancellationToken, CancelledError
from corridorkey.preview_background import (
    MAX_WORK_BYTES,
    BackgroundRecipe,
    PreparedBackground,
    compose_photo,
    prepare_background,
)


def _linear(values):
    values = np.asarray(values,np.float32)
    magnitude = np.abs(values)
    return np.sign(values)*np.where(magnitude<=.04045,magnitude/12.92,((magnitude+.055)/1.055)**2.4)


class PreviewBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.path = self.root/'photo.png'
        self.rgb = np.array([[[0,64,128],[255,192,96],[32,100,200]],
                             [[255,0,100],[128,128,128],[10,20,30]]],np.uint8)
        self.assertTrue(cv2.imwrite(str(self.path),self.rgb[:,:,::-1]))

    def tearDown(self):
        self.temp.cleanup()

    def recipe(self,path=None,**changes):
        values = {'profile':'assumed_srgb','expected_hw':(2,3)}
        values.update(changes)
        return BackgroundRecipe.capture(path or self.path,**values)

    def prepared(self):
        return prepare_background(self.recipe())

    def test_capture_is_immutable_no_io_and_admits_explicit_work_bounds(self):
        with (patch.object(Path,'resolve',side_effect=AssertionError('no resolve')),
              patch.object(Path,'stat',side_effect=AssertionError('no stat')),
              patch('os.open',side_effect=AssertionError('no open'))):
            recipe = self.recipe()
            hd = BackgroundRecipe.capture('/not/opened/photo.png',profile='assumed_srgb',expected_hw=(1080,1920))
            self.assertLessEqual(hd.work_bytes_estimate,MAX_WORK_BYTES)
        with self.assertRaises(FrozenInstanceError):
            invalid:Any = recipe
            invalid.path = '/changed'
        for changes in ({'max_pixels':5},{'max_work_bytes':1},{'expected_hw':(2048,2048)},
                        {'expected_hw':(True,3)},{'expected_hw':[2,3]}, {'geometry':'crop'},
                        {'profile':'automatic'},{'max_encoded_bytes':0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.recipe(**changes)
        for value in (BackgroundRecipe,PreparedBackground,{1,'mixed'}):
            invalid_argument:Any = value
            with self.assertRaises(TypeError):
                BackgroundRecipe.capture(invalid_argument,profile='assumed_srgb',expected_hw=(2,3))
            with self.assertRaises(TypeError):
                prepare_background(invalid_argument)

    def test_actual_rgb_png_snapshot_hash_pixels_provenance_and_detached_state(self):
        raw = self.path.read_bytes()
        prepared = self.prepared()
        self.assertEqual(prepared.sha256,hashlib.sha256(raw).hexdigest())
        self.assertEqual(prepared.native_hw,(2,3))
        self.assertEqual(prepared.profile,'assumed_srgb')
        self.assertIs(type(prepared.linear_rgb_f32),bytes)
        actual = np.frombuffer(prepared.linear_rgb_f32,'<f4').reshape(2,3,3)
        np.testing.assert_allclose(actual,_linear(self.rgb/np.float32(255)),rtol=1e-6,atol=1e-7)
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(prepared.identity,self.prepared().identity)
        self.assertIn('ICC',prepared.provenance['icc_policy'])
        self.assertIn('EXIF orientation ignored',prepared.provenance['orientation_policy'])
        detached = prepared.provenance
        detached['native_hw'][0] = 99
        self.assertEqual(prepared.provenance['native_hw'],[2,3])
        self.assertEqual(raw,self.path.read_bytes())
        self.assertTrue(cv2.imwrite(str(self.path),np.zeros((2,3,3),np.uint8)))
        np.testing.assert_allclose(actual,_linear(self.rgb/np.float32(255)),rtol=1e-6,atol=1e-7)
        self.assertNotEqual(prepared.identity,self.prepared().identity)

    def test_analytic_srgb_composite_is_linear_fresh_and_preserves_inputs(self):
        background = self.prepared()
        foreground = np.linspace(-.2,1.2,18,dtype=np.float32).reshape(2,3,3)
        alpha = np.array([[0.,1.,.5],[.25,.75,.5]],np.float32)
        fg_raw,alpha_raw = foreground.tobytes(),alpha.tobytes()
        result = compose_photo(foreground,alpha,background,foreground_linear=False)
        backdrop = np.frombuffer(background.linear_rgb_f32,'<f4').reshape(2,3,3)
        expected = _linear(foreground)*alpha[:,:,None]+backdrop*(1-alpha[:,:,None])
        np.testing.assert_allclose(result,expected,rtol=2e-6,atol=1e-7)
        self.assertEqual(result.dtype,np.float32)
        self.assertFalse(np.shares_memory(result,foreground))
        self.assertEqual((foreground.tobytes(),alpha.tobytes()),(fg_raw,alpha_raw))
        result[:] = 0
        self.assertEqual((foreground.tobytes(),alpha.tobytes()),(fg_raw,alpha_raw))

    def test_actual_float_exr_linear709_preserves_hdr_negative_and_coverage_endpoints(self):
        path = self.root/'linear.exr'
        rgb = np.array([[[-1.,.2,8.],[2.,4.,16.],[.1,.2,.3]],[[3.,-2.,1.],[.5,.8,1.2],[8.,4.,2.]]],np.float32)
        self.assertTrue(cv2.imwrite(str(path),rgb[:,:,::-1]))
        background = prepare_background(self.recipe(path,profile='linear_bt709_d65'))
        np.testing.assert_array_equal(np.frombuffer(background.linear_rgb_f32,'<f4').reshape(2,3,3),rgb)
        foreground = np.full((2,3,3),4,np.float32)
        alpha = np.zeros((2,3,1),np.float32)
        np.testing.assert_array_equal(compose_photo(foreground,alpha,background,foreground_linear=True),rgb)
        alpha[:] = 1
        np.testing.assert_array_equal(compose_photo(foreground,alpha,background,foreground_linear=True),foreground)
        self.assertGreater(float(compose_photo(foreground,alpha,background,foreground_linear=True).max()),1.)
        with patch('cv2.imdecode',side_effect=AssertionError('profile admission before decode')),self.assertRaisesRegex(ValueError,'profile'):
            prepare_background(self.recipe(path,profile='assumed_srgb'))

    def test_rgb_uint16_png_and_jpeg_stored_orientation_no_implicit_transform(self):
        path = self.root/'rgb16.png'
        rgb16 = self.rgb.astype(np.uint16)*257
        self.assertTrue(cv2.imwrite(str(path),rgb16[:,:,::-1]))
        prepared = prepare_background(self.recipe(path))
        np.testing.assert_allclose(np.frombuffer(prepared.linear_rgb_f32,'<f4').reshape(2,3,3),_linear(rgb16/np.float32(65535)),rtol=1e-6,atol=1e-7)
        jpeg = self.root/'oriented.jpg'
        ok,encoded = cv2.imencode('.jpg',self.rgb[:,:,::-1])
        self.assertTrue(ok)
        # Real JPEG EXIF orientation6 (90deg), which this explicit raw-pixel policy ignores.
        tiff = b'II'+struct.pack('<HIH',42,8,1)+struct.pack('<HHIHHI',274,3,1,6,0,0)
        exif = b'Exif\0\0'+tiff
        raw = encoded.tobytes()
        jpeg.write_bytes(raw[:2]+b'\xff\xe1'+struct.pack('>H',len(exif)+2)+exif+raw[2:])
        decoded = cv2.imdecode(np.frombuffer(jpeg.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.shape,(2,3,3))
        prepared = prepare_background(self.recipe(jpeg))
        np.testing.assert_allclose(np.frombuffer(prepared.linear_rgb_f32,'<f4').reshape(2,3,3),_linear(decoded[:,:,::-1]/np.float32(255)),rtol=1e-6,atol=1e-7)

    def test_wrong_channels_geometry_format_and_encoded_budget_refuse_before_decoder(self):
        for channels in (1,4):
            path = self.root/f'channels-{channels}.png'
            image = np.zeros((2,3),np.uint8) if channels==1 else np.zeros((2,3,4),np.uint8)
            self.assertTrue(cv2.imwrite(str(path),image))
            with patch('cv2.imdecode',side_effect=AssertionError('no native decode')),self.assertRaisesRegex(ValueError,'RGB'):
                prepare_background(self.recipe(path))
        for changes in ({'expected_hw':(3,2)},{'max_encoded_bytes':10},{'profile':'linear_bt709_d65'}):
            with self.subTest(changes=changes),patch('cv2.imdecode',side_effect=AssertionError('no decode')),self.assertRaises(ValueError):
                prepare_background(self.recipe(**changes))

    def test_alias_hardlink_parent_alias_and_decode_contract_refused(self):
        link = self.root/'alias.png'
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            prepare_background(self.recipe(link))
        link.unlink()
        os.link(self.path,link)
        with self.assertRaises(ValueError):
            self.prepared()
        link.unlink()
        parent = self.root/'alias-dir'
        parent.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare_background(self.recipe(parent/self.path.name))
        invalid_decodes = (None,np.zeros((3,2,3),np.uint8),np.zeros((2,3,4),np.uint8),np.zeros((2,3,3),np.float32))
        for decoded in invalid_decodes:
            with self.subTest(decoded=type(decoded)),patch('cv2.imdecode',return_value=decoded),self.assertRaises(ValueError):
                self.prepared()
        exr = self.root/'nonfinite.exr'
        values = np.zeros((2,3,3),np.float32)
        values[0,0,0] = np.nan
        self.assertTrue(cv2.imwrite(str(exr),values))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            prepare_background(self.recipe(exr,profile='linear_bt709_d65'))

    def test_growth_during_read_and_path_replacement_during_decode_refused(self):
        raw = self.path.read_bytes()
        calls = [0]
        def check():
            calls[0] += 1
            if calls[0]==3:
                with self.path.open('ab') as stream:
                    stream.write(b'growth')
        with self.assertRaises(ValueError):
            prepare_background(self.recipe(),check=check)
        self.path.write_bytes(raw)
        decode = cv2.imdecode
        def replaced(*args):
            image = decode(*args)
            other = self.root/'replacement.png'
            other.write_bytes(raw)
            other.replace(self.path)
            return image
        with patch('cv2.imdecode',side_effect=replaced),self.assertRaisesRegex(ValueError,'changed'):
            self.prepared()

    def test_cooperative_cancellation_before_io_after_decode_and_during_compose(self):
        token = CancellationToken()
        token.cancel('fixture before IO')
        with patch('os.open',side_effect=AssertionError('no IO')),self.assertRaises(CancelledError):
            prepare_background(self.recipe(),check=token.check)
        token = CancellationToken()
        decode = cv2.imdecode
        def cancelled(*args):
            image = decode(*args)
            token.cancel('after native decode')
            return image
        with patch('cv2.imdecode',side_effect=cancelled),self.assertRaises(CancelledError):
            prepare_background(self.recipe(),check=token.check)
        background = self.prepared()
        calls = [0]
        def compose_check():
            calls[0] += 1
            if calls[0]==2:
                raise CancelledError('fixture after transfer')
        foreground = np.ones((2,3,3),np.float32)
        alpha = np.ones((2,3,1),np.float32)
        before = foreground.tobytes(),alpha.tobytes()
        with self.assertRaises(CancelledError):
            compose_photo(foreground,alpha,background,foreground_linear=False,check=compose_check)
        self.assertEqual(before,(foreground.tobytes(),alpha.tobytes()))

    def test_composition_and_immutable_instance_contracts_refuse_invalid_inputs(self):
        background = self.prepared()
        foreground = np.ones((2,3,3),np.float32)
        alpha = np.ones((2,3,1),np.float32)
        for bad_fg,bad_alpha in ((foreground.astype(np.float64),alpha),(foreground,alpha.astype(np.float64)),
                                 (foreground,np.full((2,3,1),1.1,np.float32)),(foreground,np.full((2,3,1),np.nan,np.float32)),
                                 (np.full((2,3,3),np.inf,np.float32),alpha),(foreground,np.ones((3,2),np.float32))):
            with self.assertRaises(ValueError):
                compose_photo(bad_fg,bad_alpha,background,foreground_linear=True)
        invalid:Any = PreparedBackground
        with self.assertRaises(TypeError):
            compose_photo(foreground,alpha,invalid,foreground_linear=True)
        with self.assertRaises(ValueError):
            replace(background,linear_rgb_f32=bytes(len(background.linear_rgb_f32)))
        with self.assertRaises(ValueError):
            replace(background,native_hw=(3,2))
        with self.assertRaises(ValueError):
            replace(background,identity='0'*64)
        with self.assertRaises(TypeError):
            json.dumps({1,'mixed'})

    def test_final_callback_mutation_is_fenced_before_publication(self):
        from corridorkey.project_collection import assert_pin as actual_assert
        observed = [False]
        def assert_after_callback(pin):
            # The last cooperative callback must precede this final fence.
            observed[0] = True
            actual_assert(pin)
        def check():
            if observed[0]:
                raise AssertionError('No callback may follow final source identity fence')
        with patch('corridorkey.preview_background.assert_pin',side_effect=assert_after_callback):
            self.prepared_with_check(check)
        original = cv2.imdecode
        decoded = [False]
        def mark(*args):
            value = original(*args)
            decoded[0] = True
            return value
        def mutate():
            if decoded[0]:
                raw = self.path.read_bytes()
                self.path.write_bytes(raw)
        with patch('cv2.imdecode',side_effect=mark),self.assertRaisesRegex(ValueError,'changed'):
            prepare_background(self.recipe(),check=mutate)

    def prepared_with_check(self,check):
        return prepare_background(self.recipe(),check=check)

    def test_public_prepared_provenance_work_geometry_profile_and_finite_byte_admission(self):
        background = self.prepared()
        def changed(**updates):
            metadata = background.provenance
            metadata.update(updates)
            raw = json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()
            return replace(background,provenance_json=raw,identity=hashlib.sha256(raw).hexdigest())
        for update in ({'max_work_bytes':1},{'work_bytes_estimate':1},{'max_pixels':1},{'geometry':'crop'},
                       {'storage_channels':True},{'native_hw':[True,3]},{'storage_format':'exr'},
                       {'icc_policy':'ICC converted silently'},{'extra':'unknown'}):
            with self.subTest(update=update),self.assertRaises((ValueError,TypeError)):
                changed(**update)
        metadata = background.provenance
        values = np.full((2,3,3),np.nan,np.float32).astype('<f4').tobytes()
        metadata['linear_rgb_sha256'] = hashlib.sha256(values).hexdigest()
        raw = json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            replace(background,linear_rgb_f32=values,provenance_json=raw,identity=hashlib.sha256(raw).hexdigest())

    def test_readonly_and_negative_stride_foreground_transfer_preserves_caller_views(self):
        background = self.prepared()
        stored = np.linspace(-.2,1.2,18,dtype=np.float32).reshape(2,3,3)
        readonly = stored.copy()
        readonly.setflags(write=False)
        reversed_view = stored[:,::-1]
        alpha = np.full((2,3,1),.5,np.float32)
        backdrop = np.frombuffer(background.linear_rgb_f32,'<f4').reshape(2,3,3)
        for foreground in (readonly,reversed_view):
            before = foreground.tobytes(),foreground.flags.writeable,foreground.strides,alpha.tobytes()
            actual = compose_photo(foreground,alpha,background,foreground_linear=False)
            np.testing.assert_allclose(actual,.5*_linear(foreground)+.5*backdrop,rtol=2e-6,atol=1e-7)
            self.assertEqual(before,(foreground.tobytes(),foreground.flags.writeable,foreground.strides,alpha.tobytes()))
            self.assertFalse(np.shares_memory(actual,foreground))
