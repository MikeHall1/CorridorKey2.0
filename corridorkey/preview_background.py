"""Captured RGB photo backgrounds for preview only; no rendering or model work.

Profiles are user declarations: PNG/JPEG is assumed sRGB; EXR is declared linear
BT.709/D65. Embedded ICC and EXIF orientation are not interpreted or converted.
Stored pixel order is retained. Exact native geometry is required: no crop,
resize, orientation correction or display conversion occurs in this module.

The declared work estimate reserves sixteen float32 RGB equivalents plus two
encoded-byte buffers and fixed overhead. It is conservative admission accounting,
not a hard process RSS cap or a codec allocation guarantee. Preparation and math
belong on the existing preview worker; capture() does no filesystem I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .native_image_admission import MAX_PIXELS, admit_image
from .project_collection import admit_file, assert_pin

MAX_ENCODED_BYTES = 32 * 1024**2
MAX_WORK_BYTES = 512 * 1024**2
MAX_EXTENT = 8192
MAX_PROVENANCE_BYTES = 16 * 1024
PROFILES = ('assumed_srgb','linear_bt709_d65')
VERSION = 'photo-linear709-native-exact-v1'
WORKING_BASIS = 'linear BT.709/D65; engine/model working-basis assumption, not a display calibration'
ICC_POLICY = 'Embedded ICC/colour metadata ignored; explicit caller profile assumed, no ICC/OCIO conversion'
ORIENTATION_POLICY = 'Stored pixel order; EXIF orientation ignored; no rotation/crop/resize'
SNAPSHOT_POLICY = 'Captured immutable file snapshot; reselect to reload later file changes'
WORK_SCOPE = 'Conservative admission estimate; not a hard process RSS or codec allocation cap'
Check = Callable[[],None] | None


def _check(check: Check) -> None:
    if check is not None:
        check()


def _integer(value: Any,low: int,high: int,name: str) -> None:
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f'{name} requires an exact integer in [{low},{high}]')


def _shape(value: Any) -> None:
    if type(value) is not tuple or len(value)!=2:
        raise ValueError('Photo native geometry requires an immutable (height,width) tuple')
    for side in value:
        _integer(side,1,MAX_EXTENT,'Photo native extent')
    _integer(value[0]*value[1],1,MAX_PIXELS,'Photo native pixel count')


def _canonical(value: Any) -> bytes:
    raw = json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_PROVENANCE_BYTES:
        raise ValueError('Photo provenance exceeds the byte bound')
    return raw


def _pairs(items):
    result = {}
    for key,value in items:
        if key in result:
            raise ValueError('Duplicate photo provenance field')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Photo provenance must be finite JSON')


@dataclass(frozen=True)
class BackgroundRecipe:
    path: str
    profile: str
    expected_hw: tuple[int,int]
    geometry: str = 'native_exact'
    max_encoded_bytes: int = MAX_ENCODED_BYTES
    max_pixels: int = MAX_PIXELS
    max_work_bytes: int = MAX_WORK_BYTES

    def __post_init__(self):
        if (not isinstance(self.path,str) or not self.path or len(self.path.encode())>32768
                or any(ord(char)<32 for char in self.path) or not Path(self.path).is_absolute()
                or '..' in Path(self.path).parts):
            raise ValueError('Photo path requires explicit absolute local text without traversal/control characters')
        if self.profile not in PROFILES or self.geometry!='native_exact':
            raise ValueError('Choose assumed_srgb or linear_bt709_d65 and native_exact photo geometry')
        _shape(self.expected_hw)
        _integer(self.max_encoded_bytes,1,MAX_ENCODED_BYTES,'Photo encoded byte bound')
        _integer(self.max_pixels,1,MAX_PIXELS,'Photo pixel bound')
        _integer(self.max_work_bytes,1,MAX_WORK_BYTES,'Photo work byte bound')
        if self.expected_hw[0]*self.expected_hw[1]>self.max_pixels:
            raise ValueError('Photo expected native geometry exceeds its pixel budget')
        if self.work_bytes_estimate>self.max_work_bytes:
            raise ValueError('Photo conservative work estimate exceeds its declared work budget (not a hard RSS cap)')

    @property
    def work_bytes_estimate(self) -> int:
        return self.expected_hw[0]*self.expected_hw[1]*192+2*self.max_encoded_bytes+2*1024**2

    @classmethod
    def capture(cls,path: str | Path,*,profile: str,expected_hw: tuple[int,int],geometry: str='native_exact',
                max_encoded_bytes: int=MAX_ENCODED_BYTES,max_pixels: int=MAX_PIXELS,
                max_work_bytes: int=MAX_WORK_BYTES) -> BackgroundRecipe:
        if not isinstance(path,(str,Path)):
            raise TypeError('Photo capture requires path text or a Path instance')
        return cls(str(path),profile,expected_hw,geometry,max_encoded_bytes,max_pixels,max_work_bytes)


@dataclass(frozen=True)
class PreparedBackground:
    identity: str
    sha256: str
    native_hw: tuple[int,int]
    profile: str
    linear_rgb_f32: bytes = field(repr=False)
    provenance_json: bytes = field(repr=False)

    def __post_init__(self):
        _shape(self.native_hw)
        if self.profile not in PROFILES:
            raise ValueError('Prepared photo has an unsupported declared profile')
        if (type(self.linear_rgb_f32) is not bytes or len(self.linear_rgb_f32)!=self.native_hw[0]*self.native_hw[1]*12
                or type(self.provenance_json) is not bytes or len(self.provenance_json)>MAX_PROVENANCE_BYTES):
            raise ValueError('Prepared photo requires bounded immutable float32 RGB/provenance bytes')
        if any(not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value) for value in (self.identity,self.sha256)):
            raise ValueError('Prepared photo identities require lowercase SHA256')
        data = json.loads(self.provenance_json,object_pairs_hook=_pairs,parse_constant=_constant)
        fields = {'version','path','content_sha256','linear_rgb_sha256','native_hw','profile','geometry',
                  'storage_format','storage_bit_depth','storage_channels','working_basis','icc_policy',
                  'orientation_policy','snapshot_policy','work_bytes_estimate','max_work_bytes',
                  'max_encoded_bytes','max_pixels','work_scope'}
        if not isinstance(data,dict) or set(data)!=fields:
            raise ValueError('Prepared photo provenance requires the complete exact metadata fields')
        recipe = BackgroundRecipe.capture(data['path'],profile=self.profile,expected_hw=self.native_hw,
                                         geometry=data['geometry'],max_encoded_bytes=data['max_encoded_bytes'],
                                         max_pixels=data['max_pixels'],max_work_bytes=data['max_work_bytes'])
        formats = ('png','jpeg') if self.profile=='assumed_srgb' else ('exr',)
        depths = (8,) if data['storage_format']=='jpeg' else (16,32) if data['storage_format']=='exr' else (8,16)
        if (data['storage_format'] not in formats or type(data['storage_bit_depth']) is not int or data['storage_bit_depth'] not in depths
                or type(data['storage_channels']) is not int or data['storage_channels']!=3
                or type(data['work_bytes_estimate']) is not int or data['work_bytes_estimate']!=recipe.work_bytes_estimate
                or data['working_basis']!=WORKING_BASIS or data['icc_policy']!=ICC_POLICY
                or data['orientation_policy']!=ORIENTATION_POLICY or data['snapshot_policy']!=SNAPSHOT_POLICY
                or data['work_scope']!=WORK_SCOPE or not isinstance(data['native_hw'],list)
                or any(type(side) is not int for side in data['native_hw'])):
            raise ValueError('Prepared photo provenance geometry/work/storage/interpretation proof differs')
        if (_canonical(data)!=self.provenance_json
                or hashlib.sha256(self.provenance_json).hexdigest()!=self.identity
                or data.get('content_sha256')!=self.sha256 or data.get('profile')!=self.profile
                or data.get('native_hw')!=list(self.native_hw) or data.get('version')!=VERSION
                or data.get('linear_rgb_sha256')!=hashlib.sha256(self.linear_rgb_f32).hexdigest()):
            raise ValueError('Prepared photo pixels/provenance/identity differ')
        import numpy as np
        if not np.isfinite(np.frombuffer(self.linear_rgb_f32,dtype='<f4')).all():
            raise ValueError('Prepared photo contains nonfinite linear RGB')

    @property
    def provenance(self) -> dict[str,Any]:
        return json.loads(self.provenance_json)


def prepare_background(recipe: BackgroundRecipe,*,check: Check=None) -> PreparedBackground:
    if not isinstance(recipe,BackgroundRecipe):
        raise TypeError('Photo preparation requires a BackgroundRecipe instance')
    _check(check)
    pin = admit_file(recipe.path)
    admitted = admit_image(recipe.path,expected_hw=recipe.expected_hw,max_pixels=recipe.max_pixels,
                           max_bytes=recipe.max_encoded_bytes,check=check)
    if admitted.channels!=3:
        raise ValueError('Photo backgrounds require RGB without an alpha or grayscale channel')
    if ((recipe.profile=='assumed_srgb' and admitted.format not in ('png','jpeg'))
            or (recipe.profile=='linear_bt709_d65' and admitted.format!='exr')):
        raise ValueError('Photo declared profile differs from supported PNG/JPEG sRGB or EXR linear709 storage; choose explicitly')
    _check(check)
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(admitted.encoded_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
    _check(check)
    if decoded is None or decoded.ndim!=3 or decoded.shape!=(recipe.expected_hw[0],recipe.expected_hw[1],3):
        raise ValueError('Decoded photo geometry/channels differ from its admitted RGB header')
    expected_dtype = np.float32 if admitted.format=='exr' else np.uint16 if admitted.bit_depth==16 else np.uint8
    if decoded.dtype!=expected_dtype:
        raise ValueError('Decoded photo dtype differs from its admitted storage contract')
    rgb = np.asarray(decoded[:,:,::-1],dtype=np.float32,order='C')
    if admitted.format!='exr':
        rgb /= 65535. if admitted.bit_depth==16 else 255.
    if not np.isfinite(rgb).all():
        raise ValueError('Photo contains nonfinite RGB values')
    if recipe.profile=='assumed_srgb':
        from .color_utils import srgb_to_linear
        rgb = srgb_to_linear(rgb)
    _check(check)
    if not np.isfinite(rgb).all():
        raise ValueError('Photo transfer conversion produced nonfinite values')
    immutable = np.ascontiguousarray(rgb,dtype='<f4').tobytes()
    provenance = {'version':VERSION,'path':recipe.path,'content_sha256':admitted.sha256,
                  'linear_rgb_sha256':hashlib.sha256(immutable).hexdigest(),'native_hw':list(recipe.expected_hw),
                  'profile':recipe.profile,'geometry':recipe.geometry,'storage_format':admitted.format,
                  'storage_bit_depth':admitted.bit_depth,'storage_channels':3,
                  'working_basis':WORKING_BASIS,
                  'icc_policy':ICC_POLICY,
                  'orientation_policy':ORIENTATION_POLICY,
                  'snapshot_policy':SNAPSHOT_POLICY,
                  'work_bytes_estimate':recipe.work_bytes_estimate,'max_work_bytes':recipe.max_work_bytes,
                  'max_encoded_bytes':recipe.max_encoded_bytes,'max_pixels':recipe.max_pixels,
                  'work_scope':WORK_SCOPE}
    raw = _canonical(provenance)
    _check(check)
    assert_pin(pin)
    return PreparedBackground(hashlib.sha256(raw).hexdigest(),admitted.sha256,recipe.expected_hw,recipe.profile,immutable,raw)


def compose_photo(foreground,alpha,background: PreparedBackground,*,foreground_linear: bool,check: Check=None):
    """Return fresh linear709 float32 RGB; coverage and caller pixels stay intact."""
    if not isinstance(background,PreparedBackground) or type(foreground_linear) is not bool:
        raise TypeError('Photo composition requires a prepared instance and explicit foreground transfer boolean')
    _check(check)
    import numpy as np

    if (not isinstance(foreground,np.ndarray) or foreground.dtype!=np.float32
            or foreground.shape!=(background.native_hw[0],background.native_hw[1],3)):
        raise ValueError('Photo foreground requires matching native float32 RGB')
    if (not isinstance(alpha,np.ndarray) or alpha.dtype!=np.float32
            or alpha.shape not in (background.native_hw,(*background.native_hw,1))):
        raise ValueError('Photo alpha requires matching float32 coverage HxW or HxWx1')
    if not np.isfinite(foreground).all() or not np.isfinite(alpha).all() or np.any((alpha<0)|(alpha>1)):
        raise ValueError('Photo foreground must be finite and alpha finite coverage in [0,1]')
    coverage = alpha if alpha.ndim==3 else alpha[:,:,None]
    backdrop = np.frombuffer(background.linear_rgb_f32,dtype='<f4').reshape((*background.native_hw,3))
    if not np.isfinite(backdrop).all():
        raise ValueError('Prepared photo contains nonfinite linear RGB')
    if foreground_linear:
        linear = foreground
    else:
        from .color_utils import srgb_to_linear
        linear = srgb_to_linear(np.array(foreground,dtype=np.float32,order='C',copy=True))
    _check(check)
    with np.errstate(over='ignore',invalid='ignore'):
        result = np.asarray(linear*coverage+backdrop*(1-coverage),dtype=np.float32,order='C')
    _check(check)
    if not np.isfinite(result).all():
        raise ValueError('Photo composite overflows finite linear float32 RGB')
    return result
