import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OfxAdapterTests(unittest.TestCase):
    def test_official_headers_and_actual_shared_module_exports(self):
        vendor = ROOT / 'third_party/OpenFX'
        metadata = json.loads((vendor / 'provenance.json').read_text())
        for name, expected in metadata['files'].items():
            raw = (vendor / name).read_bytes()
            self.assertEqual(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(), expected['git_blob_sha'])
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('Native compiler unavailable; native OFX module NOT VERIFIED')
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'CorridorKeyCached.ofx'
            pic = [] if sys.platform == 'win32' else ['-fPIC']
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                            *pic, '-shared', '-I', str(vendor / 'include'), str(ROOT / 'native/ofx/adapter.cpp'),
                            '-o', str(binary)], check=True, capture_output=True)
            code = '''import ctypes,json,os,sys
library=ctypes.CDLL(sys.argv[1])
library.OfxGetNumberOfPlugins.argtypes=[]
library.OfxGetNumberOfPlugins.restype=ctypes.c_int
library.OfxGetPlugin.argtypes=[ctypes.c_int]
library.OfxGetPlugin.restype=ctypes.c_void_p
print(json.dumps({'pid':os.getpid(),'count':library.OfxGetNumberOfPlugins(),
                 'first':bool(library.OfxGetPlugin(0)),'negative':bool(library.OfxGetPlugin(-1)),
                 'past_end':bool(library.OfxGetPlugin(1))}))
'''
            result = subprocess.run([sys.executable, '-c', code, str(binary)], check=True,
                                    capture_output=True, text=True, timeout=20)
            observed = json.loads(result.stdout)
            self.assertNotEqual(observed.pop('pid'), os.getpid())
            self.assertEqual(observed, {'count': 1, 'first': True, 'negative': False, 'past_end': False})
            # The child has exited before temporary DLL cleanup on every platform.
            binary.unlink()
            self.assertFalse(binary.exists())
