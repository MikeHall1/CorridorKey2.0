"""Actual native InstanceChanged capture through fixture suites, not Resolve."""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = b'CorridorKey/source/linear-rgb-f32/v1\0'


class OfxCaptureNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('Native compiler unavailable; OFX capture mechanics NOT VERIFIED')
        cls.build = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.build.name)/'OfxCaptureHostTest'
        subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
                        '-I',str(ROOT/'third_party/OpenFX/include'),str(ROOT/'native/ofx/capture_host_test.cpp'),
                        str(ROOT/'native/ofx/adapter.cpp'),'-o',str(cls.binary)],check=True,capture_output=True,timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def capture(self,scenario,parent=None):
        result = subprocess.run([str(self.binary),str(parent or self.parent),scenario],check=True,capture_output=True,text=True,timeout=10)
        observed = json.loads(result.stdout)
        self.assertEqual(observed['acquired'],observed['released'])
        self.assertEqual(observed['starts'],observed['ends'])
        return observed

    def verify_bundle(self,root):
        self.assertTrue((root/'COMPLETE').is_file())
        self.assertEqual((root/'COMPLETE').stat().st_size,0)
        data = json.loads((root/'capture.json').read_bytes())
        self.assertEqual(set(data),{'schema_version','type','capture_id','source_scope','host_start','frame_count','width','height','fps','pixel_aspect','colour_contract','payload','source_digests'})
        self.assertEqual((data['schema_version'],data['type'],data['source_scope']),(1,'corridorkey_ofx_capture','ofx_instance_upstream'))
        self.assertRegex(data['capture_id'],'^[0-9a-f]{32}$')
        self.assertEqual((data['host_start'],data['frame_count'],data['width'],data['height']),(-2,3,5,3))
        self.assertEqual((data['fps'],data['pixel_aspect']),(23.976,1))
        self.assertEqual(data['colour_contract'],'scene_linear_bt709_d65_opaque_rgb')
        expected_frames = []
        for index in range(3):
            payload = bytearray()
            for top in range(3):
                for x in range(5):
                    payload.extend(struct.pack('<fff',index+top*.25+x*.125,-0.0 if top==0 and x==0 else -2+top*.5,4+index))
            expected_frames.append(bytes(payload))
        raw = (root/'frames.rgb32').read_bytes()
        self.assertEqual(raw,b''.join(expected_frames))  # Negative zero, HDR and row orientation/padding.
        self.assertEqual(data['payload'],{'path':'frames.rgb32','bytes':540,'sha256':hashlib.sha256(raw).hexdigest()})
        self.assertEqual(data['source_digests'],[hashlib.sha256(DOMAIN+struct.pack('<III',index,5,3)+frame).hexdigest() for index,frame in enumerate(expected_frames)])
        self.assertEqual({path.name for path in root.iterdir()},{'frames.rgb32','capture.json','COMPLETE'})
        return data

    def test_real_instancechanged_native_capture_exact_wire_float_bits_and_leases(self):
        observed = self.capture('success')
        self.assertEqual((observed['status'],observed['acquired'],observed['sets'],observed['last']),(0,3,1,True))
        roots = list(self.parent.iterdir())
        self.assertEqual(len(roots),1)
        self.verify_bundle(roots[0])

    def test_progress_v1_reentrancy_numeric_locale_and_postcommit_parameter_failure(self):
        for scenario in ('v1','recursive','locale','setter-failure'):
            with self.subTest(scenario=scenario):
                parent = self.parent/scenario
                parent.mkdir()
                observed = self.capture(scenario,parent)
                self.assertEqual(observed['status'],0)
                self.assertEqual(observed['last'],scenario!='setter-failure')
                self.assertEqual(observed['recursive'],scenario=='recursive')
                self.verify_bundle(next(parent.iterdir()))

    def test_repeated_explicit_capture_uses_fresh_exclusive_roots(self):
        observed = self.capture('repeat')
        self.assertEqual((observed['status'],observed['acquired'],observed['sets']),(0,6,2))
        roots = list(self.parent.iterdir())
        self.assertEqual(len(roots),2)
        first,second = (self.verify_bundle(root) for root in roots)
        self.assertNotEqual(first['capture_id'],second['capture_id'])
        self.assertEqual(first['source_digests'],second['source_digests'])

    def test_missing_capabilities_unconfirmed_controls_and_invalid_ranges_no_images(self):
        for scenario in ('missing-progress','no-temporal','undeclared','range','budget','large-budget','fps','action-scale','other-param','plugin-edited'):
            with self.subTest(scenario=scenario):
                parent = self.parent/scenario
                parent.mkdir()
                observed = self.capture(scenario,parent)
                self.assertNotEqual(observed['status'],0)
                self.assertEqual(observed['acquired'],0)
                self.assertEqual(list(parent.iterdir()),[])
                self.assertFalse(observed['last'])

    def test_cancellation_image_contract_failures_progress_end_and_final_abort_cleanup(self):
        for scenario in ('cancel','abort','alpha','nan','field','image-par','image-scale','stride','end-failure','final-abort'):
            with self.subTest(scenario=scenario):
                parent = self.parent/scenario
                parent.mkdir()
                observed = self.capture(scenario,parent)
                self.assertNotEqual(observed['status'],0)
                self.assertEqual(list(parent.iterdir()),[])
                self.assertFalse(observed['last'])

    def test_unknown_manifest_conflict_preserved_no_commit_and_no_recursive_delete(self):
        observed = self.capture('manifest-conflict')
        self.assertNotEqual(observed['status'],0)
        roots = list(self.parent.iterdir())
        self.assertEqual(len(roots),1)
        self.assertEqual((roots[0]/'capture.json').read_text(),'UNKNOWN FILE MUST SURVIVE')
        self.assertEqual({path.name for path in roots[0].iterdir()},{'capture.json'})
        self.assertFalse(observed['last'])

    @unittest.skipIf(os.name=='nt','Child RLIMIT_FSIZE fault injection is POSIX-only; Windows write failure NOT VERIFIED')
    def test_actual_child_file_size_write_failure_cleans_owned_partial_bytes(self):
        observed = self.capture('write-failure')
        self.assertNotEqual(observed['status'],0)
        self.assertEqual(list(self.parent.iterdir()),[])
        self.assertFalse(observed['last'])

    def test_capture_parent_alias_refused_before_pixel_leases(self):
        target = self.parent/'target'
        target.mkdir()
        alias = self.parent/'alias'
        alias.symlink_to(target,target_is_directory=True)
        observed = self.capture('success',alias)
        self.assertNotEqual(observed['status'],0)
        self.assertEqual(observed['acquired'],0)
        self.assertEqual(list(target.iterdir()),[])
