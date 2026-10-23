import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from corridorkey.ofx_cache import seal
from tests import test_quick_runtime

ROOT = Path(__file__).resolve().parents[1]


class OfxCacheTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job

    def test_real_exports_native_random_access_and_corruption(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('Native C++ compiler unavailable: OFX reader NOT VERIFIED')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, diagnostics = self.run_job(folder, settings_overrides={'start_frame': 2, 'num_frames': 4})
            record = {'schema_version': 1, 'status': 'complete', 'input': str(root / 'source'), 'output_dir': str(root / 'output'),
                      'diagnostics': diagnostics.snapshot()}
            report = root / 'record.json'
            report.write_text(json.dumps(record))
            cache = seal(report, root)
            self.assertIn('source-rgb-draft', cache.name)
            binary = root / 'probe'
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                            str(ROOT / 'native/ofx/cache_probe.cpp'), '-o', str(binary)], check=True, capture_output=True)
            for index in (3, 0, 2, 1, 0):
                result = subprocess.run([str(binary), str(cache), str(index)], capture_output=True, text=True, check=True)
                values = result.stdout.split()
                self.assertEqual(values[:4], ['64', '64', '4', '2'])
                self.assertAlmostEqual(float(values[-1]), .4, places=5)
            for mutation in ('payload', 'table', 'overflow'):
                broken = root / f'{mutation}.ckcache'
                data = bytearray(cache.read_bytes())
                if mutation == 'payload':
                    data[-1] ^= 1
                elif mutation == 'table':
                    data[100] ^= 1
                else:
                    struct.pack_into('<II', data, 8, 2**31, 2**31)
                broken.write_bytes(data)
                result = subprocess.run([str(binary), str(broken), '3'], capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(cache.parent.glob('.cache-stage')))
            host = root / 'host'
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                            '-I', str(ROOT / 'third_party/OpenFX/include'), str(ROOT / 'native/ofx/host_test.cpp'),
                            str(ROOT / 'native/ofx/adapter.cpp'), '-o', str(host)], check=True, capture_output=True)
            result = subprocess.run([str(host), str(cache), str(root)], capture_output=True, text=True, check=True)
            self.assertIn('host ABI tests passed', result.stdout)
            import numpy as np

            from corridorkey.video_io import read_image_rgb
            foreground = sorted(Path(diagnostics.outputs['fg']).glob('*.exr'))
            alpha = sorted(Path(diagnostics.outputs['alpha']).glob('*.exr'))
            for instance in (0, 1):
                for index in (0, 2, 3):
                    output = np.fromfile(root / f'{instance}-{index}.rgba', dtype=np.float32).reshape(64, 64, 4)
                    a = read_image_rgb(str(alpha[index]))[..., :1]
                    expected = np.concatenate((read_image_rgb(str(foreground[index])) * a, a), axis=2)
                    np.testing.assert_allclose(output, expected, atol=1e-7)

    def test_nonuniform_signed_hdr_host_rows_under_address_and_undefined_sanitizers(self):
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('Native compiler unavailable: sanitizer OFX check NOT VERIFIED')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            width, height = 5, 4
            y, x = np.indices((height, width))
            fg = np.stack(((x-2)/2, (y+1)*1.25, np.full_like(x, .25, dtype=np.float32)), axis=2).astype(np.float32)
            alpha = ((x+y)/(width+height-2)).astype(np.float32)[..., None]
            outputs = {}
            for kind, pixels in [('fg', fg), ('alpha', alpha)]:
                directory = root / 'output' / kind
                writer = ExrSequenceWriter(str(directory), 24., (width, height), channels=pixels.shape[2], dtype='float', frame_name=kind)
                try:
                    for _ in range(3):
                        writer.write(pixels[..., ::-1] if kind == 'fg' else pixels)
                finally:
                    writer.release()
                outputs[kind] = str(directory)
            record = {'schema_version': 1, 'status': 'complete', 'output_dir': str(root / 'output'), 'diagnostics': {
                'source': {'native_hw': [height, width], 'start_frame': 7, 'num_frames': 3, 'native_fps': 24.},
                'frames_submitted': 3, 'outputs': outputs, 'output_contracts': {
                    'fg': {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear', 'foreground_mode': 'model_reconstructed'},
                    'alpha': {'format': 'exr', 'channels': 1}}, 'execution': {'neural_inference': False, 'fixture': 'explicit HDR codec fixture'}}}
            report = root / 'record.json'
            report.write_text(json.dumps(record))
            cache = seal(report, root)
            host = root / 'sanitized-host'
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic', '-fsanitize=address,undefined',
                            '-fno-omit-frame-pointer', '-I', str(ROOT / 'third_party/OpenFX/include'),
                            str(ROOT / 'native/ofx/host_test.cpp'), str(ROOT / 'native/ofx/adapter.cpp'), '-o', str(host)],
                           check=True, capture_output=True)
            result = subprocess.run([str(host), str(cache), str(root)], capture_output=True, text=True, check=True)
            self.assertNotIn('runtime error', result.stderr)
            self.assertNotIn('AddressSanitizer', result.stderr)
            expected = np.concatenate((fg * alpha, alpha), axis=2)
            for instance in (0, 1):
                for index in (0, 1, 2):
                    output = np.fromfile(root / f'{instance}-{index}.rgba', dtype=np.float32).reshape(height, width, 4)
                    np.testing.assert_allclose(output, expected, atol=1e-7)
            record['schema_version'] = 2
            report.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'schema_version'):
                seal(report, root)
