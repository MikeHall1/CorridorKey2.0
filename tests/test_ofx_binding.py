"""Independent native source-binding fixtures; no inference or Resolve claims."""
import hashlib
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = b'CorridorKey/source/linear-rgb-f32/v1\0'


def bound_fixture(width=5, height=4):
    count = 3
    record = hashlib.sha256(b'explicit independent source-binding fixture').digest()
    prefix = struct.pack('<8sIIIqd32s', b'CKOFX002', width, height, count, 7, 24., record)
    payloads, table = [], bytearray()
    for index in range(count):
        source, output = bytearray(), bytearray()
        for top in range(height):
            for x in range(width):
                source.extend(struct.pack('<fff', (index + 1) / 4 + x / 16, (top + 1) / 8, -.5))
                output.extend(struct.pack('<ffff', x / 2 - 1, top + index / 4, 2., (x + top) / (width + height - 2)))
        payloads.append(output)
        table.extend(hashlib.sha256(record + struct.pack('<I', index) + output).digest())
        table.extend(hashlib.sha256(DOMAIN + struct.pack('<III', index, width, height) + source).digest())
    return prefix + hashlib.sha256(prefix + table).digest() + table + b''.join(payloads)


class OfxBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('Native compiler unavailable: bound source checks NOT VERIFIED')
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.host = cls.root / 'host'
        cls.probe = cls.root / 'probe'
        common = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        subprocess.run([*common, '-I', str(ROOT / 'third_party/OpenFX/include'),
                        str(ROOT / 'native/ofx/host_test.cpp'), str(ROOT / 'native/ofx/adapter.cpp'),
                        '-o', str(cls.host)], check=True, capture_output=True)
        subprocess.run([*common, str(ROOT / 'native/ofx/cache_probe.cpp'), '-o', str(cls.probe)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_exact_all_source_dependencies_before_output_with_signed_strides(self):
        cache = self.root / 'source-bound.ckcache'
        cache.write_bytes(bound_fixture())
        result = subprocess.run([str(self.host), str(cache), str(self.root)], check=True,
                                capture_output=True, text=True, timeout=30)
        self.assertIn('host ABI tests passed', result.stdout)
        self.assertNotIn('runtime error', result.stderr)
        self.assertNotIn('AddressSanitizer', result.stderr)
        for instance in (0, 1):
            for index in (0, 1, 2):
                raw = (self.root / f'{instance}-{index}.rgba').read_bytes()
                actual = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
                for top in range(4):
                    for x in range(5):
                        alpha = (x + top) / 7
                        offset = (top * 5 + x) * 4
                        expected = ((x / 2 - 1) * alpha, (top + index / 4) * alpha, 2 * alpha, alpha)
                        for channel in range(4):
                            self.assertAlmostEqual(actual[offset + channel], expected[channel], places=6)

    def test_bound_table_integrity_and_admission(self):
        for mutation in ('source_digest', 'output_digest', 'count', 'version'):
            with self.subTest(mutation=mutation):
                raw = bytearray(bound_fixture())
                if mutation == 'source_digest':
                    raw[132] ^= 1
                elif mutation == 'output_digest':
                    raw[100] ^= 1
                elif mutation == 'count':
                    struct.pack_into('<I', raw, 16, 16385)
                else:
                    raw[:8] = b'CKOFX003'
                cache = self.root / f'{mutation}.ckcache'
                cache.write_bytes(raw)
                result = subprocess.run([str(self.probe), str(cache), '0'], capture_output=True,
                                        text=True, check=False, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('AddressSanitizer', result.stderr)

    def test_source_hash_chunk_boundary_preserves_full_rows(self):
        width, height = 4101, 2
        cache = self.root / 'chunk-boundary.ckcache'
        cache.write_bytes(bound_fixture(width, height))
        result = subprocess.run([str(self.host), str(cache), str(self.root)], check=True,
                                capture_output=True, text=True, timeout=30)
        self.assertIn('host ABI tests passed', result.stdout)
        self.assertNotIn('runtime error', result.stderr)
        self.assertNotIn('AddressSanitizer', result.stderr)
        for instance in (0, 1):
            for index in (0, 1, 2):
                raw = (self.root / f'{instance}-{index}.rgba').read_bytes()
                for top in (0, height - 1):
                    for x in (0, 4095, 4096, 4100):
                        actual = struct.unpack_from('<ffff', raw, (top * width + x) * 16)
                        alpha = (x + top) / (width + height - 2)
                        expected = ((x / 2 - 1) * alpha, (top + index / 4) * alpha, 2 * alpha, alpha)
                        # Native float32 multiplication rounds after alpha decode.
                        for channel in range(4):
                            self.assertAlmostEqual(actual[channel], expected[channel], delta=max(1e-6, abs(expected[channel]) * 1e-7))
