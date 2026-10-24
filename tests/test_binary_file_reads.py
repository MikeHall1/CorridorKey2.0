"""Raw descriptor readers preserve binary bytes instead of CRT text translation."""
import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.corrections import load_tracked_asset
from corridorkey.durable_writer_prefix import DurableWriterPrefix
from corridorkey.native_image_admission import _read as read_image
from corridorkey.portable_project import _metadata
from corridorkey.project_collection import admit_file, read_file
from corridorkey.trimap_refinement import _read as read_trimap


class BinaryFileReadTests(unittest.TestCase):
    def test_raw_file_readers_request_binary_mode_and_preserve_all_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / 'binary.bin'
            payload = b'\x89PNG\r\n\x1a\nraw-binary\r\n\x00payload'
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            tracked = root / 'tracked.exr'
            # Valid uncompressed alpha samples contain CRLF and Ctrl-Z bytes.
            pixels = np.full((2, 3), struct.unpack('<f', b'\r\n\x1a>')[0], np.float32)
            self.assertTrue(cv2.imwrite(str(tracked), pixels,
                                        [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_NO]))
            encoded = tracked.read_bytes()
            self.assertIn(b'\r\n\x1a', encoded)
            owner: Any = SimpleNamespace(store=SimpleNamespace(max_bytes=4096),
                                         stats={'receipt_bytes_hashed': 0}, _check=lambda: None)
            decoded = []
            original_decode = cv2.imdecode

            def decode(buffer, flags):
                decoded.append(buffer.tobytes())
                return original_decode(buffer, flags)

            readers = [
                ('image', lambda: read_image(path, 4096, lambda: None)[0], payload),
                ('trimap', lambda: read_trimap(path, lambda: None)[0], payload),
                ('collection', lambda: read_file(admit_file(path), capture=True)[1], payload),
                ('metadata', lambda: _metadata(admit_file(path), 4096, None), payload),
                ('writer', lambda: DurableWriterPrefix._receipt(owner, path),
                 {'bytes': len(payload), 'sha256': digest}),
                ('tracked', lambda: load_tracked_asset(tracked, expected_hw=(2, 3),
                                                       expected_sha256=hashlib.sha256(encoded).hexdigest()), pixels),
            ]
            original_open, original_read = os.open, os.read
            native_binary = getattr(os, 'O_BINARY', 0)
            binary_flag = native_binary or 1 << 29
            for name, reader, expected in readers:
                with self.subTest(reader=name):
                    opened_flags, used, eof = {}, set(), set()

                    def opened(target, flags, *args, opened_flags=opened_flags, **kwargs):
                        descriptor = original_open(target, (flags & ~binary_flag) | native_binary, *args, **kwargs)
                        opened_flags[descriptor] = flags
                        return descriptor

                    def read(descriptor, count, used=used, opened_flags=opened_flags, eof=eof):
                        used.add(descriptor)
                        if opened_flags[descriptor] & binary_flag:
                            return original_read(descriptor, count)
                        if descriptor in eof:
                            return b''
                        block = original_read(descriptor, count)
                        if b'\x1a' in block:
                            block = block.split(b'\x1a', 1)[0]
                            eof.add(descriptor)
                        return block.replace(b'\r\n', b'\n')

                    with (patch.object(os, 'O_BINARY', binary_flag, create=True),
                          patch.object(os, 'open', opened), patch.object(os, 'read', read),
                          patch('cv2.imdecode', side_effect=decode)):
                        actual = reader()
                    if name == 'tracked':
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(decoded, [encoded])
                    else:
                        self.assertEqual(actual, expected)
                    self.assertTrue(used)
                    self.assertTrue(all(opened_flags[descriptor] & binary_flag for descriptor in used))
                    self.assertEqual(path.read_bytes(), payload)
                    self.assertEqual(tracked.read_bytes(), encoded)
