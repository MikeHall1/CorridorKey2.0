import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.identities import file_sha256
from corridorkey.ofx_cache import seal


def exports(root: Path) -> tuple[Path, dict]:
    output = root / 'output'
    for kind, channels in [('fg', 3), ('alpha', 1)]:
        folder = output / kind
        folder.mkdir(parents=True)
        for index in range(2):
            shape = (2, 3, channels) if channels == 3 else (2, 3)
            cv2.imwrite(str(folder / f'{kind}_{index:06d}.exr'), np.full(shape, .4, np.float32))
    record = {'schema_version': 1, 'status': 'complete', 'output_dir': str(output), 'diagnostics': {
        'source': {'native_hw': [2, 3], 'num_frames': 2, 'start_frame': 0, 'native_fps': 24.},
        'frames_submitted': 2, 'outputs': {kind: str(output / kind) for kind in ('fg', 'alpha')},
        'output_contracts': {'fg': {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear'},
                             'alpha': {'format': 'exr', 'channels': 1}}}}
    path = root / 'record.json'
    path.write_text(json.dumps(record))
    return path, record


class OfxSealTests(unittest.TestCase):
    def test_invalid_metadata_is_rejected_before_cache_directory_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, record = exports(root)
            for name, value in [('native_hw', [2]), ('native_fps', True), ('num_frames', True), ('start_frame', -1)]:
                changed = json.loads(json.dumps(record))
                changed['diagnostics']['source'][name] = value
                path.write_text(json.dumps(changed))
                with self.subTest(field=name), self.assertRaises((TypeError, ValueError)):
                    seal(path, root)
                self.assertFalse(list(root.glob('ofx-cache-*')))

    def test_actual_unexpected_raster_channels_reject_without_publication(self):
        for kind, channels in [('alpha', 3), ('fg', 4)]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record, _ = exports(root)
                target = root / 'output' / kind / f'{kind}_000000.exr'
                cv2.imwrite(str(target), np.full((2, 3, channels), .4, np.float32))
                with self.assertRaisesRegex(ValueError, 'channel layout'):
                    seal(record, root)
                self.assertFalse(list(root.rglob('*.ckcache')))
                self.assertFalse(list(root.rglob('.cache-stage')))

    def test_cancel_during_conversion_and_before_publication_preserves_exports(self):
        for stage in ('ofx_convert_frames', 'ofx_publish'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record, _ = exports(root)
                before = {p: file_sha256(p) for p in (root / 'output').rglob('*.exr')}
                token = CancellationToken()

                def progress(event, stage=stage, token=token):
                    if event.stage == stage and event.completed > 0:
                        token.cancel('explicit test cancel')

                with self.assertRaises(CancelledError):
                    seal(record, root, control=JobControl(token, progress))
                self.assertFalse(list(root.rglob('*.ckcache')))
                self.assertFalse(list(root.rglob('.cache-stage')))
                self.assertEqual(before, {p: file_sha256(p) for p in before})

    def test_cancel_after_committed_publication_reports_actual_complete_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record, _ = exports(root)
            token = CancellationToken()
            publish = os.link

            def late_cancel(source, destination):
                publish(source, destination)
                token.cancel('arrived after publication')

            with patch('corridorkey.ofx_cache.os.link', side_effect=late_cancel):
                cache = seal(record, root, control=JobControl(token))
            self.assertTrue(cache.is_file())
            self.assertTrue(token.cancelled)

    def test_committed_cleanup_error_retains_authoritative_cache_and_primary_cancel(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record, _ = exports(root)
                unlink = Path.unlink

                def denied(path, *args, unlink=unlink, **kwargs):
                    if path.name == '.cache-stage':
                        raise PermissionError('injected owned cleanup denial')
                    return unlink(path, *args, **kwargs)

                token = CancellationToken()

                def progress(event, token=token, cancel=cancel):
                    if cancel and event.stage == 'ofx_publish':
                        token.cancel('primary cancellation')

                with patch.object(Path, 'unlink', denied), self.assertLogs('corridorkey.ofx_cache', level='WARNING'):
                    if cancel:
                        with self.assertRaises(CancelledError) as raised:
                            seal(record, root, control=JobControl(token, progress))
                        self.assertIn('Owned stage cleanup failed', str(raised.exception.__notes__))
                    else:
                        self.assertTrue(seal(record, root).is_file())
