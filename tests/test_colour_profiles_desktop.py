"""Explicit sequence interpretation reaches viewers; no model-quality claims."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def working709(rgb):
    # Independent rounded BT.2020 D65 -> BT.709 D65 oracle, display tolerance 1 code.
    matrix = np.array([[1.660491, -.587641, -.072850],
                       [-.124550, 1.132900, -.008349],
                       [-.018151, -.100579, 1.118730]], dtype=np.float64)
    return np.asarray(rgb @ matrix.T, dtype=np.float32)


def display_rgb(rgb):
    # Final viewer clipping; preserve signed/HDR data in the source fixture.
    positive = np.maximum(rgb, 0)
    encoded = np.where(positive <= .0031308, 12.92 * positive,
                       1.055 * positive ** (1 / 2.4) - .055)
    return np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)


def write_rgb(path, rgb):
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1])):
        raise RuntimeError('Fixture EXR write failed')


class ColourPreviewTests(unittest.TestCase):
    def test_source_transform_signed_hdr_and_coverage_separate(self):
        from corridorkey.desktop_preview import preview
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rgb = np.array([[[.2, .05, .01], [-.2, .3, 1.7], [.4, .4, .4]]], np.float32)
            source, output, alpha = root / 'source.exr', root / 'output.exr', root / 'alpha.exr'
            write_rgb(source, rgb)
            write_rgb(output, working709(rgb))
            self.assertTrue(cv2.imwrite(str(alpha), np.full((1, 3), .5, np.float32)))
            original = source.read_bytes()
            converted = preview(str(source), 0, linear=True, source_colour_profile='linear_bt2020_d65')
            np.testing.assert_allclose(converted.pixels, display_rgb(working709(rgb)), atol=1)
            np.testing.assert_array_equal(converted.pixels, preview(str(output), 0, linear=True).pixels)
            self.assertFalse(np.array_equal(converted.pixels, preview(str(source), 0, linear=True).pixels))
            np.testing.assert_array_equal(preview(str(source), 0, linear=True).pixels,
                                          preview(str(source), 0, linear=True, source_colour_profile='linear_bt709_d65').pixels)
            self.assertTrue(np.all(preview(str(alpha), 0, coverage=True).pixels == 128))
            self.assertEqual(source.read_bytes(), original)

    def test_unpremultiply_precedes_conversion_and_video_refuses_before_decode(self):
        from corridorkey.desktop_preview import preview
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'premult.exr'
            straight = np.full((2, 3, 3), [.2, .05, .01], np.float32)
            rgba = np.concatenate((straight * .5, np.full((2, 3, 1), .5, np.float32)), axis=-1)
            self.assertTrue(cv2.imwrite(str(path), rgba[..., [2, 1, 0, 3]]))
            result = preview(str(path), 0, linear=True, alpha_mode='premultiplied', source_colour_profile='linear_bt2020_d65')
            np.testing.assert_allclose(result.pixels, display_rgb(working709(straight)), atol=1)
            with patch('corridorkey.desktop_preview.cv2.VideoCapture') as decode:
                with self.assertRaisesRegex(ValueError, 'video'):
                    preview(str(Path(folder) / 'clip.mp4'), 0, linear=True, source_colour_profile='linear_bt2020_d65')
                decode.assert_not_called()
            with self.assertRaises(ValueError):
                preview(str(path), 0, source_colour_profile='linear_bt2020_d65')

    def test_correction_editor_uses_same_source_basis_and_preserves_coverage(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.session import JobRequest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            rgb = np.full((6, 8, 3), [.2, .05, .01], np.float32)
            write_rgb(source / '0.exr', rgb)
            original = (source / '0.exr').read_bytes()
            job = JobRequest.create(source, root / 'out', InferenceConfig(frame_dir_linear=True, source_colour_profile='linear_bt2020_d65'))
            editor = CorrectionEditor(job, root)
            mask = root / 'soft-mask.exr'
            self.assertTrue(cv2.imwrite(str(mask), np.full((6, 8), .5, np.float32)))
            editor.import_mask(mask, kind='keep', frame=0, strength=1, direction='both', span=[0, 1])
            frame, overlay = editor.view(0, editor.document.to_dict()['entries'][0]['id'])
            np.testing.assert_allclose(frame.pixels, display_rgb(working709(rgb)), atol=1)
            self.assertIsNotNone(overlay)
            self.assertTrue(np.all(overlay == 128))
            self.assertEqual((source / '0.exr').read_bytes(), original)


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class ColourProfileDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['colour-profile-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.rgb = np.full((6, 8, 3), [.2, .05, .01], np.float32)
        for index in range(3):
            write_rgb(self.source / f'{index}.exr', self.rgb)
        self.before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(Path(__file__).with_name('worker_fixture.py'))]))
        self.window = DesktopWindow(bridge)
        self.window.set_source(str(self.source))
        self.window.output.setText(str(self.root))
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        from tests.qt_cleanup import dispose_widget
        self.window.close()
        self.window.bridge.thread.join(10)
        self.window.loader.thread.join(10)
        self.window.poll()
        self.assertFalse(self.window.bridge.thread.is_alive())
        self.assertFalse(self.window.loader.thread.is_alive())
        self.assertEqual(self.before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()})
        dispose_widget(self.window)
        self.folder.cleanup()

    def until_preview(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if self.window.pixmap is not None:
                return
            time.sleep(.01)
        self.fail(self.window.preview_note.text())

    def test_common_form_project_and_queued_settings_are_immutable_without_submission(self):
        from PySide6.QtWidgets import QCheckBox, QComboBox

        from corridorkey.desktop_state import load_project, save_project
        queued = self.window.job()
        self.window.open_common_settings()
        panel = self.window.settings_panel
        profile = panel.controls['engine', 'source_colour_profile']
        linear = panel.controls['engine', 'frame_dir_linear']
        assert isinstance(profile, QComboBox) and isinstance(linear, QCheckBox)
        self.assertEqual(profile.currentText(), 'legacy')
        profile.setCurrentText('linear_bt2020_d65')
        with self.assertRaises(ValueError):
            panel.values()
        linear.setChecked(True)
        panel.apply()
        current = self.window.job()
        self.assertEqual(current.configurations()[0].source_colour_profile, 'linear_bt2020_d65')
        self.assertEqual(queued.configurations()[0].source_colour_profile, 'legacy')
        project = self.root / 'project.json'
        save_project(project, current)
        restored, _ = load_project(project)
        self.assertEqual(restored, current)
        self.window.request_preview()
        self.until_preview()
        settings = json.loads(self.window.config.toPlainText())
        settings['source_colour_profile'] = 'legacy'
        self.window.config.setPlainText(json.dumps(settings))
        self.assertIsNone(self.window.pixmap)
        self.assertEqual(self.window.preview_serial, -1)
        self.assertEqual(self.window.jobs, {})
        self.assertEqual(self.window.bridge.commands.qsize(), 0)

    def test_synchronized_source_uses_completed_profile_not_changed_editor(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        output = self.root / 'fg'
        output.mkdir()
        for index in range(2):
            write_rgb(output / f'{index}.exr', working709(self.rgb))
        job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr', frame_dir_linear=True, source_colour_profile='linear_bt2020_d65'))
        record = {'status': 'complete', 'input': job.input_path, 'diagnostics': {
            'source': {'start_frame': 1, 'num_frames': 2, 'native_hw': [6, 8], 'native_fps': 24.},
            'frames_submitted': 2, 'outputs': {'fg': str(output)}}}
        self.window.jobs['render'] = {'job': job, 'status': 'result', 'event': {'payload': {'record': record}}}
        self.window.refresh_table()
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Foreground')
        self.window.synchronize_reference()
        self.until_preview()
        self.window.view.setCurrentText('Source')
        self.until_preview()
        self.assertIsNotNone(self.window.reference)
        assert self.window.reference is not None and self.window.pixmap is not None
        expected = display_rgb(working709(self.rgb))[0, 0]
        for image in (self.window.reference.toImage(), self.window.pixmap.toImage()):
            pixel = image.pixelColor(0, 0)
            np.testing.assert_allclose([pixel.red(), pixel.green(), pixel.blue()], expected, atol=1)
        assert self.window.moving_reference is not None
        self.assertEqual(self.window.moving_reference.source_colour_profile, 'linear_bt2020_d65')
        self.assertEqual(json.loads(self.window.config.toPlainText())['source_colour_profile'], 'legacy')
        self.assertEqual(self.window.bridge.commands.qsize(), 0)
