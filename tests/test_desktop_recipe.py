"""Recipe preparation creates a frozen batch; execution stays explicit."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.batch import load_jobs
from corridorkey.config import InferenceConfig
from corridorkey.desktop_recipe import prepare_recipe
from corridorkey.session import JobRequest
from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.job = JobRequest.create(self.source, self.root / 'outputs', InferenceConfig(output_format='exr'))

    def tearDown(self):
        self.folder.cleanup()

    def test_fresh_recipe_roundtrips_without_touching_source_or_creating_outputs(self):
        hint = self.source / 'hint.png'
        hint.write_bytes(b'untouched')
        job = JobRequest.create(self.source, self.root / 'outputs', InferenceConfig(output_format='exr', seed=21),
                                initial_hint_path=hint, profile_stages=True)
        first = prepare_recipe(self.root, job)
        before = first.read_bytes()
        second = prepare_recipe(self.root, self.job)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)
        jobs, weights = load_jobs(first)
        self.assertIsNone(weights)
        self.assertEqual(jobs[0].configurations()[0].seed, 21)
        self.assertTrue(jobs[0].profile_stages)
        self.assertEqual(jobs[0].initial_hint_path, str(hint))
        self.assertEqual(Path(jobs[0].output_dir).parent, self.root / 'outputs')
        self.assertFalse((self.root / 'outputs').exists())
        self.assertEqual(hint.read_bytes(), b'untouched')

    def test_rejects_unsupported_profile_and_protected_location_before_writing(self):
        video = JobRequest.create(self.source, self.root / 'outputs')
        with self.assertRaisesRegex(ValueError, 'EXR'):
            prepare_recipe(self.root, video)
        with self.assertRaisesRegex(ValueError, 'separate'):
            prepare_recipe(self.source, self.job)
        self.assertEqual(list(self.source.iterdir()), [])
        self.assertFalse(list(self.root.glob('corridorkey-recipe-*')))

    def test_oversized_saved_recipe_rejected_before_owned_directory_creation(self):
        with patch('corridorkey.desktop_recipe.MAX_FRAME_BYTES', 2500), self.assertRaisesRegex(ValueError, 'size limit'):
            prepare_recipe(self.root, self.job)
        self.assertFalse(list(self.root.glob('corridorkey-recipe-*')))

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
    def test_window_prepares_current_settings_without_dispatch_and_panel_blocks_racing_run(self):
        from PySide6.QtWidgets import QApplication

        from corridorkey.desktop_window import DesktopWindow
        app = QApplication.instance() or QApplication(['recipe-test', '-platform', 'offscreen'])
        window = DesktopWindow()
        try:
            window.populate_job(self.job)
            with patch('corridorkey.desktop_window.QFileDialog.getExistingDirectory', return_value=str(self.root)):
                window.prepare_recovery()
            assert window.recovery.io_thread is not None
            window.recovery.io_thread.join(10)
            window.recovery.workspace.setText(str(self.root / 'cache'))
            window.recovery.request(False)
            self.assertIn('Wait for recovery I/O', window.recovery.status.text())
            window.recovery.poll()
            path = Path(window.recovery.batch.text())
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())['jobs'][0]['settings']['output_format'], 'exr')
            self.assertFalse(window.jobs)
            self.assertTrue(window.bridge.commands.empty())
        finally:
            window.close()
            window.bridge.thread.join(10)
            window.poll()
            app.processEvents()
            dispose_widget(window)
