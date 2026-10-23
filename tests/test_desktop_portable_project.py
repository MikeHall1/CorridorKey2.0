"""Real tiny portable projects through one Qt-owned cancellable file operation."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from corridorkey.asset_inventory import AssetSelection
from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.corrections import load_corrections
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


@unittest.skipUnless(importlib.util.find_spec('PySide6') is not None, 'Optional Qt binding unavailable')
class DesktopPortableProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['portable-project-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_portable_project import PortableProjectDialog
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        reference = np.full((24, 32, 3), (20, 150, 20), np.uint8)
        image = reference.copy()
        image[8:16, 10:22] = (20, 20, 220)
        for frame in range(4):
            cv2.imwrite(str(self.source / f'frame{frame}.png'), image)
        plate, samples = self.root / 'plate.png', self.root / 'samples.png'
        cv2.imwrite(str(plate), reference)
        mask = np.full((24, 32), 255, np.uint8)
        mask[6:18, 8:24] = 0
        cv2.imwrite(str(samples), mask)
        config = InferenceConfig(output_format='exr', start_frame=1, num_frames=2, frame_dir_fps=30.,
                                 low_vram=True, outputs={'alpha', 'fg'}, temp_dir=self.root / 'original-scratch')
        initial = JobRequest.create(self.source, self.root / 'original-output', config, profile_stages=True)
        editor = CorrectionEditor(initial, self.root)
        editor.prepare_plate(frame=0, reference_path=plate, reference_profile='srgb', samples_path=samples)
        editor.plate_view('guidance')
        editor.accept_plate()
        correction = editor.save()
        config.corrections_path = correction
        self.job = JobRequest.create(self.source, self.root / 'original-output', config, profile_stages=True)
        self.correction_identity = load_corrections(correction).identity
        self.models = self.root / 'models'
        self.models.mkdir()
        self.weights = self.models / 'local.pt'
        self.weights.write_bytes(b'local reference only; not a compatible model')
        self.selection = AssetSelection(self.models, self.weights)
        assets = load_corrections(correction).asset_paths()
        self.originals = {path: path.read_bytes() for path in [*self.source.iterdir(), correction, *assets, self.weights]}
        self.publication_parent = self.root / 'packages'
        self.publication_parent.mkdir()
        self.output, self.scratch = self.root / 'new-output', self.root / 'new-scratch'
        self.output.mkdir()
        self.scratch.mkdir()
        self.dialog = PortableProjectDialog(self.job, frame=1, selection=self.selection)
        self.loaded = []
        self.dialog.loaded.connect(lambda *values: self.loaded.append(values))
        self.dialog.show()

    def tearDown(self):
        self.dialog.shutdown()
        self.until(lambda: not self.dialog.busy)
        self.dialog.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.dialog)
        for path, raw in self.originals.items():
            if path.exists():
                self.assertEqual(path.read_bytes(), raw)
        self.temp.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.dialog.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail(self.dialog.status.text())

    def inventory(self):
        self.dialog.inventory_draft()
        self.until(lambda: not self.dialog.busy)
        self.assertIsNotNone(self.dialog.inventory, self.dialog.status.text())

    def verify(self):
        self.dialog.verify()
        self.until(lambda: not self.dialog.busy)

    def publish(self):
        self.dialog.destination.setText(str(self.publication_parent))
        self.dialog.publish()
        self.until(lambda: not self.dialog.busy)
        published = self.dialog.published_path
        assert published is not None, self.dialog.status.text()
        return published

    def inspected(self):
        inspection = self.dialog.inspection
        assert inspection is not None, self.dialog.status.text()
        return inspection

    def open_package(self, package):
        self.dialog.package_path.setText(str(package))
        self.dialog.inventory_package()
        self.until(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.mode, 'package', self.dialog.status.text())

    def dependency_row(self, role):
        assert self.dialog.report is not None
        return next(index for index, row in enumerate(self.dialog.report['dependencies']) if row['role'] == role)

    def test_metadata_preview_defaults_and_explicit_reference_only_apply_after_move(self):
        self.assertIsNone(self.dialog.inventory)
        self.assertEqual(self.dialog.copy_budget.value(), 0)
        self.assertFalse(self.dialog.publish_button.isEnabled())
        self.inventory()
        self.assertEqual(self.dialog.selected_collect_ids(), ())
        self.assertIn('NOT VERIFIED', self.dialog.models.text())
        self.assertIn(str(self.weights), self.dialog.models.text())
        self.assertIn('source start 1', self.dialog.details.toPlainText())
        self.verify()
        self.assertIsNotNone(self.dialog.plan, self.dialog.status.text())
        published = self.publish()
        assert published is not None
        self.assertEqual(list(published.rglob('*.png')), [])
        moved = self.publication_parent / 'moved-project'
        published.rename(moved)
        self.open_package(moved)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.verify()
        self.assertTrue(self.inspected().ready, self.dialog.status.text())
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertEqual(self.loaded, [])
        self.dialog.output.setText(str(self.output))
        self.dialog.scratch.setText(str(self.scratch))
        self.dialog.apply_button.click()
        self.until(lambda: not self.dialog.busy)
        self.assertEqual(len(self.loaded), 1)
        job, frame, output, model_report = self.loaded[0]
        config, _ = job.configurations()
        self.assertEqual((frame, config.start_frame, config.num_frames, config.frame_dir_fps), (1, 1, 2, 30.))
        self.assertEqual(config.outputs, {'alpha', 'fg'})
        self.assertTrue(config.low_vram)
        self.assertTrue(job.profile_stages)
        self.assertEqual(Path(output), self.output)
        self.assertEqual(Path(job.output_dir), self.output)
        self.assertEqual(Path(config.temp_dir), self.scratch)
        self.assertEqual(load_corrections(config.corrections_path).identity, self.correction_identity)
        self.assertEqual(json.loads(self.inspected().manifest_json)['original_job'], job_to_wire(self.job))
        assert self.dialog.report is not None
        self.assertEqual(model_report['requested_job_identity'], self.dialog.report['requested_job_identity'])
        self.assertNotIn('original_requested_job', model_report)
        self.assertNotIn('project_report', model_report)
        self.assertLessEqual(len(json.dumps(model_report).encode()), 64 * 1024)
        self.assertEqual(model_report['model_references']['weights'], str(self.weights))
        self.assertIn('NOT VERIFIED', model_report['compatibility'])

    def test_selected_collection_budget_fresh_publication_and_complete_sequence(self):
        from PySide6.QtCore import Qt
        self.inventory()
        for role in ('source', 'corrections'):
            item = self.dialog.table.item(self.dependency_row(role), 0)
            assert item is not None
            item.setCheckState(Qt.CheckState.Checked)
        self.verify()
        self.assertIsNone(self.dialog.plan)
        self.assertIn('budget', self.dialog.status.text().lower())
        self.dialog.copy_budget.setValue(4)
        self.verify()
        self.assertIsNotNone(self.dialog.plan, self.dialog.status.text())
        published = self.publish()
        self.open_package(published)
        self.verify()
        self.assertTrue(self.inspected().ready, self.dialog.status.text())
        self.dialog.output.setText(str(self.output))
        self.dialog.scratch.setText(str(self.scratch))
        self.dialog.apply()
        self.until(lambda: not self.dialog.busy)
        job = self.loaded[0][0]
        relocated = Path(job.input_path)
        self.assertTrue(relocated.is_relative_to(published))
        self.assertEqual(sorted(path.name for path in relocated.iterdir()), ['frame0.png', 'frame1.png', 'frame2.png', 'frame3.png'])
        config, _ = job.configurations()
        document = load_corrections(config.corrections_path, source=relocated)
        self.assertEqual(document.identity, self.correction_identity)
        self.assertTrue(all(path.is_relative_to(published) for path in document.asset_paths()))
        self.assertFalse(list(published.rglob('*.pt')))

    def test_missing_reference_requires_verified_relink_and_rejects_different_content(self):
        self.inventory()
        self.verify()
        published = self.publish()
        relocated = self.root / 'relocated-source'
        self.source.rename(relocated)
        wrong = self.root / 'wrong-source'
        wrong.mkdir()
        for original in relocated.iterdir():
            cv2.imwrite(str(wrong / original.name), np.zeros((24, 32, 3), np.uint8))
        self.open_package(published)
        self.verify()
        self.assertFalse(self.inspected().ready)
        row = self.dependency_row('source')
        self.dialog.table.selectRow(row)
        self.dialog.relink_path.setText(str(wrong))
        self.dialog.stage_relink()
        self.verify()
        self.assertFalse(self.inspected().ready)
        self.assertEqual(self.loaded, [])
        self.dialog.table.selectRow(self.dependency_row('source'))
        self.dialog.relink_path.setText(str(relocated))
        self.dialog.stage_relink()
        self.verify()
        self.assertTrue(self.inspected().ready, self.dialog.status.text())
        self.dialog.package_path.setText(str(self.root / 'another-package'))
        self.assertIsNone(self.dialog.inspection)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        relocated.rename(self.source)

    def test_cancellation_and_close_keep_owner_and_drop_late_results(self):
        from corridorkey import portable_project
        original = portable_project.inventory_project
        entered, release = threading.Event(), threading.Event()

        def blocked(*args, check, **kwargs):
            entered.set()
            while not release.wait(.002):
                check()
            return original(*args, check=check, **kwargs)

        with patch.object(portable_project, 'inventory_project', side_effect=blocked):
            self.dialog.inventory_draft()
            self.until(entered.is_set)
            self.dialog.cancel()
            self.until(lambda: not self.dialog.busy)
        self.assertIn('cancelled', self.dialog.status.text().lower())
        self.assertIsNone(self.dialog.inventory)
        self.inventory()
        self.assertEqual(self.loaded, [])
        entered.clear()

        def late(*args, check, **kwargs):
            entered.set()
            release.wait(3)
            return original(*args, check=lambda: None, **kwargs)

        self.dialog.inventory = None
        with patch.object(portable_project, 'inventory_project', side_effect=late):
            self.dialog.inventory_draft()
            self.until(entered.is_set)
            self.assertFalse(self.dialog.shutdown())
            self.assertTrue(self.dialog.closing)
            self.assertFalse(self.dialog.inventory_button.isEnabled())
            release.set()
            self.until(lambda: not self.dialog.busy)
        self.assertIsNone(self.dialog.inventory)
        self.assertEqual(self.loaded, [])
        self.assertFalse(self.dialog.timer.isActive())

    def test_main_rejection_ack_is_truthful_and_retains_emitted_correction_snapshot(self):
        self.inventory()
        self.verify()
        published = self.publish()
        self.open_package(published)
        self.verify()
        self.assertTrue(self.inspected().ready)
        self.dialog.output.setText(str(self.output))
        self.dialog.scratch.setText(str(self.scratch))
        message = 'Draft unchanged: correction draft became dirty during file verification.'
        self.dialog.loaded.connect(lambda *_: self.dialog.acknowledge_application(False, message))
        self.dialog.apply()
        self.until(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.status.text(), message)
        assert self.dialog.application_report is not None
        self.assertFalse(self.dialog.application_report['accepted'])
        retained = self.dialog.last_emitted_job
        assert retained is not None
        config, _ = retained.configurations()
        assert config.corrections_path is not None
        self.assertTrue(Path(config.corrections_path).is_file())
        self.assertIn(str(config.corrections_path), self.dialog.details.toPlainText())
        self.dialog.acknowledge_application(True, 'Obsolete acknowledgement')
        self.assertEqual(self.dialog.status.text(), message)

    def test_unsupported_large_preview_frames_preserve_values_and_refuse_before_materialization(self):
        from corridorkey.desktop_state import DESKTOP_MAX_PREVIEW_FRAME
        from corridorkey.portable_project import ProjectInspection
        for frame in (DESKTOP_MAX_PREVIEW_FRAME + 1, 2**40):
            with self.subTest(frame=frame):
                self.dialog.frame = frame
                self.inventory()
                self.verify()
                published = self.publish()
                self.open_package(published)
                self.verify()
                self.assertTrue(self.inspected().ready)
                self.dialog.output.setText(str(self.output))
                self.dialog.scratch.setText(str(self.scratch))
                self.assertFalse(self.dialog.apply_button.isEnabled())
                self.assertIn(str(frame), self.dialog.status.text())
                self.assertIn('unsupported', self.dialog.status.text())
                before = set(self.scratch.iterdir())
                with patch.object(ProjectInspection, 'apply', side_effect=AssertionError('Unsupported frame must not materialize a correction snapshot')) as materialize:
                    self.dialog.apply()
                    self.app.processEvents()
                    self.dialog.poll()
                    materialize.assert_not_called()
                self.assertEqual(set(self.scratch.iterdir()), before)
                self.assertEqual(self.loaded, [])
                assert self.dialog.report is not None
                self.assertEqual(self.dialog.report['preview_frame'], frame)
        # The signal boundary itself carries Python integers exactly; Apply above
        # refuses these unsupported values instead of emitting or clamping them.
        exact = []
        self.dialog.loaded.connect(lambda _job, frame, _output, _context: exact.append(frame))
        self.dialog.loaded.emit(None, 2**40, '', {})
        self.assertEqual(exact, [2**40])

    def test_cancel_and_close_after_actual_commit_retain_snapshot_without_late_emission(self):
        from corridorkey.portable_project import ProjectInspection
        self.inventory()
        self.verify()
        published = self.publish()
        self.open_package(published)
        self.verify()
        self.dialog.output.setText(str(self.output))
        self.dialog.scratch.setText(str(self.scratch))
        real_apply = ProjectInspection.apply
        paths = []
        for closing in (False, True):
            with self.subTest(closing=closing):
                if closing:
                    self.verify()  # Cancellation invalidates the previous UI verification.
                committed, release = threading.Event(), threading.Event()

                def commit_then_hold(inspection, *, output_dir, temp_dir, check, committed=committed, release=release):
                    job = real_apply(inspection, output_dir=output_dir, temp_dir=temp_dir, check=check)
                    committed.set()
                    if not release.wait(3):
                        raise RuntimeError('Fixture delivery release deadline')
                    return job

                with patch.object(ProjectInspection, 'apply', commit_then_hold):
                    self.dialog.apply()
                    self.until(committed.is_set)
                    if closing:
                        self.assertFalse(self.dialog.shutdown())
                    else:
                        self.dialog.cancel()
                    self.assertEqual(self.loaded, [])
                    release.set()
                    self.until(lambda: not self.dialog.busy)
                retained = self.dialog.last_resolved_job
                assert retained is not None
                self.assertIs(self.dialog.last_resolved, retained)
                self.assertIs(self.dialog.last_emitted_job, retained)
                config, _ = retained.configurations()
                assert config.corrections_path is not None
                artifact = Path(config.corrections_path)
                self.assertTrue(artifact.is_file())
                self.assertTrue(artifact.is_relative_to(self.scratch))
                self.assertEqual(load_corrections(artifact, source=retained.input_path).identity, self.correction_identity)
                paths.append(artifact)
                self.assertIn(str(artifact), self.dialog.details.toPlainText())
                self.assertIn(str(artifact), self.dialog.status.text())
                assert self.dialog.application_report is not None
                self.assertFalse(self.dialog.application_report['emitted'])
                self.assertFalse(self.dialog.application_report['accepted'])
                self.assertEqual(self.dialog.application_report['resolved_correction_snapshot'], str(artifact))
                self.assertLessEqual(len(json.dumps(self.dialog.application_report).encode()), 64 * 1024)
                self.assertEqual(self.loaded, [])
                self.assertEqual(self.dialog.original_wire, job_to_wire(self.job))
        self.assertEqual(len(set(paths)), 2)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_frozen_pagination_exposes_last_relink_and_preserves_global_collection_choices(self):
        from PySide6.QtCore import Qt

        from corridorkey import portable_project
        from corridorkey.desktop_portable_project import (
            _json_bytes,
            compact_project_report,
        )
        self.inventory()
        self.verify()
        verified_plan = self.dialog.plan
        rows = [{'id': 'source', 'role': 'source', 'path': str(self.source), 'kind': 'sequence',
                 'status': 'verified', 'file_count': 4, 'files': [], 'bytes': 100, 'required': True}]
        rows.extend({'id': f'mask-{index}', 'role': 'correction_asset', 'path': str(self.root / f'mask-{index}.png'),
                     'kind': 'file', 'status': 'missing', 'file_count': 1, 'files': [], 'bytes': 1,
                     'required': True} for index in range(300))
        report = {'dependencies': rows, 'requested_job': job_to_wire(self.job), 'preview_frame': 1,
                  'model_references': {}, 'ready': True, 'read_bytes': 100, 'copy_bytes': 0}
        self.dialog.full_report = _json_bytes(report)
        self.dialog.collect_ids = {'source'}
        self.dialog._show_report(compact_project_report(report))
        self.dialog.inventory = self.dialog.report
        last_page = self.dialog.page_count - 1
        with patch.object(portable_project, 'inventory_project', side_effect=AssertionError('No rescan for pagination')), \
                patch.object(portable_project, 'inspect_project', side_effect=AssertionError('No rehash for pagination')), \
                patch.object(portable_project, 'plan_project', side_effect=AssertionError('No replanning for pagination')):
            self.dialog.request_page(last_page)
            self.until(lambda: not self.dialog.busy)
            self.assertIs(self.dialog.plan, verified_plan)
            last_row = self.dialog.table.rowCount() - 1
            last = self.dialog.table.item(last_row, 0)
            assert last is not None
            self.assertEqual(last.data(Qt.ItemDataRole.UserRole), 'mask-299')
            last.setCheckState(Qt.CheckState.Checked)
            self.assertEqual(self.dialog.selected_collect_ids(), ('mask-299', 'source'))
            self.dialog.request_page(0)
            self.until(lambda: not self.dialog.busy)
            first = self.dialog.table.item(0, 0)
            assert first is not None
            self.assertEqual(first.checkState(), Qt.CheckState.Checked)
            self.assertEqual(self.dialog.selected_collect_ids(), ('mask-299', 'source'))
            self.dialog.mode = 'package'
            self.dialog.request_page(last_page)
            self.until(lambda: not self.dialog.busy)
            self.dialog.table.selectRow(self.dialog.table.rowCount() - 1)
            replacement = self.root / 'deliberate-replacement.png'
            self.dialog.relink_path.setText(str(replacement))
            self.dialog.stage_relink()
            self.assertEqual(self.dialog.relinks['mask-299'], replacement)
            self.dialog.request_page(0)
            self.until(lambda: not self.dialog.busy)
            self.dialog.request_page(last_page)
            self.until(lambda: not self.dialog.busy)
            staged = self.dialog.table.item(self.dialog.table.rowCount() - 1, 3)
            assert staged is not None
            self.assertIn('Staged relink', staged.text())
            self.assertEqual(self.loaded, [])


@unittest.skipUnless(importlib.util.find_spec('PySide6') is not None, 'Optional Qt binding unavailable')
class PortableReportBoundsTests(unittest.TestCase):
    def test_100000_file_inventory_is_counted_without_traversal_or_large_gui_payload(self):
        from corridorkey.desktop_portable_project import (
            _loaded_context,
            compact_project_report,
        )

        class MetadataOnlyFiles(list):
            def __iter__(self):
                raise AssertionError('GUI formatter must not traverse or stringify file inventories')

        files = MetadataOnlyFiles([{'relative_path': 'frame.png', 'bytes': 1}] * 100000)
        rows = [{'id': 'source', 'role': 'source', 'path': '/source', 'kind': 'sequence', 'status': 'available',
                 'files': files, 'bytes': 100000, 'required': True, 'depends_on': []}]
        rows.extend({'id': f'asset-{index}', 'role': 'correction_asset', 'path': '/' + 'long-path/' * 400,
                     'status': 'available; content not verified', 'files': [], 'file_count': 0,
                     'bytes': 1, 'required': True, 'depends_on': []} for index in range(1000))
        report = {'dependencies': rows, 'model_references': {'asset_root': '/models', 'weights': '/models/model.pt', 'entries': []},
                  'requested_job': None, 'ready': False, 'read_bytes': 0, 'copy_bytes': 0, 'preview_frame': 0}
        compact = compact_project_report(report)
        self.assertLessEqual(len(json.dumps(compact).encode()), 64 * 1024)
        self.assertEqual(compact['dependency_count'], 1001)
        self.assertEqual(compact['total_file_count'], 100000)
        self.assertGreater(compact['omitted_dependencies'], 0)
        self.assertLessEqual(len(compact['dependencies']), 128)
        self.assertTrue(all('files' not in row for row in compact['dependencies']))
        self.assertEqual(len(files), 100000)
        context, frame = _loaded_context(SimpleNamespace(report=report, manifest_json=b'{}', package_path='/package'), Path('/output'), Path('/scratch'))
        self.assertLessEqual(len(json.dumps(context).encode()), 64 * 1024)
        self.assertEqual(frame, 0)
        self.assertEqual(context['file_count'], 100000)
        self.assertNotIn('dependencies', context)
        self.assertNotIn('requested_job', context)
