"""Existing editor purpose selector; immutable saved semantics."""
import unittest
from typing import TYPE_CHECKING

from tests import test_desktop_corrections as fixtures

if TYPE_CHECKING:
    from corridorkey.desktop_corrections import CorrectionDialog


@unittest.skipUnless(fixtures.QT_AVAILABLE, 'Optional Qt binding unavailable')
class RegionPurposePanelTests(unittest.TestCase):
    editor: 'CorrectionDialog'

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['region-purpose-test', '-platform', 'offscreen'])

    setUp = fixtures.DesktopCorrectionTests.setUp
    tearDown = fixtures.DesktopCorrectionTests.tearDown
    until = fixtures.DesktopCorrectionTests.until
    paint_and_commit = fixtures.DesktopCorrectionTests.paint_and_commit
    save = fixtures.DesktopCorrectionTests.save
    def test_distinct_region_purpose_brush_inspection_and_history(self):
        editor = self.editor
        editor.kind.setCurrentIndex(editor.kind.findData('region'))
        for purpose, target in [('composite','despill'),('diagnostic','tracking_overlay')]:
            editor.new_mask()
            self.until(lambda: editor.pending is None)
            editor.region_purpose.setCurrentIndex(editor.region_purpose.findData(purpose))
            self.assertTrue(editor.region_purpose.isEnabled())
            self.assertEqual(editor.parameters()['purpose'],purpose)
            self.paint_and_commit('Refinement region — expansion/feather only')
            entry = editor.entries[-1]
            self.assertEqual((entry['purpose'],entry['target']),(purpose,target))
            self.assertFalse(editor.region_purpose.isEnabled())
            item = editor.table.item(len(editor.entries)-1,0)
            assert item is not None
            self.assertIn(purpose,item.text())
            editor.edit()
            self.until(lambda: editor.pending is None)
            self.assertEqual(editor.entries[-1]['purpose'],purpose)
        self.assertIn('captured',editor.region_note.text())
        self.save()
