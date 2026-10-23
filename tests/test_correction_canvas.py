import importlib.util
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class CorrectionCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['canvas-test', '-platform', 'offscreen'])

    def test_native_mapping_and_real_mouse_strokes_respect_letterboxing(self):
        import numpy as np
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        from corridorkey.correction_canvas import (
            CorrectionCanvas,
            image_rect,
            native_point,
        )
        self.assertEqual(image_rect(400, 400, 8, 6), (0, 50, 400, 300))
        self.assertIsNone(native_point(10, 10, (0, 50, 400, 300), 8, 6))
        self.assertEqual(native_point(399, 349, (0, 50, 400, 300), 8, 6), (7, 5))
        canvas = CorrectionCanvas()
        try:
            canvas.resize(400, 400)
            canvas.set_frame(np.zeros((6, 8, 3), np.uint8), 8, 6, None)
            canvas.radius = 1
            canvas.show()
            self.app.processEvents()
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            assert canvas.draft is not None
            self.assertFalse(canvas.draft.any())
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(225, 225))
            self.assertEqual(canvas.draft[3, 4], 255)
            self.assertEqual(canvas.draft[0, 0], 0)
            canvas.value = 0
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(225, 225))
            self.assertEqual(canvas.draft[3, 4], 0)
            self.assertFalse(canvas.grab().isNull())
        finally:
            from tests.qt_cleanup import dispose_widget
            dispose_widget(canvas)
