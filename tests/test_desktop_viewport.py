"""Actual offscreen A/B geometry and immutable decoded-preview pixels."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class ViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['viewport-test', '-platform', 'offscreen'])

    def setUp(self):
        from PySide6.QtGui import QColor, QPixmap

        from corridorkey.desktop_viewport import PreviewViewport, ViewportState
        self.state = ViewportState()
        self.a, self.b = PreviewViewport(self.state), PreviewViewport(self.state)
        self.a.resize(400, 200)
        self.b.resize(200, 300)
        self.original = QPixmap(100, 50)
        self.original.fill(QColor(35, 90, 120))
        self.a.setPixmap(self.original)
        self.b.setPixmap(self.original)
        self.a.show()
        self.b.show()
        self.app.processEvents()

    def tearDown(self):
        dispose_widget(self.a)
        dispose_widget(self.b)

    def test_fit_uses_common_scale_and_normalized_center(self):
        from PySide6.QtCore import QPointF
        self.state.fit()
        self.assertEqual(self.state.zoom, 2)
        self.assertEqual(self.a.image_rect().width(), 200)
        self.assertEqual(self.b.image_rect().width(), 200)
        self.assertEqual(self.a.source_at(QPointF(200, 100)), QPointF(.5, .5))
        self.assertEqual(self.b.source_at(QPointF(100, 150)), QPointF(.5, .5))
        self.assertIn('decoded preview', self.state.label)

    def test_matched_pan_and_zoom_survive_resize_and_frame_replacement(self):
        from PySide6.QtCore import QPointF
        self.state.set_transform(3, QPointF(.3, .6))
        for view in (self.a, self.b):
            center = QPointF(view.width() / 2, view.height() / 2)
            self.assertEqual(view.source_at(center), QPointF(.3, .6))
            self.assertAlmostEqual(view.source_at(center + QPointF(30, 0)).x(), .4)
        self.b.resize(600, 450)
        self.b.setPixmap(self.original)
        self.app.processEvents()
        self.assertEqual(self.state.zoom, 3)
        self.assertEqual(self.state.center, QPointF(.3, .6))
        self.state.zoom_by(2)
        self.assertEqual(self.a.image_rect().width(), self.b.image_rect().width())

    def test_wheel_keeps_point_under_cursor_and_drag_updates_both(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QMouseEvent, QWheelEvent
        self.state.set_transform(2)
        position = QPointF(220, 110)
        before = self.a.source_at(position)
        wheel = QWheelEvent(position, position, QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        self.a.wheelEvent(wheel)
        after = self.a.source_at(position)
        self.assertAlmostEqual(before.x(), after.x())
        self.assertAlmostEqual(before.y(), after.y())
        old = QPointF(self.state.center)
        press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(200, 100), QPointF(200, 100),
                            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        move = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(210, 110), QPointF(210, 110),
                           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        release = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(210, 110), QPointF(210, 110),
                              Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        self.a.mousePressEvent(press)
        self.a.mouseMoveEvent(move)
        self.a.mouseReleaseEvent(release)
        self.assertLess(self.state.center.x(), old.x())
        self.assertEqual(self.a.state.center, self.b.state.center)
        self.assertIsNone(self.a._drag)

    def test_destroyed_peer_is_removed_before_fit_queries(self):
        from corridorkey.desktop_viewport import PreviewViewport
        peer = PreviewViewport(self.state)
        peer.setPixmap(self.original)
        dispose_widget(peer)
        self.state.fit()
        self.assertNotIn(peer, self.state._views)

    def test_zoom_pan_bounds_and_nonfinite_rejection(self):
        from PySide6.QtCore import QPointF
        self.state.set_transform(1e9, QPointF(-100, 100))
        self.assertEqual(self.state.zoom, 16)
        self.assertEqual(self.state.center, QPointF(0, 1))
        self.state.set_transform(1e-9)
        self.assertEqual(self.state.zoom, .01)
        for zoom in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(zoom=zoom), self.assertRaises(ValueError):
                self.state.set_transform(zoom)
        with self.assertRaises(ValueError):
            self.state.set_transform(1, QPointF(float('nan'), 0))

    def test_pixel_bytes_png_source_and_pinned_copy_unchanged(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPixmap
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.png'
            self.assertTrue(self.original.save(str(source)))
            before = source.read_bytes()
            image = QPixmap(str(source))
            self.a.setPixmap(image)
            self.b.setPixmap(self.a.pixmap())
            original = self.a.pixmap().toImage()
            self.state.set_transform(12, QPointF(.1, .9))
            self.a.grab()  # Exercise actual QPainter rendering offscreen.
            self.b.grab()
            self.assertEqual(self.a.pixmap().toImage(), original)
            self.assertEqual(self.b.pixmap().toImage(), original)
            self.assertEqual(source.read_bytes(), before)
            self.a.setText('Decoding…')
            self.assertTrue(self.a.pixmap().isNull())
            self.assertEqual(self.a.text(), 'Decoding…')
            self.assertEqual(self.b.pixmap().toImage(), original)


if __name__ == '__main__':
    unittest.main()
