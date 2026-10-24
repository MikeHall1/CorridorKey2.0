"""Matched A/B viewing of thumbnail or explicitly admitted native image pixels."""
from __future__ import annotations

import math
import weakref

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QWidget


class ViewportState(QObject):
    """Shared decoded-pixel scale and normalized source centre for both views."""
    changed = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.zoom = 1.0
        self.native_pixels = False
        self.center = QPointF(.5, .5)
        self.fitting = True
        self._views: weakref.WeakSet[PreviewViewport] = weakref.WeakSet()

    @property
    def label(self) -> str:
        unit = 'native image pixels' if self.native_pixels else 'decoded preview'
        exact = ' · 1:1 image/logical canvas pixels' if self.native_pixels and self.zoom == 1 else ''
        return f'{self.zoom * 100:.1f}% of {unit}' + exact + (' · Fit' if self.fitting else '')

    def fit(self) -> None:
        self.fitting = True
        self.center = QPointF(.5, .5)
        self.refresh_fit()

    def refresh_fit(self) -> None:
        if self.fitting:
            scales = []
            for view in list(self._views):
                try:
                    if not view._pixmap.isNull() and view.width() > 0 and view.height() > 0:
                        scales.append(min(view.width() / view._pixmap.width(), view.height() / view._pixmap.height()))
                except RuntimeError:
                    # Qt can destroy a child before its Python wrapper is released.
                    self._views.discard(view)
            if scales:
                self.zoom = min(16.0, max(.01, min(scales)))
        self.changed.emit()

    def set_transform(self, zoom: float, center: QPointF | None = None) -> None:
        if type(zoom) not in (int, float) or not math.isfinite(zoom) or zoom <= 0:
            raise ValueError('Viewport zoom must be finite and positive.')
        selected = self.center if center is None else center
        if not math.isfinite(selected.x()) or not math.isfinite(selected.y()):
            raise ValueError('Viewport centre must be finite.')
        self.zoom = min(16.0, max(.01, zoom))
        self.center = QPointF(min(1.0, max(0.0, selected.x())), min(1.0, max(0.0, selected.y())))
        self.fitting = False
        self.changed.emit()

    def zoom_by(self, factor: float) -> None:
        self.set_transform(self.zoom * factor)


class PreviewViewport(QWidget):
    """QLabel-compatible image/text surface with shared wheel zoom and drag pan.

    Mapping uses decoded dimensions: a legacy thumbnail capped at 1280, or
    explicitly admitted native image pixels. A/B share centre and scale even at different widget
    sizes; neither interaction modifies the pixmap or render settings.
    """
    def __init__(self, state: ViewportState, text: str = '', parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._pixmap = QPixmap()
        self._text = text
        self._drag: QPointF | None = None
        self.state._views.add(self)
        self.state.changed.connect(self.update)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap.copy()
        self._pixmap.setDevicePixelRatio(1.0)
        self._text = ''
        self.state.refresh_fit()

    def pixmap(self) -> QPixmap:
        return self._pixmap.copy()

    def setText(self, text: str) -> None:
        self._text = text
        self._pixmap = QPixmap()
        self.state.refresh_fit()

    def text(self) -> str:
        return self._text

    def source_at(self, position: QPointF) -> QPointF:
        """Normalized decoded source coordinate, including outside-image points."""
        if self._pixmap.isNull():
            return QPointF(self.state.center)
        return QPointF(self.state.center.x() + (position.x() - self.width() / 2) / (self.state.zoom * self._pixmap.width()),
                       self.state.center.y() + (position.y() - self.height() / 2) / (self.state.zoom * self._pixmap.height()))

    def image_rect(self) -> QRectF:
        width, height = self._pixmap.width() * self.state.zoom, self._pixmap.height() * self.state.zoom
        return QRectF(self.width() / 2 - self.state.center.x() * width,
                      self.height() / 2 - self.state.center.y() * height, width, height)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), '#15191d')
        painter.setPen('#cbd2d9')
        if self._pixmap.isNull():
            painter.drawText(self.rect().adjusted(8, 8, -8, -8),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._text)
        else:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not self.state.native_pixels)
            painter.drawPixmap(self.image_rect(), self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen('#45515c')
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.state.refresh_fit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap.isNull() or not event.angleDelta().y():
            event.ignore()
            return
        before = self.source_at(event.position())
        factor = 1.25 if event.angleDelta().y() > 0 else .8
        zoom = min(16.0, max(.01, self.state.zoom * factor))
        center = QPointF(before.x() - (event.position().x() - self.width() / 2) / (zoom * self._pixmap.width()),
                         before.y() - (event.position().y() - self.height() / 2) / (zoom * self._pixmap.height()))
        self.state.set_transform(zoom, center)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._pixmap.isNull():
            self._drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None or self._pixmap.isNull():
            event.ignore()
            return
        delta = event.position() - self._drag
        self._drag = event.position()
        self.state.set_transform(self.state.zoom, QPointF(
            self.state.center.x() - delta.x() / (self.state.zoom * self._pixmap.width()),
            self.state.center.y() - delta.y() / (self.state.zoom * self._pixmap.height())))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            event.ignore()
