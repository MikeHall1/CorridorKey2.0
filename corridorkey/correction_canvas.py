"""Binary brush in native image coordinates with explicit letterbox mapping."""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget


def image_rect(width: int, height: int, native_width: int, native_height: int) -> tuple[float, float, float, float]:
    if min(width, height, native_width, native_height) <= 0:
        raise ValueError('Canvas/image dimensions must be positive.')
    scale = min(width / native_width, height / native_height)
    w, h = native_width * scale, native_height * scale
    return (width - w) / 2, (height - h) / 2, w, h


def native_point(x: float, y: float, rect: tuple[float, float, float, float],
                 native_width: int, native_height: int) -> tuple[int, int] | None:
    left, top, width, height = rect
    if not left <= x < left + width or not top <= y < top + height:
        return None
    return min(native_width - 1, int((x - left) * native_width / width)), min(native_height - 1, int((y - top) * native_height / height))


class CorrectionCanvas(QWidget):
    native_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self.draft: NDArray[np.uint8] | None = None
        self.source: QPixmap | None = None
        self.saved_overlay: NDArray[np.uint8] | None = None
        self.overlay_kind = 'coverage'
        self.native_width = self.native_height = 0
        self.radius, self.value = 12, 255
        self.previous: tuple[int, int] | None = None
        self.painting = True
        self.click_mode = False
        self.click_markers: list[tuple[int, int, bool]] = []
        self.dirty = False

    def set_frame(self, pixels: NDArray[np.uint8], width: int, height: int, overlay: NDArray[np.uint8] | None, *, overlay_kind: str = 'coverage'):
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.source = QPixmap.fromImage(image)
        self.native_width, self.native_height = width, height
        self.draft = np.zeros((height, width), np.uint8)
        if overlay_kind not in {'coverage', 'trimap'}:
            raise ValueError('Unsupported authored overlay kind.')
        self.saved_overlay = overlay
        self.overlay_kind = overlay_kind
        self.previous = None
        self.dirty = False
        self.update()

    def clear_draft(self):
        if self.draft is not None:
            self.draft.fill(0)
            self.dirty = False
            self.update()

    def point(self, position: QPointF) -> tuple[int, int] | None:
        if self.source is None:
            return None
        return native_point(position.x(), position.y(), image_rect(self.width(), self.height(), self.native_width, self.native_height),
                            self.native_width, self.native_height)

    def _stroke(self, point):
        if self.draft is None:
            return
        if self.previous is not None:
            cv2.line(self.draft, self.previous, point, (float(self.value),), max(1, self.radius * 2), lineType=cv2.LINE_8)
        cv2.circle(self.draft, point, self.radius, (float(self.value),), thickness=-1, lineType=cv2.LINE_8)
        self.previous = point
        self.dirty = True
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.click_mode and event.button() == Qt.MouseButton.LeftButton:
            point = self.point(event.position())
            if point is not None:
                self.native_clicked.emit(*point)
            return
        if self.painting and event.button() == Qt.MouseButton.LeftButton:
            point = self.point(event.position())
            if point is not None:
                self._stroke(point)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.painting and not self.click_mode and event.buttons() & Qt.MouseButton.LeftButton:
            point = self.point(event.position())
            if point is None:
                self.previous = None  # Do not bridge strokes across letterboxing.
            else:
                self._stroke(point)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.previous = None

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.source is None or self.draft is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Bind a source, then Load frame to paint native regions.')
            return
        left, top, width, height = image_rect(self.width(), self.height(), self.native_width, self.native_height)
        target = QRectF(left, top, width, height)
        painter.drawPixmap(target, self.source, QRectF(self.source.rect()))
        size = (max(1, round(width)), max(1, round(height)))
        for mask, colour in [(self.saved_overlay, (0, 255, 80)), (self.draft, (40, 150, 255))]:
            if mask is None:
                continue
            coverage = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            rgba = np.zeros((*coverage.shape, 4), np.uint8)
            rgba[:, :, :3] = colour
            if mask is self.saved_overlay and self.overlay_kind == 'trimap':
                rgba[coverage == 0] = (255, 70, 70, 110)
                rgba[coverage == 255] = (0, 255, 80, 110)
                rgba[coverage == 128] = (0, 0, 0, 0)
            else:
                rgba[:, :, 3] = (coverage.astype(np.uint16) * 100 // 255).astype(np.uint8)
            image = QImage(rgba.data, size[0], size[1], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
            painter.drawImage(target, image)

        for index, (x, y, positive) in enumerate(self.click_markers):
            center = QPointF(left + (x + .5) * width / self.native_width,
                             top + (y + .5) * height / self.native_height)
            painter.setPen(QColor('white'))
            painter.setBrush(QColor(0, 210, 80) if positive else QColor(245, 85, 45))
            painter.drawEllipse(center, 4., 4.)
            painter.drawText(center + QPointF(6, -5), str(index + 1))
