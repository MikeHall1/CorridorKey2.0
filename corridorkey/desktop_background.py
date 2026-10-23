"""Explicit photo-preview controls; the workspace owns preparation and pixels."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class BackgroundDialog(QDialog):
    prepare_requested = Signal(str, str)
    changed = Signal()
    cancel_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Photo composite background')
        self.resize(660, 340)
        self.waiting = False
        root = QVBoxLayout(self)
        intro = QLabel('Choose a local RGB photo and declare its interpretation. Prepare explicitly reads and decodes it on the existing preview worker. Native dimensions must exactly match the selected completed foreground/alpha exports; no crop, stretch or automatic colour fallback. The captured snapshot stays fixed until prepared again. Render settings and exports never change.')
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QFormLayout()
        self.path = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.path)
        choose = QPushButton('Choose…')
        choose.clicked.connect(self.choose)
        row.addWidget(choose)
        form.addRow('Photo', row)
        self.profile = QComboBox()
        self.profile.addItem('Assume sRGB · PNG/JPEG RGB (embedded profiles/orientation not interpreted)', 'assumed_srgb')
        self.profile.addItem('Scene-linear BT.709 D65 · RGB EXR', 'linear_bt709_d65')
        form.addRow('Interpretation', self.profile)
        root.addLayout(form)
        limits = QLabel('Initial bounds: 32 MiB encoded, 4,194,304 pixels, declared 512 MiB work estimate. RGB only; no alpha, ICC/OCIO transform or EXIF orientation automation. Estimates are admission bounds, not a hard process RAM limit.')
        limits.setWordWrap(True)
        root.addWidget(limits)
        buttons = QHBoxLayout()
        self.prepare = QPushButton('Prepare selected photo')
        self.prepare.clicked.connect(self.request_prepare)
        buttons.addWidget(self.prepare)
        cancel = QPushButton('Cancel preparation')
        cancel.clicked.connect(lambda: self.cancel_requested.emit())
        buttons.addWidget(cancel)
        clear = QPushButton('Use checkerboard')
        clear.clicked.connect(lambda: self.clear_requested.emit())
        buttons.addWidget(clear)
        root.addLayout(buttons)
        self.status = QLabel('No photo prepared. Opening this dialog performs no image I/O.')
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.path.textChanged.connect(self.inputs_changed)
        self.profile.currentIndexChanged.connect(self.inputs_changed)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose local RGB composite photo', '', 'RGB images (*.png *.jpg *.jpeg *.exr)')
        if path:
            self.path.setText(path)

    def inputs_changed(self, *_):
        self.changed.emit()
        self.status.setText('Photo selection changed; prepare explicitly before using the new snapshot.')

    def request_prepare(self):
        if self.waiting:
            return
        path = self.path.text().strip()
        if not path:
            self.status.setText('Choose a local RGB photo first.')
            return
        self.prepare_requested.emit(path, self.profile.currentData())

    def set_waiting(self, waiting: bool, message: str):
        self.waiting = waiting
        self.prepare.setEnabled(not waiting)
        self.status.setText(message[:4096])

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.waiting:
            self.cancel_requested.emit()
        event.accept()
