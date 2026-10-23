"""Explicit common settings editor; advanced JSON remains the complete contract."""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .desktop_state import pretty_settings
from .session import JobRequest
from .worker_protocol import job_from_wire, job_to_wire

# Text numeric controls avoid rounding or clamping values loaded from advanced JSON.
ENGINE: dict[str, str | tuple[str, ...]] = {
    'output_format': ('video', 'exr'), 'exr_dtype': ('half', 'float'),
    'frame_dir_fps': 'float', 'frame_dir_linear': 'bool', 'model_scale': 'float',
    'source_colour_profile': ('legacy', 'linear_bt709_d65', 'linear_bt2020_d65'),
    'source_witness': 'bool',
    'frame_dir_alpha': ('ignore', 'premultiplied'),
    'cutout_alpha_mode': ('straight', 'premultiplied'), 'cutout_linear': 'bool',
    'despill_strength': 'float', 'despill_screen': ('green', 'blue'),
    'carry_hint': 'bool', 'hint_quality': 'float', 'carry_hint_quality': 'float',
    'auto_vram': 'bool', 'auto_vram_budget_mb': 'optional_int', 'low_vram': 'bool',
    'hann_chunk': 'int', 'hann_stride': 'int', 'stage2_batch': 'int',
    'foundation_cache_mb': 'int',
    'temp_map_budget_bytes': 'optional_int',
    'process_tree_rss_stop_bytes': 'optional_int',
    'quick_key': 'bool', 'quick_interval': 'int',
    'quick_approach': ('offline', 'hold', 'causal'),
    'quick_fallback': ('stop', 'approximate', 'neural'), 'quick_adaptive': 'bool',
}
WRITER: dict[str, str | tuple[str, ...]] = {
    'ffmpeg_bin': 'text', 'codec': 'text', 'crf': 'int', 'preset': 'text',
    'pix_fmt': 'text', 'bitrate': 'optional_text', 'threads': 'int', 'timeout_seconds': 'float',
}
LABELS = {
    'output_format': 'Output format', 'exr_dtype': 'EXR precision',
    'model_scale': 'Spatial model scale (0 < scale ≤ 1; 1: native)',
    'frame_dir_fps': 'Sequence FPS', 'frame_dir_linear': 'Sequence is scene-linear',
    'source_colour_profile': 'Sequence RGB profile',
    'source_witness': 'Record exact source for bound cache (linear RGB sequence, EXR only)',
    'frame_dir_alpha': 'Source alpha association', 'cutout_alpha_mode': 'Cutout alpha association',
    'cutout_linear': 'Linear cutout RGB', 'despill_strength': 'Despill strength [0,1]',
    'despill_screen': 'Screen colour', 'carry_hint': 'Carry predicted hints',
    'hint_quality': 'Hint trust [0,1]', 'carry_hint_quality': 'Carry trust [0,1]',
    'auto_vram': 'Automatic scheduling', 'auto_vram_budget_mb': 'Auto selection cap MiB (blank: unset)',
    'low_vram': 'Low VRAM', 'hann_chunk': 'Hann chunk frames', 'hann_stride': 'Hann stride frames',
    'stage2_batch': 'Stage 2 batch frames', 'foundation_cache_mb': 'RAM foundation cache MiB (0: off)',
    'temp_map_budget_bytes': 'Managed scratch ceiling bytes (blank: unlimited)',
    'process_tree_rss_stop_bytes': 'Worker tree RSS stop bytes (blank: off)',
    'quick_key': 'Quick Key approximation', 'quick_interval': 'Anchor interval',
    'quick_approach': 'Propagation', 'quick_fallback': 'Uncertainty policy',
    'quick_adaptive': 'Adaptive neural anchors', 'ffmpeg_bin': 'FFmpeg executable',
    'codec': 'Video codec', 'crf': 'CRF', 'preset': 'Encoder preset', 'pix_fmt': 'Pixel format',
    'bitrate': 'Bitrate (blank: unset)', 'threads': 'Encoder threads', 'timeout_seconds': 'Writer timeout seconds',
}


def validated_settings(config: str, writer: str) -> tuple[str, str]:
    wire = job_to_wire(JobRequest.create('.', '.'))
    wire.update(config_json=config, writer_json=writer)
    return pretty_settings(job_from_wire(wire))


class SettingsPanel(QDialog):
    apply_requested = Signal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Common job settings')
        self.resize(670, 740)
        self.original: tuple[str, str] | None = None
        self.controls: dict[tuple[str, str], QCheckBox | QComboBox | QLineEdit] = {}
        self.outputs: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        note = QLabel('Apply validates all settings together. Existing advanced options are retained. EXR is linear; half saturates beyond its range. Video clips display RGB. Quick Key is an explicit approximation. Automatic scheduling is a heuristic, not a memory limit; low VRAM requires RAM cache 0. Blank optional fields mean unset.')
        note.setWordWrap(True)
        layout.addWidget(note)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        for group, specs in (('engine', ENGINE), ('writer', WRITER)):
            page = QWidget()
            form = QFormLayout(page)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            if group == 'engine':
                row = QHBoxLayout()
                for name in ('alpha', 'fg', 'checker', 'cutout'):
                    checkbox = QCheckBox(name)
                    self.outputs[name] = checkbox
                    row.addWidget(checkbox)
                form.addRow('Exports (at least one)', row)
            for name, spec in specs.items():
                if spec == 'bool':
                    control = QCheckBox()
                elif isinstance(spec, tuple):
                    control = QComboBox()
                    control.addItems(list(spec))
                else:
                    control = QLineEdit()
                self.controls[group, name] = control
                form.addRow(LABELS[name], control)
                if name == 'model_scale':
                    scope = QLabel('Below 1 explicitly selects approximate spatial draft processing. Exports retain native dimensions; this is separate from Quick Key temporal/source-RGB draft. Quality is unverified. Full quality resets scale to 1. Apply affects future jobs only.')
                    scope.setWordWrap(True)
                    form.addRow(scope)
                if name == 'source_colour_profile':
                    scope = QLabel('legacy preserves existing assumed BT.709 interpretation. Explicit BT.709 D65 or BT.2020 D65 profiles require scene-linear image sequences; BT.2020 RGB is converted to the working/output BT.709 basis. Signed HDR RGB is preserved; coverage is unchanged. No automatic tag detection or full colour management. Model training primaries remain unverified. Explicit profiles do not support source witnesses. Apply affects future jobs only.')
                    scope.setWordWrap(True)
                    form.addRow(scope)
                if name == 'process_tree_rss_stop_bytes':
                    scope = QLabel('Sampled current resident memory of the worker and tracked descendants; shared pages may count more than once. Blank disables this policy. 0 is a literal threshold. Cooperative checks can miss peaks or overshoot during native work; this is not a hard RAM cap. Unavailable required measurements fail explicitly. Applying affects future jobs only.')
                    scope.setWordWrap(True)
                    form.addRow(scope)
                if name == 'temp_map_budget_bytes':
                    scope = QLabel('0 allows no managed map payload. The ceiling covers aggregate RVM, foundation, baton and anchor logical map bytes. It excludes outputs, caches, filesystem overhead and whole RAM/VRAM usage. Oversize jobs are rejected; quality settings remain unchanged. Apply affects future jobs only.')
                    scope.setWordWrap(True)
                    form.addRow(scope)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            tabs.addTab(scroll, group.title())
        self.status = QLabel('No settings loaded')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        apply_button = QPushButton('Apply to future jobs')
        apply_button.clicked.connect(self.apply)
        buttons.addWidget(apply_button)
        close_button = QPushButton('Close')
        close_button.clicked.connect(self.hide)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def load(self, config: str, writer: str):
        config_clean, writer_clean = validated_settings(config, writer)
        for group, raw in (('engine', config_clean), ('writer', writer_clean)):
            data = json.loads(raw)
            for (scope, name), control in self.controls.items():
                if scope != group:
                    continue
                value = data[name]
                if isinstance(control, QCheckBox):
                    control.setChecked(value)
                elif isinstance(control, QComboBox):
                    if control.findText(value) < 0:
                        raise ValueError(f'Unsupported form value for {name}: {value}')
                    control.setCurrentText(value)
                else:
                    control.setText('' if value is None else str(value))
            if group == 'engine':
                for name, checkbox in self.outputs.items():
                    checkbox.setChecked(name in data['outputs'])
        self.original = (config, writer)
        self.status.setText('Loaded. Apply changes future job settings only; no work is submitted.')
        self.show()

    def values(self) -> tuple[str, str]:
        if self.original is None:
            raise ValueError('Load settings first.')
        records: dict[str, dict[str, Any]] = dict(zip(('engine', 'writer'), (json.loads(raw) for raw in self.original)))
        for (group, name), control in self.controls.items():
            spec = (ENGINE if group == 'engine' else WRITER)[name]
            if isinstance(control, QCheckBox):
                value = control.isChecked()
            elif isinstance(control, QComboBox):
                value = control.currentText()
            else:
                text = control.text()
                try:
                    if spec == 'optional_int' and not text.strip():
                        value = None
                    elif spec in ('int', 'optional_int'):
                        value = int(text)
                    elif spec == 'float':
                        value = float(text)
                    elif spec == 'optional_text' and not text.strip():
                        value = None
                    else:
                        value = text
                except ValueError as exc:
                    raise ValueError(f'{LABELS[name]}: invalid number') from exc
            records[group][name] = value
        records['engine']['outputs'] = [name for name, checkbox in self.outputs.items() if checkbox.isChecked()]
        return validated_settings(*(json.dumps(records[group], allow_nan=False) for group in ('engine', 'writer')))

    def apply(self):
        try:
            config, writer = self.values()
            assert self.original is not None
            self.apply_requested.emit(*self.original, config, writer)
        except (ValueError, TypeError, OSError) as exc:
            self.status.setText(str(exc))
