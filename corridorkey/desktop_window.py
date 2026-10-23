"""Working native desktop shell; inference remains in WorkerClient's child."""
from __future__ import annotations

import json
import queue
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .asset_inventory import AssetSelection
from .desktop_assets import AssetsPanel
from .desktop_benchmark import BenchmarkPanel
from .desktop_bridge import DesktopBridge
from .desktop_cache import CacheMaintenancePanel
from .desktop_comparison import RenderView, common_range, completed_view
from .desktop_corrections import CorrectionDialog
from .desktop_diagnostic_ranges import DiagnosticRangesPanel
from .desktop_loader import PreviewLoader
from .desktop_ofx import OfxCachePanel
from .desktop_playback import PlaybackControls
from .desktop_portable_project import PortableProjectDialog
from .desktop_preview import PreviewPair
from .desktop_progressive import ProgressivePreviewPanel
from .desktop_queue import QueuePanel
from .desktop_readiness import ReadinessPanel
from .desktop_recovery import RecoveryPanel
from .desktop_resources import ResourcesPanel
from .desktop_settings import SettingsPanel
from .desktop_state import (
    DESKTOP_MAX_PREVIEW_FRAME,
    load_project,
    pretty_settings,
    save_project,
    settings_job,
)
from .desktop_viewport import PreviewViewport, ViewportState
from .preview_sequence import PreparedSequence
from .session import JobRequest
from .worker_client import WorkerClient
from .worker_protocol import decode_frame, job_from_wire, job_to_wire, request_frame

TERMINAL = {'result', 'failed', 'cancelled', 'rejected', 'lost'}


class DesktopWindow(QMainWindow):
    def __init__(self, bridge: DesktopBridge | None = None):
        super().__init__()
        self.setWindowTitle('CorridorKey · Keying workspace')
        self.resize(1240, 880)
        self.setAcceptDrops(True)
        self.bridge = bridge if bridge is not None else DesktopBridge()
        self.loader = PreviewLoader()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.closing = False
        self.preview_serial = 0
        self._status_revision = 0
        self._details_revision = 0
        self._comparison_error: tuple[str, int, int] | None = None
        self.statusBar().messageChanged.connect(self._status_changed)
        self.background: tuple[float, float, float] | None = None
        self.pixmap: QPixmap | None = None
        self.reference: QPixmap | None = None
        self.preview_description = ''
        self.moving_reference: RenderView | None = None
        self.reference_mode = ''
        self.reference_description = ''
        self.editor: CorrectionDialog | None = None
        self.portable_project: PortableProjectDialog | None = None
        self.portable_application_records: list[dict[str, Any]] = []
        self.pending_queue_root: Path | None = None
        self.asset_selection: AssetSelection | None = None
        self.resource_request: str | None = None
        self._diagnostic_binding: tuple[Any, ...] | None = None
        self._diagnostic_context: str | None = None
        self._diagnostic_editor_revision = 0
        self._diagnostic_prepare_revision: tuple[str, int] | None = None
        self._diagnostic_preview_context: str | None = None
        self._updating_diagnostic_rows = False
        self.diagnostic_ranges = DiagnosticRangesPanel(self)
        self.diagnostic_ranges.preview_requested.connect(self.preview_diagnostic_range)
        self.diagnostic_ranges.prepare_started.connect(self.begin_diagnostic_preparation)
        self.diagnostic_ranges.prepared_job.connect(self.apply_diagnostic_preparation)
        self.resources = ResourcesPanel(self)
        self.ofx_cache = OfxCachePanel(self)
        self.cache_panel = CacheMaintenancePanel(self)
        self.resources.worker_requested.connect(self.request_resources)
        self.assets = AssetsPanel(self)
        self.assets.use_requested.connect(self.activate_assets)
        self.settings_panel = SettingsPanel(self)
        self.settings_panel.apply_requested.connect(self.apply_common_settings)
        self.queue_panel = QueuePanel(self)
        self.queue_panel.activate_requested.connect(self.activate_queue)
        self.queue_panel.load_requested.connect(self.load_queue_settings)
        self.recovery = RecoveryPanel(self)
        self.recovery.run_requested.connect(self.submit_recovery)
        self.recovery.cancel_requested.connect(self.cancel_recovery)
        self.queue_panel.recovery_requested.connect(self.recovery.load_payload)
        self.pending_benchmark: tuple[Path, Path] | None = None
        self.benchmark = BenchmarkPanel(self)
        self.benchmark.run_requested.connect(self.start_benchmark)
        self.benchmark.cancel_requested.connect(self.cancel_benchmark)
        self.readiness = ReadinessPanel(self)
        self.readiness.run_requested.connect(self.submit_readiness)
        self._build()
        self.progressive = ProgressivePreviewPanel(self)
        self.source.textChanged.connect(self.playback.navigate)
        self.config.textChanged.connect(self.playback.navigate)
        for field in (self.source, self.output, self.first_hint, self.full_hint):
            field.textChanged.connect(self.diagnostic_editor_changed)
        self.config.textChanged.connect(self.diagnostic_editor_changed)
        self.writer.textChanged.connect(self.diagnostic_editor_changed)
        self.profile.toggled.connect(self.diagnostic_editor_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        self._button('Readiness / first render…', self.readiness.show, toolbar)
        self._button('Open video…', self.choose_video, toolbar)
        self._button('Open frames…', self.choose_frames, toolbar)
        projects = QPushButton('Projects…')
        project_menu = QMenu(projects)
        project_menu.addAction('Open desktop snapshot…', self.choose_project)
        project_menu.addAction('Save desktop snapshot…', self.save_project)
        project_menu.addAction('Portable projects…', self.open_portable_project)
        projects.setMenu(project_menu)
        toolbar.addWidget(projects)
        self._button('Developer benchmarks…', self.benchmark.show, toolbar)
        self._button('Queue history…', self.queue_panel.show, toolbar)
        self._button('Recovery…', self.recovery.show, toolbar)
        self._button('Local assets…', self.assets.show, toolbar)
        corrections = QPushButton('Corrections…')
        menu = QMenu(corrections)
        menu.addAction('Edit attached / new document', self.open_editor)
        menu.addAction('Open correction document…', self.open_editor_document)
        corrections.setMenu(menu)
        toolbar.addWidget(corrections)
        toolbar.addStretch()
        self._button('Stop worker', self.stop_worker, toolbar)
        self._button('New worker', self.new_worker, toolbar)
        layout.addLayout(toolbar)
        self.backend = QLabel('CUDA inference required · Assets checked by worker at render · Live GPU memory unavailable')
        self.backend.setWordWrap(True)
        backend_bar = QHBoxLayout()
        backend_bar.addWidget(self.backend, 1)
        self._button('Progressive preview…', lambda: self.progressive.show(), backend_bar)
        self._button('Resources…', self.resources.show, backend_bar)
        layout.addLayout(backend_bar)
        self.queue_label = QLabel('Queue: session only · choose Queue history to create/open durable storage')
        self.queue_label.setWordWrap(True)
        layout.addWidget(self.queue_label)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)
        left, right = QWidget(), QWidget()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([390, 830])
        controls, workspace = QVBoxLayout(left), QVBoxLayout(right)
        form = QFormLayout()
        self.source, self.output = QLineEdit(), QLineEdit()
        self.output.setPlaceholderText('Each render creates a new child directory')
        self.first_hint, self.full_hint = QLineEdit(), QLineEdit()
        form.addRow('Source (or drop one path)', self.source)
        form.addRow('Output parent', self.output)
        self._button('Choose output parent…', self.choose_output, controls)
        form.addRow('First hint (optional)', self.first_hint)
        form.addRow('Full hint (optional)', self.full_hint)
        controls.addLayout(form)
        quality = QHBoxLayout()
        self._button('Full quality', self.full_quality, quality)
        self._button('Quick Key draft', self.draft_quality, quality)
        controls.addLayout(quality)
        self.quality_note = QLabel('Full quality by default. Quick Key uses sparse neural anchors and current source RGB.')
        self.quality_note.setWordWrap(True)
        controls.addWidget(self.quality_note)
        limits = QFormLayout()
        self.limit = QSpinBox()
        self.limit.setRange(0, 1000000000)
        self.limit.setSpecialValueText('All frames')
        self.start = QComboBox()
        self.start.addItems(['begin', 'middle', 'random_middle'])
        limits.addRow('Frame limit', self.limit)
        limits.addRow('Video start', self.start)
        controls.addLayout(limits)
        self.limit.valueChanged.connect(self.change_range)
        self.start.currentTextChanged.connect(self.change_range)
        source_range = QHBoxLayout()
        self.range_in = QLineEdit('0')
        self.range_out = QLineEdit()
        self.range_out.setPlaceholderText('Source end')
        source_range.addWidget(QLabel('Source In'))
        source_range.addWidget(self.range_in)
        source_range.addWidget(QLabel('Out (exclusive)'))
        source_range.addWidget(self.range_out)
        controls.addLayout(source_range)
        range_actions = QHBoxLayout()
        self._button('Mark In', partial(self.mark_range, False), range_actions)
        self._button('Mark Out', partial(self.mark_range, True), range_actions)
        self._button('Apply source in/out', self.apply_source_range, range_actions)
        controls.addLayout(range_actions)
        self.range_note = QLabel('Apply sets future renders; blank Out means source end. Existing queued jobs keep their range.')
        self.range_note.setWordWrap(True)
        controls.addWidget(self.range_note)
        self.profile = QCheckBox('Profile stages (adds synchronization overhead)')
        controls.addWidget(self.profile)
        self._button('Common job settings…', self.open_common_settings, controls)
        advanced = QTabWidget()
        self.config, self.writer = QPlainTextEdit(), QPlainTextEdit()
        default = JobRequest.create('.', '.')
        config, writer = pretty_settings(default)
        self.config.setPlainText(config)
        self.writer.setPlainText(writer)
        self.config.textChanged.connect(self.sync_range)
        self.config.textChanged.connect(self.sync_quality)
        advanced.addTab(self.config, 'Advanced engine settings')
        advanced.addTab(self.writer, 'Writer settings')
        controls.addWidget(advanced, 1)
        help_text = QLabel('All fields use shared core names and validation. corrections_path attaches an existing saved correction document. foundation_cache_mb=0 disables RAM reuse. EXR is linear; video is clipped preview output.')
        help_text.setWordWrap(True)
        controls.addWidget(help_text)
        self._button('Add render to queue', self.enqueue, controls)
        self._button('Prepare current job for recovery…', self.prepare_recovery, controls)
        viewer_bar = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems(['Source', 'Alpha', 'Foreground', 'Checker composite', 'Solid composite'])
        viewer_bar.addWidget(self.view)
        self.frame = QSpinBox()
        self.frame.setRange(0, DESKTOP_MAX_PREVIEW_FRAME)
        self.frame_label = QLabel('Frame (zero-based)')
        viewer_bar.addWidget(self.frame_label)
        viewer_bar.addWidget(self.frame)
        self._button('Preview', self.request_preview, viewer_bar)
        self._button('Background…', self.choose_background, viewer_bar)
        workspace.addLayout(viewer_bar)
        self.viewport_state = ViewportState(self)
        self.image = PreviewViewport(self.viewport_state, 'Open a video or frame directory, then select Preview.\nExport views use a selected completed queue job.')
        self.image.setMinimumSize(300, 220)
        self.image.setStyleSheet('background:#15191d;color:#cbd2d9;border:1px solid #45515c')
        comparison = QHBoxLayout()
        comparison.addWidget(self.image, 1)
        self.reference_image = PreviewViewport(self.viewport_state, 'A · Pin a displayed frame for comparison')
        self.reference_image.setMinimumSize(180, 160)
        self.reference_image.setStyleSheet('background:#15191d;color:#cbd2d9;border:1px solid #45515c')
        comparison.addWidget(self.reference_image, 1)
        workspace.addLayout(comparison, 3)
        comparison_bar = QHBoxLayout()
        self._button('Pin displayed frame as A', self.pin_reference, comparison_bar)
        self._button('Selected render as synchronized A', self.synchronize_reference, comparison_bar)
        self._button('Clear A', self.clear_reference, comparison_bar)
        self._button('Zoom in', lambda: self.viewport_state.zoom_by(1.25), comparison_bar)
        self._button('Zoom out', lambda: self.viewport_state.zoom_by(.8), comparison_bar)
        self._button('Fit A/B', self.viewport_state.fit, comparison_bar)
        self.zoom_label = QLabel(self.viewport_state.label)
        self.zoom_label.setToolTip('Matched A/B wheel zoom and left-drag pan. Scale refers to the decoded preview, capped at 1280 pixels; no native-pixel inspection claim.')
        self.viewport_state.changed.connect(lambda: self.zoom_label.setText(self.viewport_state.label))
        comparison_bar.addWidget(self.zoom_label)
        workspace.addLayout(comparison_bar)
        self.reference_note = QLabel('A is a fixed display snapshot; compare the same frame and view for render A/B.')
        self.reference_note.setWordWrap(True)
        self.reference_note.setMinimumHeight(44)
        workspace.addWidget(self.reference_note)
        self.preview_note = QLabel('No model preview has been rendered.')
        self.preview_note.setWordWrap(True)
        self.preview_note.setMinimumHeight(44)
        workspace.addWidget(self.preview_note)
        self.playback = PlaybackControls(self)
        workspace.addWidget(self.playback)
        self.config.textChanged.connect(self.invalidate_source_preview)
        self.frame.valueChanged.connect(self.refresh_comparison)
        self.view.currentTextChanged.connect(self.refresh_comparison)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Source', 'Status', 'Stage / work', 'Output directory'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.show_record)
        self.table.itemSelectionChanged.connect(self.playback.selection_changed)
        workspace.addWidget(self.table, 2)
        queue_bar = QHBoxLayout()
        self._button('Cancel selected', self.cancel_selected, queue_bar)
        self._button('Clear finished', self.clear_finished, queue_bar)
        self._button('Diagnostic ranges…', self.open_diagnostic_ranges, queue_bar)
        self._button('Prepare OFX cache…', self.open_ofx_cache, queue_bar)
        self._button('Manage recovery cache…', self.open_cache_management, queue_bar)
        workspace.addLayout(queue_bar)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.textChanged.connect(self._details_changed)
        self.details.setPlaceholderText('Actual worker events, errors and completed diagnostics appear here.')
        workspace.addWidget(self.details, 1)
        self.statusBar().showMessage('Ready · No model loaded')

    def invalidate_source_preview(self):
        if self.view.currentText() == 'Source' and self.moving_reference is None:
            self.preview_serial = -1
            self.pixmap = None
            self.image.setText('Settings changed. Select Preview to decode this source.')
            self.preview_note.setText('Source interpretation applies to the next explicit preview; no work submitted.')

    def _status_changed(self, _message):
        self._status_revision += 1

    def _details_changed(self):
        self._details_revision += 1

    def error(self, message):
        self._comparison_error = None
        self.statusBar().showMessage(str(message))
        self.details.setPlainText(str(message))

    def _comparison_failed(self, message):
        self.error(message)
        self._comparison_error = (str(message), self._status_revision, self._details_revision)

    def _comparison_recovered(self):
        owned = self._comparison_error
        self._comparison_error = None
        if owned is None:
            return
        message, status_revision, details_revision = owned
        # Independent revisions preserve newer worker/UI information, including
        # messages that changed away and back to the same text during decoding.
        if self._status_revision == status_revision and self.statusBar().currentMessage() == message:
            self.statusBar().showMessage(f'Synchronized comparison ready · source frame {self.frame.value()}')
        if self._details_revision == details_revision and self.details.toPlainText() == message:
            request_id = self.selected()
            if request_id is not None:
                self.details.setPlainText(json.dumps(self.jobs[request_id].get('event'), indent=2))
            else:
                self.details.clear()

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open video', '', 'Video (*.mov *.mp4 *.mkv *.avi *.webm);;All files (*)')
        if path:
            self.set_source(path)

    def choose_frames(self):
        path = QFileDialog.getExistingDirectory(self, 'Open frame directory')
        if path:
            self.set_source(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Select output parent')
        if path:
            self.output.setText(path)

    def set_source(self, path):
        if self.moving_reference is not None:
            self.clear_reference()
        self.source.setText(str(Path(path).resolve()))
        self.frame.setValue(0)
        self.view.setCurrentIndex(0)
        self.image.setText('Select Preview to decode this source.')
        # Invalidate an in-flight request from the previous source.
        self.preview_serial = -1
        self.pixmap = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        data = event.mimeData()
        if data is not None and len(data.urls()) == 1 and data.urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        data = event.mimeData()
        if data is not None and len(data.urls()) == 1 and data.urls()[0].isLocalFile():
            self.set_source(data.urls()[0].toLocalFile())
            event.acceptProposedAction()

    def _settings(self):
        return decode_frame(self.config.toPlainText().encode() + b'\n')

    def open_common_settings(self):
        try:
            self.settings_panel.load(self.config.toPlainText(), self.writer.toPlainText())
        except (ValueError, TypeError, OSError) as exc:
            self.error(exc)

    def apply_common_settings(self, old_config: str, old_writer: str, config: str, writer: str):
        if (self.config.toPlainText(), self.writer.toPlainText()) != (old_config, old_writer):
            self.settings_panel.status.setText('Settings changed since this form opened. Reopen Common job settings before applying.')
            return
        self.config.setPlainText(config)
        self.writer.setPlainText(writer)
        self.settings_panel.original = (config, writer)
        self.settings_panel.status.setText('Validated settings applied to future jobs. No work submitted.')

    def update_settings(self, changes):
        try:
            settings = self._settings()
            settings.update(changes)
            self.config.setPlainText(json.dumps(settings, indent=2))
            return True
        except ValueError as exc:
            self.error(exc)
            return False

    def sync_range(self):
        try:
            settings = self._settings()
        except ValueError:
            return  # Incomplete JSON during editing is validated again before use.
        count = settings.get('num_frames')
        start = settings.get('start_mode')
        if type(count) is int and count <= self.limit.maximum():
            self.limit.blockSignals(True)
            self.limit.setValue(max(0, count))
            self.limit.blockSignals(False)
        if start in ('begin', 'middle', 'random_middle'):
            self.start.blockSignals(True)
            self.start.setCurrentText(start)
            self.start.blockSignals(False)
        explicit = settings.get('start_frame')
        if explicit is None:
            self.range_in.setText('0' if start == 'begin' else '')
        elif type(explicit) is int:
            self.range_in.setText(str(explicit))
        if type(count) is int and count > 0 and (type(explicit) is int or start == 'begin'):
            self.range_out.setText(str((explicit or 0) + count))
        else:
            self.range_out.clear()

    def mark_range(self, out: bool):
        try:
            index = self.frame.value()
            if self.moving_reference is None and self.view.currentText() != 'Source':
                selected = self.selected_render_view()
                if index >= selected.count:
                    raise ValueError('Preview export frame is outside the selected render.')
                index += selected.start
            (self.range_out if out else self.range_in).setText(str(index + int(out)))
            self.range_note.setText('Range staged. Apply source in/out to change future renders.')
        except (ValueError, TypeError, OSError) as exc:
            self.error(exc)

    def apply_source_range(self):
        try:
            start = int(self.range_in.text().strip())
            end_text = self.range_out.text().strip()
            end = int(end_text) if end_text else None
            if start < 0 or (end is not None and end <= start):
                raise ValueError('Source In must be nonnegative and Out must be greater than In.')
            changes = {'start_frame': start, 'num_frames': end - start if end is not None else -1, 'start_mode': 'begin'}
            # Validate the full future configuration before touching existing text.
            settings = self._settings()
            settings.update(changes)
            settings_job(self.source.text(), self.output.text() or str(Path.cwd()),
                         json.dumps(settings), self.writer.toPlainText())
            if self.update_settings(changes):
                self.range_note.setText('Source range applied to future jobs. Source EOF caps Out; corrections must match the selected timeline.')
        except (ValueError, TypeError, OSError) as exc:
            self.error(exc)

    def sync_quality(self):
        try:
            settings = self._settings()
        except ValueError:
            self.quality_note.setText('Future quality settings incomplete; validate before rendering.')
            return
        scale = settings.get('model_scale', 1.0)
        if type(scale) not in (int, float) or not 0 < scale <= 1:
            self.quality_note.setText('Invalid spatial scale: choose a finite number in (0, 1].')
            return
        modes = []
        if scale < 1:
            modes.append(f'Spatial draft ×{scale:g}: approximate processing, native-sized exports; quality unverified.')
        interval = settings.get('quick_interval', 1)
        if settings.get('quick_key') is True and type(interval) is int and interval > 1:
            modes.append('Quick Key draft: sparse neural anchors and source-RGB foreground.')
        self.quality_note.setText('Future jobs · ' + (' '.join(modes) if modes else
            'Native spatial scale 1; full-rate neural processing. Authored corrections remain attached.'))

    def full_quality(self):
        if self.update_settings({'model_scale': 1.0, 'quick_key': False, 'quick_approach': 'offline', 'quick_auto_cuts': False,
                                 'quick_adaptive': False, 'quick_max_gap': None, 'quick_cut_frames': [], 'quick_reset_frames': []}):
            self.sync_quality()

    def draft_quality(self):
        if self.update_settings({'quick_key': True, 'quick_interval': 5, 'quick_approach': 'offline', 'quick_fallback': 'stop',
                                 'quick_adaptive': False}):
            self.sync_quality()

    def change_range(self, *_):
        changes: dict[str, Any] = {'num_frames': self.limit.value() or -1, 'start_mode': self.start.currentText()}
        if self.start.currentText() != 'begin':
            changes['start_frame'] = None
        self.update_settings(changes)

    def job(self, *, preview_only=False):
        return settings_job(self.source.text(), self.output.text() or (str(Path.cwd()) if preview_only else ''),
                            self.config.toPlainText(), self.writer.toPlainText(),
                            first_hint=self.first_hint.text(), full_hint=self.full_hint.text(), profile=self.profile.isChecked())

    def submit_readiness(self, job):
        self.progressive.suspend_for_export()
        try:
            if self.readiness.prepared is None or job != self.readiness.prepared or self.readiness.submitted_id is not None:
                raise ValueError('Prepare and review a fresh short-render snapshot before submission.')
            if self.closing or self.benchmark.busy or self.pending_queue_root is not None:
                raise ValueError('Wait for benchmark, queue activation or shutdown before rendering.')
            if len(self.jobs) >= 128:
                raise ValueError('Queue history is full. Clear finished jobs before submitting more.')
            if self.benchmark.task is not None and self.benchmark.task.snapshot()['cleanup_error']:
                raise ValueError('Previous benchmark cleanup is unverified; cannot start another owner.')
            if self.asset_selection != self.readiness.prepared_selection:
                raise ValueError('Selected model assets changed. Prepare and review a new short-render snapshot.')
            request_id = self.bridge.send('render', {'job': job_to_wire(job)})
            self.jobs[request_id] = {'job': job, 'status': 'sending', 'event': None}
            self.readiness.mark_submitted(request_id)
            self.refresh_table()
            self.statusBar().showMessage('Explicit short render submitted; current settings preserved')
        except (ValueError, OSError, RuntimeError, queue.Full) as exc:
            self.readiness.status.setText(str(exc))

    def enqueue(self):
        self.progressive.suspend_for_export()
        try:
            if self.pending_queue_root is not None:
                raise ValueError('Wait for queue workspace activation.')
            if self.benchmark.busy:
                raise ValueError('Benchmark owns processing. Wait for completion and select New worker before rendering.')
            if len(self.jobs) >= 128:
                raise ValueError('Queue history is full. Clear finished jobs before submitting more.')
            parent_job = self.job()
            output = Path(parent_job.output_dir) / f'corridorkey-{uuid.uuid4().hex}'
            job = settings_job(parent_job.input_path, str(output), parent_job.config_json, parent_job.writer_json,
                               first_hint=parent_job.initial_hint_path or '', full_hint=parent_job.hint_video_path or '',
                               profile=parent_job.profile_stages)
            request_id = self.bridge.send('render', {'job': job_to_wire(job)})
            self.jobs[request_id] = {'job': job, 'status': 'sending', 'event': None}
            self.refresh_table()
            self.statusBar().showMessage('Submitted locally; waiting for worker acceptance')
        except (ValueError, OSError, RuntimeError, queue.Full) as exc:
            self.error(exc)

    def selected(self):
        selection = self.table.selectionModel()
        rows = selection.selectedRows() if selection is not None else []
        row = rows[0].row() if len(rows) == 1 else -1
        return list(self.jobs)[row] if 0 <= row < len(self.jobs) else None

    @staticmethod
    def row_paths(state):
        if state.get('action') == 'recover':
            return Path(state['payload']['batch']).name + ' [recovery]', state['payload']['workspace']
        return Path(state['job'].input_path).name, state['job'].output_dir

    def refresh_table(self):
        selected = self.selected()
        self._updating_diagnostic_rows = True
        try:
            self.table.setRowCount(len(self.jobs))
            for row, (request_id, state) in enumerate(self.jobs.items()):
                source, output = self.row_paths(state)
                event = state.get('event') or {}
                payload = event.get('payload', {})
                stage = str(payload.get('stage', ''))
                if stage:
                    stage += f" {payload.get('completed', 0)}/{payload.get('total', '?')}"
                for col, value in enumerate([source, state['status'], stage, output]):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))
                if request_id == selected:
                    self.table.selectRow(row)
        finally:
            self._updating_diagnostic_rows = False
            self.refresh_diagnostic_context()

    def cancel_selected(self):
        request_id = self.selected()
        if request_id is not None and self.jobs[request_id]['status'] not in TERMINAL:
            try:
                self.bridge.send('cancel', {'target': request_id, 'reason': 'Cancelled from desktop'})
                self.statusBar().showMessage('Cancellation requested; waiting for terminal event')
            except (RuntimeError, queue.Full) as exc:
                self.error(exc)

    def clear_finished(self):
        self.jobs = {key: value for key, value in self.jobs.items() if value['status'] not in TERMINAL}
        self.table.clearSelection()
        self.refresh_table()

    def show_record(self):
        if self._updating_diagnostic_rows:
            return
        request_id = self.selected()
        if request_id is not None:
            self.details.setPlainText(json.dumps(self.jobs[request_id].get('event'), indent=2))
        if not ((self.playback.engine is not None or self.playback.preparing)
                and request_id == self.playback.selected_id):
            self.refresh_comparison()
        self.refresh_diagnostic_context()

    def diagnostic_selection(self) -> tuple[str, JobRequest, dict[str, Any]] | None:
        request_id = self.selected()
        if request_id is None:
            return None
        state = self.jobs[request_id]
        if state.get('action') == 'recover' or state.get('status') not in {'result', 'failed', 'cancelled'}:
            return None
        job = state.get('job')
        record = (state.get('event') or {}).get('payload', {}).get('record')
        if not isinstance(job, JobRequest) or not isinstance(record, dict):
            return None
        return request_id, job, record

    def refresh_diagnostic_context(self):
        if self._updating_diagnostic_rows:
            return
        selected = self.diagnostic_selection()
        binding = ((selected[0], id(selected[1]), id(selected[2]), self.diagnostic_protected_paths())
                   if selected is not None else None)
        if binding == self._diagnostic_binding:
            return
        self._diagnostic_binding = binding
        self._diagnostic_context = uuid.uuid4().hex if binding is not None else None
        self._diagnostic_prepare_revision = None
        self.diagnostic_ranges.invalidate_context()
        if self._diagnostic_preview_context is not None:
            self._diagnostic_preview_context = None
            self.preview_serial = -1
            self.pixmap = None
            self.image.setText('Diagnostic selection changed. Inspect the selected job again.')

    def diagnostic_editor_changed(self, *_):
        self._diagnostic_editor_revision += 1

    def diagnostic_protected_paths(self) -> tuple[str, ...]:
        protected = tuple(state['job'].output_dir for state in self.jobs.values()
                          if isinstance(state.get('job'), JobRequest))
        if self.asset_selection is not None:
            protected += (str(self.asset_selection.root),)
            if self.asset_selection.weights is not None:
                protected += (str(self.asset_selection.weights),)
        return protected

    def open_diagnostic_ranges(self):
        try:
            if self.closing:
                raise ValueError('The workspace is closing.')
            self.refresh_diagnostic_context()
            selected = self.diagnostic_selection()
            if selected is None or self._diagnostic_context is None:
                self.diagnostic_ranges.invalidate_context()
                raise ValueError('Select a terminal normal render with a recorded source range and diagnostics.')
            _, job, record = selected
            self.diagnostic_ranges.inspect_selection(job, record, self._diagnostic_context,
                existing_parent=str(Path(job.output_dir).parent), protected_paths=self.diagnostic_protected_paths())
            self.diagnostic_ranges.show()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.error(exc)

    def preview_diagnostic_range(self, options: dict[str, Any], context: str):
        self.playback.navigate()
        try:
            self.refresh_diagnostic_context()
            if self.closing or context != self._diagnostic_context:
                raise ValueError('Diagnostic selection changed. Inspect the selected job again.')
            selected = self.diagnostic_selection()
            if selected is None or options.get('path') != selected[1].input_path:
                raise ValueError('Diagnostic preview does not match the selected source.')
            index = options.get('index')
            source = selected[2].get('diagnostics', {}).get('source', {})
            start, count = source.get('start_frame'), source.get('num_frames')
            if (type(index) is not int or type(start) is not int or type(count) is not int
                    or not start <= index < start + count or index > self.frame.maximum()):
                raise ValueError('Diagnostic preview lies outside the recorded source range.')
            if self.moving_reference is not None:
                self.clear_reference()
            frame_blocked = self.frame.blockSignals(True)
            view_blocked = self.view.blockSignals(True)
            try:
                self.frame.setValue(index)
                self.view.setCurrentText('Source')
            finally:
                self.frame.blockSignals(frame_blocked)
                self.view.blockSignals(view_blocked)
            self.frame_label.setText('Source frame (zero-based)')
            self.preview_serial = self.loader.submit(**options)
            self._diagnostic_preview_context = context
            self.preview_description = f'Diagnostic source · source frame {index} · {options["path"]}'
            self.pixmap = None
            self.image.setText('Decoding selected diagnostic source frame…')
            self.preview_note.setText('Captured source interpretation; diagnostic events do not certify output quality.')
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.error(exc)

    def begin_diagnostic_preparation(self, context: str):
        self.refresh_diagnostic_context()
        self._diagnostic_prepare_revision = ((context, self._diagnostic_editor_revision)
            if not self.closing and context == self._diagnostic_context else None)

    def apply_diagnostic_preparation(self, job: JobRequest, context: str):
        self.refresh_diagnostic_context()
        expected = (context, self._diagnostic_editor_revision)
        if self.closing or context != self._diagnostic_context or self._diagnostic_prepare_revision != expected:
            self._diagnostic_prepare_revision = None
            message = ('Prepared settings were not loaded because selection or future settings changed. '
                       'Inspect and prepare again; an empty reserved range directory may remain.')
            self.diagnostic_ranges.status.setText(message)
            self.error(message)
            return
        self._diagnostic_prepare_revision = None
        config, _ = job.configurations()
        self.populate_job(job, frame=config.start_frame or 0, output_parent=str(Path(job.output_dir).parent))
        self.range_note.setText('Standalone range settings prepared. Recurrent state restarts at In. '
                                'Add render to queue explicitly to process; earlier jobs and exports are unchanged.')
        self.diagnostic_ranges.status.setText('Prepared settings loaded for a fresh range. No job submitted.')

    def open_cache_management(self):
        if self.closing:
            self.statusBar().showMessage('Workspace is closing; wait for owned cache I/O cleanup.')
            return
        if not self.cache_panel.io_busy and not self.cache_panel.root_path.text().strip():
            self.cache_panel.set_workspace(self.recovery.workspace.text())
        self.cache_panel.show()

    def open_ofx_cache(self):
        try:
            if self.closing:
                raise ValueError('Workspace is closing; wait for cleanup.')
            request_id = self.selected()
            if request_id is None or self.jobs[request_id]['status'] != 'result':
                raise ValueError('Select a completed render in the queue first.')
            state = self.jobs[request_id]
            if state.get('action') == 'recover':
                raise ValueError('A recovery request contains multiple jobs; inspect its per-job report first.')
            self.ofx_cache.load_record(state['event']['payload']['record'])
            self.ofx_cache.show()
        except (ValueError, TypeError, KeyError, OSError) as exc:
            if not self.closing:
                self.ofx_cache.reject_selection(str(exc))
            self.error(exc)

    def request_preview(self):
        self.playback.navigate()
        self._diagnostic_preview_context = None
        try:
            if self.moving_reference is not None:
                self.request_comparison()
                return
            options: dict[str, Any] = {'index': self.frame.value()}
            mode = self.view.currentText()
            if mode == 'Source':
                job = self.job(preview_only=True)
                config, _ = job.configurations()
                options.update(path=job.input_path, linear=config.frame_dir_linear if Path(job.input_path).is_dir() else False,
                               alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile)
            else:
                request_id = self.selected()
                if request_id is None or self.jobs[request_id]['status'] != 'result':
                    raise ValueError('Select a completed render for an export preview.')
                if self.jobs[request_id].get('action') == 'recover':
                    raise ValueError('Inspect the actual recovery report for per-job outputs; this is a batch request.')
                record = self.jobs[request_id]['event']['payload']['record']
                diagnostics = record['diagnostics']
                outputs = diagnostics['outputs']
                kind = 'alpha' if mode == 'Alpha' else 'fg'
                if kind not in outputs:
                    raise ValueError(f'This render did not export {kind}. Choose that output in the engine settings and render.')
                config, _ = self.jobs[request_id]['job'].configurations()
                options.update(path=outputs[kind], coverage=kind == 'alpha', linear=config.output_format == 'exr')
                if 'composite' in mode:
                    if 'alpha' not in outputs:
                        raise ValueError('Compositing requires both alpha and foreground exports.')
                    options['alpha_path'] = outputs['alpha']
                    options['background'] = self.background if mode == 'Solid composite' else None
                    if mode == 'Solid composite' and self.background is None:
                        raise ValueError('Choose a background colour first.')
            self.preview_serial = self.loader.submit(**options)
            self.preview_description = f"{mode} · frame {self.frame.value()} · {options['path']}"
            self.pixmap = None
            self.image.setText('Decoding…')
            self.preview_note.setText('Decoding requested frame…')
        except (ValueError, OSError, RuntimeError) as exc:
            self.error(exc)

    def selected_render_view(self) -> RenderView:
        request_id = self.selected()
        if request_id is None or self.jobs[request_id]['status'] != 'result':
            raise ValueError('Select a completed render for synchronized comparison.')
        state = self.jobs[request_id]
        if state.get('action') == 'recover':
            raise ValueError('Recovery batch requests require per-job report inspection.')
        return completed_view(state['job'], state['event']['payload']['record'])

    def synchronize_reference(self):
        self.playback.navigate()
        try:
            if self.view.currentText() == 'Source':
                raise ValueError('Select an export view (Alpha, Foreground or composite) for A first.')
            reference = self.selected_render_view()
            reference.options(self.view.currentText(), reference.start, self.background)
            self.moving_reference = reference
            self.reference_mode = self.view.currentText()
            self.reference = None
            self.frame.blockSignals(True)
            self.frame.setValue(reference.start)
            self.frame.blockSignals(False)
            self.frame_label.setText('Source frame (zero-based)')
            self.request_comparison()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.error(exc)

    def refresh_comparison(self, *_):
        self.playback.navigate()
        if self.moving_reference is not None:
            self.request_preview()

    def request_comparison(self):
        self.playback.navigate()
        a = self.moving_reference
        assert a is not None
        try:
            index, mode = self.frame.value(), self.view.currentText()
            if mode == 'Source':
                if self.source.text().strip() != a.source:
                    raise ValueError('Source view must use the same source as synchronized A.')
                # A's immutable source interpretation also defines synchronized Source B.
                b_options = a.options('Source', index)
                start, end = a.start, a.end
            else:
                b = self.selected_render_view()
                start, end = common_range(a, b)
                if not start <= index < end:
                    raise ValueError(f'Choose a common source frame in [{start}, {end}).')
                b_options = b.options(mode, index, self.background)
            a_options = a.options(self.reference_mode, index, self.background)
            self.preview_serial = self.loader.submit(pair={'a': a_options, 'b': b_options})
            self.reference_description = f'{self.reference_mode} · source frame {index} · export frame {index-a.start} · {a_options["path"]}'
            self.preview_description = f'{mode} · source frame {index} · decoded frame {b_options["index"]} · {b_options["path"]}'
            self.pixmap = self.reference = None
            self.image.setText('Decoding synchronized B…')
            self.reference_image.setText('Decoding synchronized A…')
            self.reference_note.setText(f'Common source range [{start}, {end}) · A view fixed at {self.reference_mode}')
            self.preview_note.setText('Decoding both frames; source path/range alignment, not content authentication.')
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            # Invalidate previous in-flight pairs when the new selection cannot align.
            self.preview_serial = -1
            self.pixmap = self.reference = None
            self.image.setText('Comparison unavailable')
            self.reference_image.setText('Comparison unavailable')
            self.preview_note.setText(str(exc))
            self._comparison_failed(exc)

    def choose_background(self):
        colour = QColorDialog.getColor(parent=self, title='Composite background (sRGB)')
        if colour.isValid():
            self.background = (colour.redF(), colour.greenF(), colour.blueF())
            self.view.setCurrentText('Solid composite')
            self.request_preview()

    def pin_reference(self):
        self.playback.navigate()
        if self.pixmap is None:
            self.error('Decode a frame successfully before pinning a comparison.')
            return
        self.moving_reference = None
        self.frame_label.setText('Frame (zero-based)')
        self.reference = self.pixmap.copy()
        self.reference_image.setPixmap(self.reference)
        self.reference_note.setText('A · ' + self.preview_description.rsplit(' · ', 1)[0])
        self.reference_note.setToolTip(self.preview_description)

    def clear_reference(self):
        self.playback.navigate()
        self.moving_reference = None
        self.preview_serial = -1
        self.frame_label.setText('Frame (zero-based)')
        self.reference = None
        self.reference_image.setText('A · Pin a displayed frame for comparison')
        self.reference_note.setText('A is a fixed display snapshot; compare the same frame and view for render A/B.')

    def choose_project(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open desktop project', '', 'JSON (*.json)')
        if path:
            self.open_project(Path(path))

    def open_project(self, path: Path):
        try:
            job, frame = load_project(path)
            self.populate_job(job, frame=frame)
        except (ValueError, OSError, TypeError) as exc:
            self.error(exc)

    def open_portable_project(self):
        if self.closing:
            return
        if self.editor is not None and self.editor.canvas.dirty:
            self.error('Commit or Clear the correction draft before preparing a portable project.')
            return
        if self.portable_project is not None:
            if self.portable_project.isVisible() or not self.portable_project.shutdown():
                self.portable_project.show()
                self.portable_project.raise_()
                return
            self.retain_portable_application(self.portable_project)
            self.portable_project.deleteLater()
            self.portable_project = None
        try:
            job = self.job()
        except (ValueError, OSError, TypeError):
            job = None  # Opening a package does not require a filled source form.
        self.portable_project = PortableProjectDialog(job, self, frame=self.frame.value(), selection=self.asset_selection)
        self.portable_project.loaded.connect(self.apply_portable_project)
        if self.portable_application_records:
            lines = ['Recent committed Apply snapshots retained during this workspace session:']
            for record in self.portable_application_records:
                report = record['report']
                lines.append(f"Frame {report.get('preview_frame')} · sent {report.get('emitted')} · accepted {report.get('accepted')} · correction snapshot {report.get('resolved_correction_snapshot') or 'none'}")
            self.portable_project.details.setPlainText('\n'.join(lines))
        self.portable_project.show()

    def retain_portable_application(self, dialog: PortableProjectDialog):
        job, report = dialog.last_resolved_job, dialog.application_report
        if job is None or not report:
            return
        # UI reports are bounded at64KiB; keep detached primitive records and
        # immutable jobs, never Qt widgets. Files remain owned in explicit scratch.
        detached = json.loads(json.dumps(report, allow_nan=False))
        key = (detached.get('project_identity'), detached.get('resolved_correction_snapshot'),
               detached.get('preview_frame'), detached.get('output_override'))
        self.portable_application_records = [record for record in self.portable_application_records
                                             if record['key'] != key]
        self.portable_application_records.append({'key': key, 'job': job, 'report': detached})
        self.portable_application_records = self.portable_application_records[-8:]

    def apply_portable_project(self, job, frame, output_override, model_report):
        def acknowledge(accepted, message):
            if self.portable_project is not None:
                self.portable_project.acknowledge_application(accepted, message)
        if self.closing:
            acknowledge(False, 'Workspace is closing; project was not applied to the draft.')
            return
        if self.editor is not None and self.editor.canvas.dirty:
            acknowledge(False, 'Correction draft changed; commit or clear it, then apply the verified project again.')
            self.error('Commit or Clear the correction draft before applying another project.')
            return
        if (not isinstance(job, JobRequest) or type(frame) is not int or frame < 0
                or not isinstance(output_override, str) or not output_override or not isinstance(model_report, dict)):
            acknowledge(False, 'Project application was rejected before changing draft settings.')
            self.error('Portable project application requires a verified job and explicit output destination.')
            return
        if frame > self.frame.maximum():
            message = f'Preview frame {frame} exceeds the desktop limit {self.frame.maximum()}; the recorded frame was preserved and no draft settings changed.'
            acknowledge(False, message)
            self.error(message)
            return
        try:
            model_description = json.dumps(model_report, indent=2, allow_nan=False)
            if len(model_description.encode('utf-8')) > 64 * 1024:
                raise ValueError('Portable project application summary exceeds the 64 KiB display bound.')
            verified_job = job_from_wire(asdict(job))
        except (ValueError, TypeError) as exc:
            acknowledge(False, 'Project application was rejected before changing draft settings: ' + str(exc))
            self.error(exc)
            return
        self.populate_job(verified_job, frame=frame, output_parent=output_override)
        self.statusBar().showMessage('Portable project settings loaded; no job submitted. Review referenced model assets in Local assets before rendering.')
        self.statusBar().setToolTip(model_description)
        acknowledge(True, 'Portable project settings loaded with explicit output/scratch; no job or model switch started.')

    def populate_job(self, job: JobRequest, *, frame=0, output_parent: str | None = None):
        if self.moving_reference is not None:
            self.clear_reference()
        self.source.setText(job.input_path)
        self.output.setText(output_parent if output_parent is not None else job.output_dir)
        self.first_hint.setText(job.initial_hint_path or '')
        self.full_hint.setText(job.hint_video_path or '')
        self.profile.setChecked(job.profile_stages)
        config, writer = pretty_settings(job)
        self.config.setPlainText(config)
        self.writer.setPlainText(writer)
        settings, _ = job.configurations()
        self.limit.blockSignals(True)
        self.start.blockSignals(True)
        self.limit.setValue(max(0, settings.num_frames))
        self.start.setCurrentText(settings.start_mode)
        self.limit.blockSignals(False)
        self.start.blockSignals(False)
        self.frame.setValue(frame)
        self.view.setCurrentText('Source')
        self.preview_serial = -1
        self.pixmap = None
        self.image.setText('Settings loaded. Select Preview to decode this source.')
        self.statusBar().showMessage('Settings loaded; no job was submitted')

    def load_queue_settings(self, job: JobRequest):
        self.populate_job(job, output_parent=str(Path(job.output_dir).parent))

    def submit_recovery(self, payload):
        self.progressive.suspend_for_export()
        try:
            if self.closing or self.benchmark.busy or self.pending_queue_root is not None:
                raise ValueError('Wait for current benchmark/queue/shutdown before recovery submission.')
            if self.bridge.journal_workspace is None:
                raise ValueError('Create/activate a durable queue workspace before submitting recovery.')
            if len(self.jobs) >= 128:
                raise ValueError('Clear finished rows before submitting more work.')
            normalized = decode_frame(request_frame('validation', 'recover', payload))['payload']
            request_id = self.bridge.send('recover', normalized)
            self.jobs[request_id] = {'action': 'recover', 'payload': normalized, 'status': 'sending', 'event': None}
            self.recovery.track(request_id)
            self.refresh_table()
        except (OSError, ValueError, RuntimeError, queue.Full) as exc:
            self.recovery.status.setText(str(exc))

    def prepare_recovery(self):
        try:
            if self.closing or self.recovery.io_busy or any(state['status'] not in TERMINAL for state in self.jobs.values()):
                raise ValueError('Wait for active jobs and recovery I/O before replacing the recovery form.')
            job = self.job()
            config, _ = job.configurations()
            if config.output_format != 'exr' or config.auto_vram:
                raise ValueError('Choose EXR and turn off automatic scheduling in job settings before preparing recovery.')
            parent = QFileDialog.getExistingDirectory(self, 'Save immutable recovery recipe in a separate directory')
            if parent:
                self.recovery.prepare_job(Path(parent), job, weights=self.asset_selection.weights if self.asset_selection else None)
        except (ValueError, TypeError, OSError) as exc:
            self.error(exc)

    def activate_assets(self, selection: AssetSelection):
        try:
            if self.closing or self.benchmark.busy or self.pending_queue_root is not None:
                raise ValueError('Wait for benchmark/queue/shutdown before changing assets.')
            if not self.bridge.done.is_set():
                raise ValueError('Stop worker and wait for shutdown before using different assets.')
            self.poll()
            if self.closing or self.benchmark.busy or self.pending_queue_root is not None or not self.bridge.done.is_set():
                raise ValueError('Processing ownership changed; wait before changing assets.')
            if self.bridge.failure is not None:
                raise ValueError('Previous worker cleanup failed: ' + self.bridge.failure)
            if self.benchmark.task is not None and self.benchmark.task.snapshot()['cleanup_error']:
                raise ValueError('Previous benchmark cleanup is unverified; cannot start another owner.')
            if self.bridge.journal_workspace is not None:
                from .queue_snapshot import separate
                separate(self.bridge.journal_workspace, [selection.root, selection.weights])
            self.bridge = DesktopBridge(partial(WorkerClient, asset_root=selection.root, weights=selection.weights),
                                        journal_workspace=self.bridge.journal_workspace)
            self.asset_selection = selection
            self.refresh_diagnostic_context()
            self.assets.status.setText('Selected for future jobs. No model loaded; CUDA inference/quality remains unverified.')
            self.backend.setText('Local assets selected: ' + str(selection.root) + ' · CUDA required · No model loaded')
        except (ValueError, OSError) as exc:
            self.assets.status.setText(str(exc))

    def cancel_recovery(self):
        request_id = self.recovery.current_request
        if request_id in self.jobs and self.jobs[request_id]['status'] not in TERMINAL:
            try:
                self.bridge.send('cancel', {'target': request_id, 'reason': 'Cancelled from recovery controls'})
                self.recovery.status.setText('Cancellation requested; waiting for terminal record.')
            except (RuntimeError, queue.Full) as exc:
                self.recovery.status.setText(str(exc))

    def activate_queue(self, root: Path):
        if self.asset_selection is not None:
            from .queue_snapshot import separate
            try:
                separate(root, [self.asset_selection.root, self.asset_selection.weights])
            except (ValueError, OSError) as exc:
                self.queue_panel.status.setText(str(exc))
                return
        if self.closing or self.benchmark.busy or self.pending_queue_root is not None:
            self.queue_panel.status.setText('Wait for current processing/shutdown before changing queue workspace.')
            return
        if self.benchmark.task is not None and self.benchmark.task.snapshot()['cleanup_error']:
            self.queue_panel.status.setText('Previous benchmark cleanup is unverified; cannot start another owner.')
            return
        if any(state['status'] not in TERMINAL for state in self.jobs.values()):
            self.queue_panel.status.setText('Finish/cancel current jobs before activating another queue.')
            return
        self.pending_queue_root = root
        self.bridge.stop()
        self.queue_panel.status.setText('Waiting for existing worker/queue owner to close…')

    def save_project(self):
        try:
            job = self.job()
            path, _ = QFileDialog.getSaveFileName(self, 'Save desktop project', '', 'JSON (*.json)')
            if path:
                save_project(Path(path), job, self.frame.value())
                self.statusBar().showMessage('Project settings saved')
        except (ValueError, OSError, TypeError) as exc:
            self.error(exc)

    def open_editor(self):
        self._open_editor()

    def open_editor_document(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open authored correction document', '', 'JSON (*.json)')
        if path:
            self._open_editor(Path(path))

    def _open_editor(self, existing: Path | None = None):
        if self.closing:
            return
        if self.editor is not None and (not self.editor.service.done.is_set() or self.editor.canvas.dirty):
            self.editor.show()
            self.error('Close the current correction editor before binding another document/source.')
            return
        try:
            job = self.job(preview_only=True)
            config, _ = job.configurations()
            existing = existing or (Path(config.corrections_path) if config.corrections_path is not None else None)
            parent = QFileDialog.getExistingDirectory(self, 'Correction-project parent (outside source)', self.output.text())
            if not parent:
                return
            previous = self.editor
            self.editor = CorrectionDialog(job, Path(parent), self, existing=existing)
            if previous is not None:
                # Actor is stopped and drafts are clear before native retirement.
                previous.close()
                previous.deleteLater()
            self.editor.saved.connect(self.attach_corrections)
            self.editor.show()
        except (ValueError, OSError, RuntimeError) as exc:
            self.error(exc)

    def attach_corrections(self, bound: JobRequest, path: str):
        try:
            current = self.job(preview_only=True)
            original, _ = bound.configurations()
            config, _ = current.configurations()
            keys = ('num_frames', 'start_mode', 'start_frame', 'seed', 'frame_dir_fps')
            if current.input_path != bound.input_path or any(getattr(config, key) != getattr(original, key) for key in keys):
                raise ValueError('Saved correction source/timeline differs from current workspace; rebind before attaching.')
            self.update_settings({'corrections_path': path})
            self.statusBar().showMessage('Saved corrections attached to future jobs; queued snapshots remain unchanged.')
        except (ValueError, OSError, RuntimeError) as exc:
            self.error(exc)

    def stop_worker(self):
        self.bridge.stop()
        self.statusBar().showMessage('Stopping worker; active work may leave partial outputs')

    def request_resources(self):
        try:
            if self.closing or self.benchmark.busy or self.pending_queue_root is not None:
                raise ValueError('Wait for benchmark/queue/shutdown before sampling the worker.')
            if self.resource_request is not None:
                raise ValueError('Wait for the outstanding worker snapshot or stop its connection explicitly.')
            self.resource_request = self.bridge.send('status', {})
            self.resources.status.setText('Waiting for worker snapshot; no render submitted.')
        except (ValueError, RuntimeError, queue.Full) as exc:
            self.resources.status.setText(str(exc))

    def new_worker(self):
        if self.closing or self.benchmark.busy or self.pending_queue_root is not None:
            self.error('Wait for benchmark/shutdown completion before starting another worker.')
            return
        if self.benchmark.task is not None and self.benchmark.task.snapshot()['cleanup_error']:
            self.error('Previous benchmark cleanup is unverified; another processing owner cannot start.')
            return
        if not self.bridge.done.is_set():
            self.error('Stop the current worker and wait for shutdown before starting another.')
            return
        if self.bridge.cleanup_error is not None:
            self.error('Previous worker cleanup is unverified; another processing owner cannot start: ' + self.bridge.cleanup_error)
            return
        self.poll()
        if not self.bridge.done.is_set() or self.pending_queue_root is not None or self.benchmark.busy:
            self.error('Processing ownership changed while refreshing; wait for the current owner.')
            return
        if self.bridge.cleanup_error is not None:
            self.error('Previous worker cleanup is unverified; another processing owner cannot start: ' + self.bridge.cleanup_error)
            return
        self.bridge = DesktopBridge(self.bridge.factory, journal_workspace=self.bridge.journal_workspace)
        self.statusBar().showMessage('New worker connection; previous jobs will not be resubmitted')

    def start_benchmark(self, manifest: Path, parent: Path):
        if self.closing or self.benchmark.busy or self.benchmark.io_busy or self.pending_queue_root is not None:
            self.benchmark.status.setText('Processing/shutdown is already in progress.')
            return
        if any(state['status'] not in TERMINAL for state in self.jobs.values()):
            self.benchmark.status.setText('Finish or cancel render jobs before benchmarking.')
            return
        if self.benchmark.task is not None and self.benchmark.task.snapshot()['cleanup_error']:
            self.benchmark.status.setText('Previous benchmark cleanup is unverified; cannot start another owner.')
            return
        self.pending_benchmark = (manifest.resolve(), parent.resolve())
        self.benchmark.waiting = True
        self.benchmark.status.setText('Stopping resident render worker before benchmarking…')
        self.bridge.stop()

    def cancel_benchmark(self):
        self.pending_benchmark = None
        self.benchmark.waiting = False
        if self.benchmark.task is not None and not self.benchmark.task.done.is_set():
            self.benchmark.task.cancel()
        self.benchmark.status.setText('Benchmark cancellation requested; waiting for owned process cleanup.')

    def poll(self):
        self.readiness.poll()
        self.diagnostic_ranges.poll()
        self.benchmark.refresh()
        self.assets.poll()
        self.resources.poll()
        self.ofx_cache.poll()
        self.cache_panel.poll()
        self.queue_panel.poll()
        self.recovery.poll()
        changed = False
        for event in self.bridge.poll():
            if self.progressive.observe(event):
                continue
            if self.resource_request is not None and event.get('request_id') == self.resource_request:
                self.resources.observe(event)
                self.resource_request = None
            self.recovery.observe(event)
            self.readiness.observe(event)
            request_id = event.get('request_id')
            if event['kind'] == 'connected':
                self.backend.setText(f"Worker PID {event['payload']['pid']} · CUDA required · Live GPU memory unavailable")
            elif event['kind'] == 'queue_opened':
                self.queue_label.setText('Durable queue: ' + event['payload']['workspace'])
                self.queue_panel.active.setText('Active queue: ' + event['payload']['workspace'])
            if request_id in self.jobs:
                state = self.jobs[request_id]
                if state['status'] not in TERMINAL:
                    state['event'] = event
                    state['status'] = event['kind']
                    if state.get('action') == 'recover' and event['kind'] == 'recorded':
                        state['payload']['batch'] = event['payload']['batch']
                    payload = event.get('payload', {})
                    detail = payload.get('error') or payload.get('stage') or ''
                    if isinstance(detail, dict):
                        detail = detail.get('message', str(detail))
                    self.statusBar().showMessage(f"{self.row_paths(state)[0]} · {event['kind']} · {detail}")
                    changed = True
        if self.bridge.done.is_set():
            if self.resource_request is not None:
                self.resources.status.setText('Worker stopped before snapshot acknowledgement: ' + str(self.bridge.failure or 'connection closed'))
                self.resource_request = None
            if self.bridge.journal_workspace is not None and self.bridge.failure is not None:
                self.queue_label.setText('Queue stopped/unavailable: ' + str(self.bridge.failure))
                self.queue_panel.active.setText('Queue stopped/unavailable: ' + str(self.bridge.failure))
            elif self.bridge.journal_workspace is not None:
                self.queue_label.setText('Durable queue stopped: ' + str(self.bridge.journal_workspace))
                self.queue_panel.active.setText('Queue stopped; no current writer: ' + str(self.bridge.journal_workspace))
            for state in self.jobs.values():
                if state['status'] not in TERMINAL:
                    state['status'] = 'lost'
                    state['event'] = {'kind': 'lost', 'payload': {'error': self.bridge.failure or 'Connection closed before terminal acknowledgement',
                                                                 'outputs_may_be_partial': True}}
                    changed = True
            if self.readiness.submitted_id is not None:
                self.readiness.observe({'request_id': self.readiness.submitted_id, 'kind': 'lost',
                                        'payload': {'error': self.bridge.failure or 'Worker connection closed'}})
            self.backend.setText('Worker stopped · ' + (self.bridge.failure or 'Clean shutdown'))
            if self.pending_queue_root is not None:
                root, self.pending_queue_root = self.pending_queue_root, None
                if self.bridge.failure is None and not self.closing:
                    self.bridge = DesktopBridge(self.bridge.factory, journal_workspace=root)
                    self.queue_label.setText('Opening durable queue: ' + str(root))
                else:
                    self.queue_panel.status.setText('Queue activation refused: ' + str(self.bridge.failure or 'application closing'))
            if self.pending_benchmark is not None:
                request, self.pending_benchmark = self.pending_benchmark, None
                self.benchmark.waiting = False
                if self.bridge.failure is not None:
                    self.benchmark.status.setText('Render worker shutdown was not clean: ' + self.bridge.failure)
                else:
                    self.benchmark.begin(*request)
            if self.closing:
                self.close()
        if changed:
            self.refresh_table()
            self.show_record()
        self.playback.tick()
        try:
            serial, frame, error = self.loader.results.get_nowait()
        except queue.Empty:
            return
        playback_valid = self.playback.complete_loader(serial, error)
        # Preparation metadata, including retired generations, never contains pixels.
        if self.playback.accept_preparation(serial, frame, error) or isinstance(frame, PreparedSequence):
            return
        if serial != self.preview_serial or playback_valid is False:
            return
        if error is not None:
            self.preview_note.setText(error)
            if self.moving_reference is not None:
                self.image.setText('Comparison unavailable')
                self.reference_image.setText('Comparison unavailable')
        elif frame is not None:
            comparison_pair = isinstance(frame, PreviewPair)
            if isinstance(frame, PreviewPair):
                pixels = frame.a.pixels
                image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
                self.reference = QPixmap.fromImage(image)
                self.reference_image.setPixmap(self.reference)
                self.reference_note.setText('A · ' + self.reference_description.rsplit(' · ', 1)[0] + '\n' + frame.a.note)
                self.reference_note.setToolTip(self.reference_description)
                frame = frame.b
            pixels = frame.pixels
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
            self.pixmap = QPixmap.fromImage(image)
            self.image.setPixmap(self.pixmap)
            self.preview_note.setText(f'B · {self.preview_description.rsplit(" · ", 1)[0]}\n{frame.width}×{frame.height} · {frame.count} frames · {frame.note}')
            self.preview_note.setToolTip(self.preview_description)
            if comparison_pair:
                self._comparison_recovered()

    def closeEvent(self, event: QCloseEvent):
        if self.editor is not None and self.editor.canvas.dirty:
            self.editor.show()
            self.editor.status.setText('Commit or Clear draft before closing the workspace.')
            self.statusBar().showMessage('Correction draft preserved. Commit or Clear it before closing.')
            event.ignore()
            return
        self.closing = True
        self.playback.pause()
        self.loader.stopping.set()
        progressive_done = self.progressive.shutdown()
        self.bridge.stop()
        self.cancel_benchmark()
        self.benchmark.dossier_panel.close()
        self.readiness.close()
        self.diagnostic_ranges.close()
        self.ofx_cache.close()
        self.cache_panel.close()
        self.queue_panel.close()
        self.assets.close()
        if self.editor is not None:
            self.editor.close()
        portable_done = self.portable_project is None or self.portable_project.shutdown()
        if portable_done and self.portable_project is not None:
            self.retain_portable_application(self.portable_project)
        editor_done = self.editor is None or self.editor.service.done.is_set()
        if self.bridge.done.is_set() and not self.readiness.io_busy and not self.diagnostic_ranges.io_busy and not self.benchmark.busy and not self.benchmark.io_busy and not self.queue_panel.io_busy and not self.recovery.io_busy and not self.assets.io_busy and not self.resources.io_busy and not self.ofx_cache.io_busy and not self.cache_panel.io_busy and editor_done and portable_done and progressive_done and not self.loader.thread.is_alive():
            self.timer.stop()
            event.accept()
        else:
            self.statusBar().showMessage('Waiting for owned worker, preview, benchmark or report I/O cleanup…')
            event.ignore()
