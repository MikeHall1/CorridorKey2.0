"""Explicit portable-project inventory, publication and relinking on one owner."""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .asset_inventory import AssetSelection
from .control import CancellationToken
from .desktop_state import DESKTOP_MAX_PREVIEW_FRAME
from .session import JobRequest
from .worker_protocol import job_to_wire

if TYPE_CHECKING:
    from .portable_project import ProjectInspection, ProjectPlan

def _text(value, limit=1024):
    if value is None:
        return None
    if not isinstance(value, (str, Path, int, float, bool)):
        return '<unsupported text field>'
    raw = str(value).encode('utf-8')
    suffix = '… [display truncated]'
    return raw.decode('utf-8') if len(raw) <= limit else raw[:max(0, limit - len(suffix.encode()))].decode('utf-8', errors='ignore') + suffix


def _json_bytes(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


PAGE_SIZE = 16


def compact_project_report(report: dict, *, page: int = 0) -> dict:
    """Worker-side <=64KiB presentation; never traverse file lists or emit them.

    Full original wire/provenance remains in the immutable core plan/inspection
    and sealed manifest. Only bounded dependency groups appear in Qt.
    """
    import hashlib

    dependencies = report.get('dependencies', [])
    page_count = max(1, (len(dependencies) + PAGE_SIZE - 1) // PAGE_SIZE)
    if type(page) is not int or not 0 <= page < page_count:
        raise ValueError('Portable inventory page is outside the frozen snapshot.')
    models = report.get('model_references', {})
    entries = models.get('entries', [])
    compact_models = {'asset_root': _text(models.get('asset_root'), 4096), 'weights': _text(models.get('weights'), 4096),
                      'scope': _text(models.get('scope'), 512),
                      'entries': [{'id': _text(entry.get('id'), 128), 'path': _text(entry.get('path'), 1024),
                                   'status': _text(entry.get('status'), 256), 'bytes': entry.get('bytes')}
                                  for entry in entries[:4]], 'omitted_entries': max(0, len(entries) - 4)}
    requested = report.get('requested_job')
    settings = []
    if requested is not None:
        from .portable_paths import archive_job

        archived = archive_job(requested)
        config, writer = archived.config, archived.writer
        settings = [f"Captured source start {config.start_frame or 0} · count {config.num_frames if config.num_frames is not None else 'all remaining'} · sequence FPS {config.frame_dir_fps} · start mode {config.start_mode}",
                    f"Captured source colour {config.source_colour_profile} · linear {config.frame_dir_linear} · alpha {config.frame_dir_alpha} · model scale {config.model_scale}",
                    f"Requested output {config.output_format} / {', '.join(sorted(config.outputs))} · EXR {config.exr_dtype} · stage2 batch {config.stage2_batch}",
                    f"Recorded output {archived.original_wire['output_dir']} and scratch {config.temp_dir} remain provenance; explicit new parents are required on Apply. Configured FFmpeg {writer.ffmpeg_bin} is not collected."]
    compact = {'dependencies': [], 'page': page, 'page_count': page_count, 'page_size': PAGE_SIZE, 'dependency_count': len(dependencies), 'omitted_dependencies': len(dependencies),
               'total_file_count': sum(row.get('file_count', len(row.get('files', []))) for row in dependencies),
               'model_references': compact_models, 'preview_frame': report.get('preview_frame', 0), 'ready': report.get('ready') is True,
               'read_bytes': report.get('read_bytes', 0), 'copy_bytes': report.get('copy_bytes', 0),
               'collection_count': len(report.get('collect_ids', [])),
               'archive_path_flavour': _text(report.get('archive_path_flavour'), 32),
               'foreign_paths': report.get('foreign_paths') is True,
               'portability_scope': _text(report.get('portability_scope'), 1024),
               'correction_identity': _text(report.get('correction_identity'), 128),
               'requested_job_identity': None if requested is None else hashlib.sha256(_json_bytes(requested)).hexdigest(),
               'requested_settings_lines': [_text(line, 2048) for line in settings],
               'scope': 'Bounded presentation only; full file lists, requested wire and provenance remain in the sealed manifest/core inspection.'}
    remaining = 64 * 1024 - len(_json_bytes(compact)) - 1024
    for row in dependencies[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]:
        item: dict[str, object] = {key: _text(row.get(key), limit) for key, limit in
                (('id', 128), ('role', 128), ('path', 1024), ('kind', 64), ('status', 256))}
        item.update(file_count=row.get('file_count', len(row.get('files', []))), bytes=row.get('bytes', 0),
                    required=row.get('required') is True, collected=row.get('collected') is True,
                    dependency_count=len(row.get('depends_on', [])))
        cost = len(_json_bytes(item)) + 1
        if cost > remaining:
            raise ValueError('Portable page exceeds its 64 KiB presentation bound.')
        compact['dependencies'].append(item)
        remaining -= cost
    compact['omitted_dependencies'] = len(dependencies) - len(compact['dependencies'])
    if len(_json_bytes(compact)) > 64 * 1024:
        raise ValueError('Portable presentation exceeds the 64 KiB display bound.')
    return compact


def _presentation(report):
    return _json_bytes(report), compact_project_report(report)


def _loaded_context(inspection, output, scratch):
    import hashlib

    report = compact_project_report(inspection.report)
    context = {'model_references': report['model_references'],
               'compatibility': 'Model/backend compatibility NOT VERIFIED; resolve local assets explicitly before rendering.',
               'archive_path_flavour': report['archive_path_flavour'], 'foreign_paths': report['foreign_paths'],
               'portability_scope': report['portability_scope'],
               'project_path': _text(inspection.package_path, 4096),
               'project_identity': hashlib.sha256(inspection.manifest_json).hexdigest(),
               'requested_job_identity': report['requested_job_identity'],
               'correction_identity': report['correction_identity'],
               'dependency_count': report['dependency_count'], 'file_count': report['total_file_count'],
               'output_override': _text(output, 1024), 'scratch_override': _text(scratch, 1024)}
    import json
    while context['model_references']['entries'] and len(json.dumps(context, allow_nan=False).encode()) > 64 * 1024:
        context['model_references']['entries'].pop()
        context['model_references']['omitted_entries'] += 1
    if len(json.dumps(context, allow_nan=False).encode()) > 64 * 1024:
        raise ValueError('Portable application context exceeds the 64 KiB signal bound.')
    return context, report['preview_frame']



class PortableProjectDialog(QDialog):
    loaded = Signal(object, object, str, object)

    def __init__(self, job: JobRequest | None, parent=None, *, frame: int = 0,
                 selection: AssetSelection | None = None):
        if job is not None and not isinstance(job, JobRequest):
            raise TypeError('A captured JobRequest instance is required, not a dataclass class.')
        if selection is not None and not isinstance(selection, AssetSelection):
            raise TypeError('Local assets must be captured as an AssetSelection instance.')
        super().__init__(parent)
        if type(frame) is not int or not 0 <= frame <= 2**63 - 1:
            raise ValueError('Preview frame must be a nonnegative integer.')
        self.job, self.frame, self.selection = job, frame, selection
        self.original_wire = job_to_wire(job) if job is not None else None
        self.setWindowTitle('CorridorKey · Portable project')
        self.resize(1060, 760)
        self.io_thread: threading.Thread | None = None
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self.cancellation = CancellationToken()
        self.closing = False
        self.operation = ''
        self.inventory: dict | None = None
        self.report: dict | None = None
        self.plan: ProjectPlan | None = None
        self.inspection: ProjectInspection | None = None
        self.inspection_ready = False
        self.inspection_frame: int | None = None
        self.package: Path | None = None
        self.relinks: dict[str, Path] = {}
        self.unverified_relinks: set[str] = set()
        self.mode = 'job'
        self.full_report: bytes | None = None
        self.page = 0
        self.page_count = 1
        self.collect_ids: set[str] = set()
        self.published_path: Path | None = None
        self.last_resolved: JobRequest | None = None
        self.last_resolved_context: dict | None = None
        self.application_report: dict | None = None
        self._application_ack: tuple[bool, str] | None = None
        self._emitting_application = False
        root = QVBoxLayout(self)
        intro = QLabel('Inventory reads bounded metadata only. Verify explicitly reads content within the chosen budget. Collection defaults to none: reference-only projects do not copy media or models. Review selected dependencies and budgets before publishing to a fresh child directory. Opening or applying a project never queues a job, loads a model, or changes the active model owner. Original files and recorded settings remain protected. Foreign Windows/POSIX packages require every render dependency collected; partial/reference-only foreign packages remain unsupported. Archived model references never activate automatically.')
        intro.setWordWrap(True)
        root.addWidget(intro)
        row = QHBoxLayout()
        self.inventory_button = self._button('Inventory captured draft', self.inventory_draft, row)
        self.package_path = QLineEdit()
        self.package_path.setPlaceholderText('Existing portable package directory')
        row.addWidget(self.package_path, 1)
        self.choose_package_button = self._button('Choose package…', self.choose_package, row)
        self.open_button = self._button('Preview package inventory', self.inventory_package, row)
        root.addLayout(row)
        row = QHBoxLayout()
        self.read_budget = QSpinBox()
        self.copy_budget = QSpinBox()
        for label, widget, default in [('Maximum verification read MiB', self.read_budget, 256),
                                        ('Maximum selected copy MiB', self.copy_budget, 0)]:
            widget.setRange(0, 65536)
            widget.setValue(default)
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            widget.valueChanged.connect(self.invalidate_verification)
        self.verify_button = self._button('Verify / prepare displayed choices', self.verify, row)
        self.cancel_button = self._button('Cancel operation', self.cancel, row)
        root.addLayout(row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Collect', 'Dependency', 'Current / staged path', 'Status', 'Files', 'Bytes'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self.collection_changed)
        root.addWidget(self.table, 2)
        row = QHBoxLayout()
        self.previous_page = self._button('Previous groups', lambda: self.request_page(self.page - 1), row)
        self.page_label = QLabel('No dependency inventory')
        row.addWidget(self.page_label, 1)
        self.next_page = self._button('Next groups', lambda: self.request_page(self.page + 1), row)
        root.addLayout(row)
        row = QHBoxLayout()
        self.relink_path = QLineEdit()
        self.relink_path.setPlaceholderText('Deliberate replacement for the selected dependency ID')
        row.addWidget(self.relink_path, 1)
        self.relink_file_button = self._button('Relink file…', lambda: self.choose_relink(False), row)
        self.relink_directory_button = self._button('Relink directory…', lambda: self.choose_relink(True), row)
        self.stage_relink_button = self._button('Stage selected relink', self.stage_relink, row)
        root.addLayout(row)
        row = QHBoxLayout()
        self.destination = QLineEdit()
        self.destination.setPlaceholderText('Existing publication parent; a fresh child will be created')
        row.addWidget(self.destination, 1)
        self.choose_destination_button = self._button('Publication parent…', lambda: self.choose_directory(self.destination), row)
        self.publish_button = self._button('Publish verified fresh package', self.publish, row)
        root.addLayout(row)
        row = QHBoxLayout()
        self.output = QLineEdit()
        self.output.setPlaceholderText('Explicit output parent for loaded draft')
        row.addWidget(self.output, 1)
        self.choose_output_button = self._button('Output parent…', lambda: self.choose_directory(self.output), row)
        self.scratch = QLineEdit()
        self.scratch.setPlaceholderText('Explicit scratch parent for loaded draft')
        row.addWidget(self.scratch, 1)
        self.choose_scratch_button = self._button('Scratch parent…', lambda: self.choose_directory(self.scratch), row)
        self.apply_button = self._button('Apply verified project to draft', self.apply, row)
        root.addLayout(row)
        self.models = QLabel('Model references are preserved separately; checkpoint/backend compatibility is NOT VERIFIED.')
        self.models.setWordWrap(True)
        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setMinimumHeight(36)
        model_scroll.setMaximumHeight(64)
        model_scroll.setWidget(self.models)
        root.addWidget(model_scroll)
        self.status = QLabel('No inventory or verification performed. Choose Inventory captured draft or preview an existing package.')
        self.status.setWordWrap(True)
        status_scroll = QScrollArea()
        status_scroll.setWidgetResizable(True)
        status_scroll.setMinimumHeight(30)
        status_scroll.setMaximumHeight(64)
        status_scroll.setWidget(self.status)
        root.addWidget(status_scroll)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(512)
        root.addWidget(self.details, 1)
        for widget in (self.output, self.scratch, self.destination):
            widget.textChanged.connect(self._buttons)
        self.package_path.textChanged.connect(self.package_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self._buttons()

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @property
    def busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    def _buttons(self, *_args):
        idle = not self.busy and not self.closing
        self.inventory_button.setEnabled(idle and self.job is not None)
        self.open_button.setEnabled(idle and bool(self.package_path.text().strip()))
        self.verify_button.setEnabled(idle and self.inventory is not None)
        self.publish_button.setEnabled(idle and self.plan is not None and bool(self.destination.text().strip()))
        self.apply_button.setEnabled(idle and self.inspection is not None and self.inspection_ready and self._supported_frame(self.inspection_frame) and bool(self.output.text().strip()) and bool(self.scratch.text().strip()))
        self.cancel_button.setEnabled(self.busy and not self.closing)
        self.previous_page.setEnabled(idle and self.full_report is not None and self.page > 0)
        self.next_page.setEnabled(idle and self.full_report is not None and self.page + 1 < self.page_count)
        for widget in (self.table, self.read_budget, self.copy_budget, self.package_path, self.choose_package_button,
                       self.destination, self.choose_destination_button, self.output, self.choose_output_button,
                       self.scratch, self.choose_scratch_button):
            widget.setEnabled(idle)
        for widget in (self.relink_path, self.relink_file_button, self.relink_directory_button, self.stage_relink_button):
            widget.setEnabled(idle and self.mode == 'package' and self.inventory is not None)

    def invalidate_verification(self, *_args):
        self.plan = None
        self.inspection = None
        self.inspection_ready = False
        self.inspection_frame = None
        if self.report is not None and not self.busy and not self.closing:
            self.status.setText('Choices changed; verify again before publication or draft application.')
        self._buttons()

    def collection_changed(self, item):
        if item.column() == 0 and self.mode == 'job':
            identifier = item.data(Qt.ItemDataRole.UserRole)
            if item.checkState() == Qt.CheckState.Checked:
                self.collect_ids.add(identifier)
            else:
                self.collect_ids.discard(identifier)
            self.invalidate_verification()

    def package_changed(self, *_args):
        self.invalidate_verification()
        if self.mode == 'package':
            self.inventory = None
            self.package = None
            self.relinks.clear()
            self.unverified_relinks.clear()
            self.full_report = None
        self._buttons()

    def _begin(self, kind, function):
        if self.busy or self.closing:
            self.status.setText('Wait for the owned operation or shutdown to finish.')
            return False
        self.cancellation = CancellationToken()
        self.operation = kind

        def work():
            try:
                value, error = function(self.cancellation.check), None
            except Exception as exc:
                logging.getLogger(__name__).exception('Portable project operation stopped')
                notes = '\n'.join(str(note)[:1024] for note in getattr(exc, '__notes__', [])[:3])
                value, error = None, (f'{type(exc).__name__}: {exc}' + ('\n' + notes if notes else ''))[:4096]
            self.results.put((kind, value, error))

        self.io_thread = threading.Thread(target=work, name='corridorkey-portable-project', daemon=False)
        self.io_thread.start()
        self.status.setText('Working: ' + kind + '. Cancellation waits for the owned file operation to retire.')
        self._buttons()
        return True

    def _selection(self):
        return {'asset_root': self.selection.root if self.selection else None,
                'weights': self.selection.weights if self.selection else None}

    def inventory_draft(self):
        if self.job is None or self.busy or self.closing:
            return
        from .portable_project import inventory_project
        self.invalidate_verification()
        job = self.job
        self._begin('draft_inventory', lambda check: _presentation(inventory_project(job, self.frame, check=check, **self._selection())))

    def choose_package(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose portable project package')
        if path:
            self.package_path.setText(path)
            self.package_changed()

    def inventory_package(self):
        if self.busy or self.closing or not self.package_path.text().strip():
            return
        from .portable_project import inventory_package
        package = Path(self.package_path.text().strip())
        self.invalidate_verification()
        self._begin('package_inventory', lambda check: (package, *_presentation(inventory_package(package, check=check))))

    def selected_collect_ids(self):
        return tuple(sorted(self.collect_ids))

    def request_page(self, page):
        if self.busy or self.closing or self.full_report is None or not 0 <= page < self.page_count:
            return
        snapshot = self.full_report

        def format_page(check):
            import json
            check()
            report = compact_project_report(json.loads(snapshot), page=page)
            check()
            return report
        self._begin('page', format_page)

    def verify(self):
        if self.busy or self.closing or self.inventory is None:
            return
        from .portable_project import inspect_project, plan_project
        read_budget, copy_budget = self.read_budget.value() * 1024**2, self.copy_budget.value() * 1024**2
        self.invalidate_verification()
        if self.mode == 'job' and self.job is not None:
            collect, job = self.selected_collect_ids(), self.job
            def prepare_plan(check):
                plan = plan_project(job, self.frame, collect_ids=collect, read_budget=read_budget, copy_budget=copy_budget, check=check, **self._selection())
                return plan, *_presentation(plan.summary)
            self._begin('plan', prepare_plan)
        elif self.package is not None:
            package, relinks = self.package, dict(self.relinks)
            def inspect(check):
                inspection = inspect_project(package, relinks=relinks, read_budget=read_budget, check=check)
                return inspection, *_presentation(inspection.report)
            self._begin('inspect', inspect)

    def choose_directory(self, field):
        path = QFileDialog.getExistingDirectory(self, 'Choose explicit local parent directory')
        if path:
            field.setText(path)

    def publish(self):
        if self.busy or self.closing or self.plan is None or not self.destination.text().strip():
            return
        from .portable_project import publish_project
        plan, parent = self.plan, Path(self.destination.text().strip())
        self._begin('publish', lambda check: publish_project(plan, parent, check=check))

    def choose_relink(self, directory):
        path = (QFileDialog.getExistingDirectory(self, 'Select the whole replacement dependency directory') if directory else
                QFileDialog.getOpenFileName(self, 'Select the replacement dependency file')[0])
        if path:
            self.relink_path.setText(path)
            self.stage_relink()

    def stage_relink(self):
        if self.busy or self.closing or self.mode != 'package' or not self.relink_path.text().strip():
            return
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        if item is None:
            self.status.setText('Select a dependency row before staging a replacement path.')
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        self.relinks[key] = Path(self.relink_path.text().strip())
        self.unverified_relinks.add(key)
        self.invalidate_verification()
        self.table.setItem(row, 2, QTableWidgetItem(str(self.relinks[key])))
        self.table.setItem(row, 3, QTableWidgetItem('Staged relink; identity NOT VERIFIED'))
        self.status.setText('Replacement staged for the selected dependency. Verify before applying; paths alone do not establish compatibility.')

    @staticmethod
    def _supported_frame(frame):
        return type(frame) is int and 0 <= frame <= DESKTOP_MAX_PREVIEW_FRAME

    def _unsupported_frame_message(self, frame):
        return f'Preview frame {frame} is unsupported by this desktop (maximum {DESKTOP_MAX_PREVIEW_FRAME}). Apply is unavailable; no correction snapshot was created or frame value changed.'

    @property
    def last_resolved_job(self):
        return self.last_resolved

    @property
    def last_emitted_job(self):
        """Compatibility alias: a committed job may be retained without emission."""
        return self.last_resolved

    def _retain_resolution(self, value, *, emitted, accepted, message):
        job, frame, output, context = value
        self.last_resolved = job
        self.last_resolved_context = context
        config, _ = job.configurations()
        artifact = str(config.corrections_path) if config.corrections_path is not None else None
        report = dict(context, emitted=emitted, accepted=accepted, preview_frame=frame,
                      resolved_correction_snapshot=artifact, application_message=_text(message, 2048))
        import json
        if len(json.dumps(report, allow_nan=False).encode()) > 64 * 1024:
            report = {'project_identity': context.get('project_identity'), 'requested_job_identity': context.get('requested_job_identity'),
                      'emitted': emitted, 'accepted': accepted, 'preview_frame': frame,
                      'resolved_correction_snapshot': artifact, 'application_message': _text(message, 512),
                      'additional_context': 'Retained separately with the committed job and immutable project inspection.'}
        self.application_report = report
        line = f'Committed verified snapshot · frame {frame} · output {_text(output, 1024)} · scratch {_text(config.temp_dir, 1024)} · correction snapshot {artifact or "none"}. Sent to draft editor: {emitted}. Retained for review; not deleted automatically.'
        self.details.setPlainText((_text(self.details.toPlainText(), 40 * 1024) or '') + '\n' + line)
        self.status.setText(message)

    def apply(self):
        if self.busy or self.closing or self.inspection is None or not self.inspection_ready or not self.output.text().strip() or not self.scratch.text().strip():
            return
        if not self._supported_frame(self.inspection_frame):
            self.status.setText(self._unsupported_frame_message(self.inspection_frame))
            return
        inspection = self.inspection
        output, scratch = Path(self.output.text().strip()), Path(self.scratch.text().strip())
        def resolve(check):
            context, frame = _loaded_context(inspection, output, scratch)
            if not self._supported_frame(frame):
                raise ValueError(self._unsupported_frame_message(frame))
            job = inspection.apply(output_dir=output, temp_dir=scratch, check=check)
            return job, frame, str(output), context
        self._begin('apply', resolve)

    def acknowledge_application(self, accepted: bool, message: str):
        """Main window acknowledges this synchronous loaded signal explicitly."""
        if type(accepted) is not bool or not isinstance(message, str) or not message.strip():
            raise ValueError('Application acknowledgement requires a boolean and explanatory text.')
        if self._emitting_application:
            bounded_message = _text(message, 4096)
            assert bounded_message is not None
            self._application_ack = accepted, bounded_message

    def _show_report(self, report):
        self.report = report
        dependencies = report.get('dependencies', [])
        selected = self.collect_ids
        self.page = report.get('page', 0)
        self.page_count = report.get('page_count', 1)
        count = report.get('dependency_count', len(dependencies))
        first = self.page * PAGE_SIZE + 1 if count else 0
        self.page_label.setText(f'Dependency groups {first}–{first + len(dependencies) - 1 if count else 0} / {count} · page {self.page + 1}/{self.page_count}')
        self.table.blockSignals(True)
        self.table.setRowCount(len(dependencies))
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        for row, dependency in enumerate(dependencies):
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, dependency['id'])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | (Qt.ItemFlag.ItemIsUserCheckable if self.mode == 'job' else Qt.ItemFlag.NoItemFlags))
            if self.mode == 'job':
                item.setCheckState(Qt.CheckState.Checked if dependency['id'] in selected else Qt.CheckState.Unchecked)
            else:
                item.setText('Collected' if dependency.get('collected') else 'External')
            self.table.setItem(row, 0, item)
            staged = dependency['id'] in self.unverified_relinks
            display_path = _text(self.relinks[dependency['id']], 1024) if staged else dependency.get('path', '')
            display_status = 'Staged relink; identity NOT VERIFIED' if staged else dependency.get('status', '')
            if self.mode == 'job' and dependency.get('collected') and dependency['id'] not in selected:
                display_status = 'Last verified plan includes through dependency closure · ' + display_status
                item.setToolTip('Included by a selected dependency, such as the corrections document. Check this row only to request it independently in the next plan.')
            values = [dependency.get('role', ''), display_path, display_status,
                      dependency.get('file_count', len(dependency.get('files', []))), dependency.get('bytes', '')]
            for column, value in enumerate(values, 1):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(2, 360)
        models = report.get('model_references', {})
        self.models.setText(f"Preserved model references · asset root: {models.get('asset_root') or 'default local assets'} · weights: {models.get('weights') or 'default local weights'} · checkpoint/backend compatibility NOT VERIFIED. No automatic model switch.")
        lines = [f"Dependencies: {report.get('dependency_count', len(dependencies))} · ready: {report.get('ready', False)} · preview frame: {report.get('preview_frame', self.frame)}",
                 f"Verification read bytes: {report.get('read_bytes', 0)} · selected copy bytes: {report.get('copy_bytes', 0)}",
                 f"Correction semantic identity: {report.get('correction_identity') or 'none / not yet verified'}",
                 'Whole source/hint sequences retain original frame names, ranges, FPS and colour settings. Originals are never moved or overwritten.']
        if report.get('archive_path_flavour'):
            lines.append(f"Archived path flavour: {report['archive_path_flavour']} · foreign: {report.get('foreign_paths', False)} · {report.get('portability_scope') or 'Inspect verified dependency paths before applying.'}")
        lines.extend(report.get('requested_settings_lines', []))
        if report.get('omitted_dependencies'):
            lines.append(f"{report['omitted_dependencies']} other dependency groups are available through Previous/Next groups. Page changes use the frozen snapshot without rescanning or losing collection choices. Verification and manifests retain the full inventory.")
        if models.get('omitted_entries'):
            lines.append(f"{models['omitted_entries']} additional model-reference entries omitted from this bounded display; complete references remain in the manifest.")
        for entry in models.get('entries', []):
            lines.append(f"Model reference {entry.get('id')}: {entry.get('status')} · {entry.get('path')} · {entry.get('bytes')} bytes; compatibility NOT VERIFIED")
        for dependency in dependencies:
            lines.append(f"{dependency.get('role')}: {'required' if dependency.get('required') else 'optional'} · {dependency.get('status')} · {dependency.get('path')}")
        self.details.setPlainText('\n'.join(lines))
        self._buttons()

    def cancel(self):
        if self.busy:
            self.cancellation.cancel('Portable project operation cancelled by user')
            if self.operation != 'page':
                self.invalidate_verification()
            self.status.setText('Cancellation requested. Waiting for the single owned operation; a committed publication will still be reported.')

    def poll(self):
        if self.io_thread is not None and self.io_thread.is_alive():
            return
        try:
            kind, value, error = self.results.get_nowait()
        except queue.Empty:
            self._buttons()
            if self.closing:
                self.close()
            return
        self.io_thread = None
        if kind == 'apply' and error is None and (self.closing or self.cancellation.cancelled):
            job, _, _, _ = value
            config, _ = job.configurations()
            reason = 'closing' if self.closing else 'cancellation'
            message = f'Apply snapshot committed before {reason}; not sent to the draft. Retained correction snapshot: {config.corrections_path or "none"}. Snapshot/report retained for review; no automatic deletion.'
            self._retain_resolution(value, emitted=False, accepted=False, message=message)
            self._buttons()
            if self.closing:
                self.close()
            return
        if not self.closing:
            if error is not None:
                self.status.setText(error)
            elif self.cancellation.cancelled and kind != 'publish':
                self.status.setText('Cancelled; no late result applied to the draft.')
            elif kind == 'draft_inventory':
                self.full_report, summary = value
                self.mode, self.package, self.inventory = 'job', None, summary
                self.relinks.clear()
                self.unverified_relinks.clear()
                self.collect_ids.clear()
                self.table.setRowCount(0)
                self._show_report(summary)
                self.status.setText('Metadata inventory only. Select collection choices and explicit budgets, then Verify.')
            elif kind == 'package_inventory':
                self.package, self.full_report, self.inventory = value
                self.collect_ids.clear()
                self.mode = 'package'
                self.relinks.clear()
                self.unverified_relinks.clear()
                self.table.setRowCount(0)
                self._show_report(self.inventory)
                self.status.setText('Package metadata only. Review missing references, stage deliberate relinks, then Verify.')
            elif kind == 'plan':
                self.plan, self.full_report, summary = value
                self._show_report(summary)
                self.status.setText('Verified immutable publication plan. Review choices and publish explicitly to a fresh child.')
            elif kind == 'inspect':
                self.inspection, self.full_report, summary = value
                self.unverified_relinks.clear()
                self.inspection_ready = summary.get('ready', False)
                self.inspection_frame = summary.get('preview_frame')
                self._show_report(summary)
                self.status.setText(self._unsupported_frame_message(self.inspection_frame) if self.inspection_ready and not self._supported_frame(self.inspection_frame) else ('Verified package ready for explicit output/scratch choices and Apply.' if self.inspection_ready else 'Package is not ready. Review missing, mismatched or unverified dependencies; no draft was changed.'))
            elif kind == 'page':
                self._show_report(value)
                self.status.setText('Frozen inventory page displayed. Verification and staged collection choices are unchanged.')
            elif kind == 'publish':
                self.published_path = Path(value)
                self.package_path.setText(str(value))
                self.plan = None
                self.status.setText('Fresh package published: ' + str(value) + '. Preview its inventory to inspect or relink; nothing was queued.')
            elif kind == 'apply':
                job, frame, output, context = value
                self.last_resolved = job
                self.last_resolved_context = context
                self._application_ack = None
                self._emitting_application = True
                try:
                    self.loaded.emit(job, frame, output, context)
                finally:
                    self._emitting_application = False
                accepted, message = self._application_ack if self._application_ack is not None else (None, 'Verified snapshot emitted; confirm draft application. No render or model switch started.')
                self._retain_resolution(value, emitted=True, accepted=accepted, message=message)
        self._buttons()
        if self.closing:
            self.close()

    def shutdown(self) -> bool:
        self.close()
        return not self.busy

    def closeEvent(self, event: QCloseEvent):
        self.closing = True
        self.cancellation.cancel('Portable project dialog closing')
        self.invalidate_verification()
        if self.busy:
            self.status.setText('Closing: waiting for the owned file operation to retire.')
            self._buttons()
            event.ignore()
        else:
            self.timer.stop()
            event.accept()
