"""Explicit local evaluation dossier preparation; no inference or quality verdict."""
from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .control import CancellationToken, CancelledError, ProgressEvent

MAX_RESULT_BYTES = 32 * 1024 * 1024


def number(value: Any, *, percent: bool = False) -> str:
    if type(value) not in (int, float):
        return 'Unavailable'
    return f'{value * 100:+.3g}%' if percent else f'{value:.8g}'


@dataclass(frozen=True)
class DossierRequest:
    operation: str
    manifest: str
    report: bytes | None = None
    parent: str = ''
    include_media: bool = False
    max_media_bytes: int = 0


def run_operation(request: DossierRequest, *, check, progress):
    if request.operation == 'build':
        from .evaluation_dossier import build_dossier
        return build_dossier(Path(request.manifest), check=check, progress=progress)
    if request.operation == 'export' and request.report is not None:
        from .evaluation_dossier_export import export_dossier
        return export_dossier(json.loads(request.report), Path(request.parent),
                              include_media=request.include_media, max_media_bytes=request.max_media_bytes,
                              check=check, progress=progress)
    raise ValueError('Build a dossier before exporting.')


class DossierTask:
    def __init__(self, request: DossierRequest, runner: Callable = run_operation):
        self.request, self.runner = request, runner
        self.cancellation = CancellationToken()
        self.results: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest: ProgressEvent | None = None
        self.thread = threading.Thread(target=self._run, name='evaluation-dossier', daemon=True)
        self.thread.start()

    @property
    def busy(self):
        return self.thread.is_alive() or not self.results.empty()

    def cancel(self):
        self.cancellation.cancel('Evaluation dossier operation cancelled by user')

    def _progress(self, event: ProgressEvent):
        with self.lock:
            self.latest = event

    def progress(self):
        with self.lock:
            return self.latest

    def _run(self):
        data: dict[str, Any] = {'status': 'failed', 'error': 'Unexpected dossier worker termination'}
        raw = b'{"status":"failed","error":"Unexpected dossier worker termination"}'
        try:
            self.cancellation.check()
            data = self.runner(self.request, check=self.cancellation.check, progress=self._progress)
            if not isinstance(data, dict):
                raise TypeError('Dossier worker must return a terminal report object.')
            raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8')
            if len(raw) > MAX_RESULT_BYTES:
                raise ValueError('Dossier result exceeds the 32 MiB interface limit.')
        except CancelledError as exc:
            raw = json.dumps({'status': 'cancelled', 'error': str(exc)[:4096]}).encode()
        except BaseException as exc:
            logging.getLogger(__name__).exception('Dossier operation failed')
            # A committed exporter result remains authoritative even if its
            # ancillary serialization fails. Never replace it with absent output.
            fallback: dict[str, Any] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:4096]}
            if self.request.operation == 'export' and isinstance(data, dict):
                fallback['status'] = data.get('status') if data.get('status') in ('complete', 'failed', 'cancelled') else 'failed'
                fallback['usable'] = data.get('usable') is True
                for key in ('destination', 'warning', 'partial_path'):
                    value = data.get(key)
                    fallback[key] = value[:4096] if isinstance(value, str) else None
            raw = json.dumps(fallback, allow_nan=False).encode()

        finally:
            self.results.put(raw)
            self.done.set()


class EvaluationDossierPanel(QDialog):
    def __init__(self, parent=None, *, runner: Callable = run_operation):
        super().__init__(parent)
        self.runner = runner
        self.task: DossierTask | None = None
        self.report: bytes | None = None
        self.media_bytes: int | None = None
        self.close_when_done = False
        self.manifest_revision = 0
        self.task_revision = -1
        self.setWindowTitle('CorridorKey · Evaluation dossier')
        self.resize(1280, 820)
        layout = QVBoxLayout(self)
        note = QLabel('Build from existing local paired reports and an explicit case/category manifest. No models run and no files leave this computer. Category summaries and declared tolerance screens are evidence, not professional quality acceptance. Missing references and unassessed metrics remain visible. Export creates a fresh directory; sources and earlier reports are preserved.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.manifest_path, self.parent_path = QLineEdit(), QLineEdit()
        self.max_media_bytes = QLineEdit('0')
        self.include_media = QCheckBox('Include assignment-blinded review media')
        form.addRow('Existing dossier manifest', self.manifest_path)
        form.addRow('Existing export parent', self.parent_path)
        form.addRow('Review media byte ceiling (0..9223372036854775807)', self.max_media_bytes)
        form.addRow(self.include_media)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.build_button, self.export_button, self.cancel_button = QPushButton('Build dossier'), QPushButton('Export dossier…'), QPushButton('Cancel operation')
        for label, callback in [('Manifest…', self.choose_manifest), ('Export parent…', self.choose_parent)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button)
        for button, callback in ((self.build_button, self.build), (self.export_button, self.export), (self.cancel_button, self.cancel)):
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel('No dossier built. Review media preparation and human review are pending.')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.coverage = QLabel('No cases assessed.')
        self.coverage.setWordWrap(True)
        self.coverage.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.coverage)
        self.media_summary = QLabel('Review media preparation and human review are pending.')
        self.media_summary.setWordWrap(True)
        self.media_summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.media_summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Case / category', 'Metric', 'Baseline', 'Candidate', 'Change / worst case', 'Coverage', 'Assessment / unavailable reason'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setWordWrap(True)
        for column, width in enumerate((150, 180, 95, 95, 310, 105)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table)
        self.result_path = QLineEdit()
        self.result_path.setReadOnly(True)
        self.result_path.setPlaceholderText('Actual exported or retained partial directory')
        layout.addWidget(self.result_path)
        self.manifest_path.textChanged.connect(self.invalidate_report)
        for widget in (self.parent_path, self.max_media_bytes):
            widget.textChanged.connect(self.invalidate_export)
        self.include_media.toggled.connect(self.invalidate_export)
        self._controls()

    @property
    def io_busy(self):
        return self.task is not None and self.task.busy

    def _can_start(self):
        parent = self.parent()
        return not (self.io_busy or self.close_when_done or getattr(parent, 'dossier_busy', False)
                    or getattr(parent, 'close_when_done', False) or getattr(parent, 'closing', False))

    def _controls(self):
        idle = self._can_start()
        self.build_button.setEnabled(idle and bool(self.manifest_path.text().strip()))
        self.export_button.setEnabled(idle and self.report is not None)
        self.cancel_button.setEnabled(self.io_busy)

    def invalidate_report(self):
        self.manifest_revision += 1
        self.report = None
        self.media_bytes = None
        self.media_summary.setText('Review media preparation and human review are pending.')
        self.table.setRowCount(0)
        self.coverage.setText('Manifest changed. Build again to assess these inputs.')
        self.invalidate_export()

    def invalidate_export(self):
        self.result_path.clear()
        self._controls()

    def choose_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select existing evaluation dossier manifest', '', 'JSON (*.json)')
        if path:
            self.manifest_path.setText(path)

    def choose_parent(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose existing parent for a fresh dossier directory')
        if path:
            self.parent_path.setText(path)

    def build(self):
        if not self._can_start():
            self.status.setText('Wait for the current benchmark, dossier operation or closing work.')
            return
        manifest = self.manifest_path.text().strip()
        if not manifest:
            self.status.setText('Choose an existing dossier manifest.')
            return
        self.invalidate_report()
        self.task_revision = self.manifest_revision
        self.task = DossierTask(DossierRequest('build', manifest), self.runner)
        self.status.setText('Building local dossier; no models run.')
        self._controls()

    def export(self):
        if not self._can_start() or self.report is None:
            self.status.setText('Build a current dossier and wait for other work before exporting.')
            return
        try:
            parent = self.parent_path.text().strip()
            if not parent:
                raise ValueError('Choose an existing export parent.')
            text = self.max_media_bytes.text().strip()
            if not text.isascii() or not text.isdecimal():
                raise ValueError('Review media byte ceiling must be an exact integer in 0..9223372036854775807.')
            maximum = int(text)
            if maximum > 2**63 - 1:
                raise ValueError('Review media byte ceiling exceeds 9223372036854775807.')
            if self.include_media.isChecked() and maximum == 0:
                raise ValueError('Including review media requires an explicit positive byte ceiling covering all selected files.')
            if self.include_media.isChecked() and self.media_bytes is not None and maximum < self.media_bytes:
                raise ValueError(f'Review media requires {self.media_bytes} bytes; raise the ceiling or leave media unchecked.')
            request = DossierRequest('export', self.manifest_path.text().strip(), self.report,
                                     parent, self.include_media.isChecked(), maximum)
            self.result_path.clear()
            self.task = DossierTask(request, self.runner)
            self.status.setText('Exporting captured dossier/options; inputs are revalidated before publication.')
        except ValueError as exc:
            self.status.setText(str(exc))
        self._controls()

    def cancel(self):
        if self.task is not None and self.io_busy:
            self.task.cancel()
            self.status.setText('Cancellation requested; waiting for actual terminal outcome.')

    def _show_report(self, report: dict[str, Any]):
        coverage = report.get('coverage', {})
        names = (('expected_cases', 'cases expected'), ('assessed_cases', 'assessed'),
                 ('partial_cases', 'partial'), ('unassessed_cases', 'unassessed'))
        counts = [f'{coverage[key]} {label}' for key, label in names if type(coverage.get(key)) is int]
        if type(coverage.get('required_cases')) is int and type(coverage.get('required_assessed_cases')) is int:
            counts.append(f"{coverage['required_assessed_cases']} / {coverage['required_cases']} required cases assessed")
        self.coverage.setText(' · '.join(counts) + '. Human review pending; metric screens are not product acceptance.')
        media = report.get('review_media')
        if isinstance(media, list) and all(isinstance(item, dict) and type(item.get('bytes')) is int and item['bytes'] >= 0 for item in media):
            self.media_bytes = sum(item['bytes'] for item in media)
            self.media_summary.setText(f'{len(media)} selected review files; {self.media_bytes} encoded bytes. Export hides engine assignment; embedded media metadata is not anonymized. Human ratings remain pending.')
        else:
            self.media_bytes = None
            self.media_summary.setText('Selected media byte inventory unavailable. Media preparation and human review are pending.')
        self.table.setRowCount(0)
        labels = {'within_limit': 'Within declared metric limit', 'regression': 'Metric regression',
                  'unassessed': 'Unassessed'}
        truncated = False
        for group, heading in (('categories', 'Category'), ('cases', 'Case')):
            for entry in report.get(group, []):
                identity = entry.get('category') if group == 'categories' else entry.get('id')
                metrics = entry.get('metrics', {})
                for metric, result in metrics.items() if metrics else [('', {})]:
                    if self.table.rowCount() >= 4096:
                        truncated = True
                        break
                    units = result.get('units', '')
                    change = f"Δ {number(result.get('absolute_delta'))}; relative {number(result.get('relative_delta'), percent=True)}"
                    worst = result.get('worst_case_median')
                    if isinstance(worst, dict):
                        change += f"\nworst case {worst.get('case_id', 'unknown')}: {number(worst.get('relative_delta'), percent=True)}"
                    absolute_worst = result.get('worst_candidate_error')
                    if isinstance(absolute_worst, dict):
                        change += (f"\nlargest error {absolute_worst.get('case_id', 'unknown')}: "
                                   f"{number(absolute_worst.get('value'))} {absolute_worst.get('units', units)}")
                    observed = result.get('worst_observed')
                    extra = ''
                    if isinstance(observed, dict):
                        extra = f"\nWorst observed: {observed.get('case_id', 'unknown')}, pair {observed.get('pair_index', 'unknown')}, {number(observed.get('relative_delta'), percent=True)}"
                    population = result.get('coverage', entry.get('coverage', {}))
                    assessed = population.get('assessed', population.get('assessed_cases'))
                    expected = population.get('expected', population.get('expected_cases'))
                    coverage_text = f'{assessed if assessed is not None else "unknown"} / {expected if expected is not None else "unknown"} assessed'
                    reason = next((row.get('reason') for row in result.get('repetitions', []) if row.get('reason')), None)
                    screen = result.get('screen')
                    assessment = labels.get(screen, str(result.get('status', entry.get('status', 'unassessed')))) if isinstance(screen, str) else str(result.get('status', entry.get('status', 'unassessed')))
                    if result.get('status') in ('partial', 'unassessed'):
                        assessment = str(result['status']).capitalize() + ' · ' + assessment
                    if reason:
                        assessment += ': ' + str(reason)
                    review_status = entry.get('review_selection_status')
                    if group == 'cases' and review_status not in (None, 'prepared', 'available', 'selected'):
                        assessment += f"; review media: {review_status}"
                        if entry.get('review_selection_reason'):
                            assessment += ': ' + str(entry['review_selection_reason'])
                    values = [f'{heading}: {identity}\n{entry.get("split", "unspecified")}',
                              f'{metric}\n{units}' if metric else 'No metrics available',
                              number(result.get('baseline_median')), number(result.get('candidate_median')),
                              change, coverage_text, assessment]
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    self.table.setRowHeight(row, 76)
                    for column, value in enumerate(values):
                        item = QTableWidgetItem(str(value)[:4096])
                        item.setToolTip((str(value) + (extra if column == 4 else ''))[:4096])
                        self.table.setItem(row, column, item)
        if truncated:
            self.coverage.setText(self.coverage.text() + ' View limited to 4096 rows; full dossier metadata is preserved for export.')

    def poll(self):
        if self.task is not None:
            progress = self.task.progress()
            if self.task.thread.is_alive() and progress is not None:
                self.status.setText(f'{progress.stage[:256]}: {progress.completed} / {progress.total if progress.total is not None else "unknown"}')
            try:
                data = json.loads(self.task.results.get_nowait())
            except queue.Empty:
                data = None
            if data is not None:
                if self.task.request.operation == 'build':
                    if self.task_revision != self.manifest_revision:
                        self.status.setText('Manifest changed during build; build again. No report selected.')
                    elif data.get('status') == 'complete' and isinstance(data.get('report'), dict):
                        self.report = json.dumps(data['report'], allow_nan=False, separators=(',', ':')).encode()
                        self._show_report(data['report'])
                        self.status.setText('Dossier built. Media preparation and human review are pending; nothing exported.')
                    else:
                        self.status.setText(f"Build {data.get('status', 'failed')}: {data.get('error') or 'No completed dossier'}"[:4096])
                else:
                    destination, partial = data.get('destination'), data.get('partial_path')
                    self.result_path.setText(str(destination or partial or ''))
                    state = 'Published dossier is usable' if data.get('usable') is True else 'Dossier is not complete'
                    review = ('Assignment-blinded packet prepared; human review is pending.' if self.task.request.include_media and data.get('usable') is True
                              else 'Metadata only; media preparation and human review are pending.')
                    self.status.setText(f"Export {data.get('status', 'failed')}. {state}. {review} {data.get('error') or ''} {data.get('warning') or ''} Retained private path: {partial or 'none'}"[:4096])
        self._controls()
        if self.close_when_done and not self.io_busy:
            self.close_when_done = False
            self.close()

    def closeEvent(self, event: QCloseEvent):
        if self.io_busy:
            self.close_when_done = True
            self.cancel()
            event.ignore()
        else:
            event.accept()
