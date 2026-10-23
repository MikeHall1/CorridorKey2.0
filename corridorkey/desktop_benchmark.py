"""Optional developer window inspecting actual benchmark runner reports."""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from .benchmark_report import export_report, read_report, report_view
from .benchmark_task import BenchmarkTask
from .desktop_evaluation_dossier import EvaluationDossierPanel


class BenchmarkPanel(QDialog):
    run_requested = Signal(object, object)
    cancel_requested = Signal()

    def __init__(self, parent=None, *, factory: Callable[..., BenchmarkTask] = BenchmarkTask):
        super().__init__(parent)
        self.setWindowTitle('CorridorKey · Developer benchmarks')
        self.resize(960, 720)
        self.factory = factory
        self.task: BenchmarkTask | None = None
        self.waiting = False
        self.revision = -1
        self.report_shown = False
        self.inspection: threading.Thread | None = None
        self.inspected: queue.Queue[tuple[str, str, str | None, str]] = queue.Queue(maxsize=1)
        self.dossier_panel = EvaluationDossierPanel(self)
        layout = QVBoxLayout(self)
        note = QLabel('Runs the selected benchmark protocol in an owned process. Paired manifests alternate matched baseline/candidate measurements. Render work must finish first; the resident render worker stops before benchmarking. CPU test times are not inference FPS. Profiles, quality references and cold/warm policy come from the selected manifest. Baseline inspection does not imply a controlled comparison.')
        note.setWordWrap(True)
        layout.addWidget(note)
        fields = QFormLayout()
        self.manifest, self.output, self.report = QLineEdit(), QLineEdit(), QLineEdit()
        self.manifest.setText(str(Path(__file__).resolve().parents[1] / 'benchmarks/suites/cpu.json'))
        self.report.setReadOnly(True)
        fields.addRow('Benchmark manifest', self.manifest)
        fields.addRow('Output parent', self.output)
        fields.addRow('Actual report', self.report)
        layout.addLayout(fields)
        buttons = QHBoxLayout()
        for label, callback in [('Select suite…', self.choose_suite), ('Output parent…', self.choose_output),
                                ('Run', self.request_run), ('Cancel', self.cancel_requested.emit),
                                ('Inspect report…', self.choose_report), ('Baseline…', self.choose_baseline),
                                ('Export report…', self.choose_export),
                                ('Evaluation dossier…', self.dossier_panel.show)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel('Idle · No measurements loaded')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        tabs = QTabWidget()
        self.report_view, self.logs, self.baseline = QPlainTextEdit(), QPlainTextEdit(), QPlainTextEdit()
        for widget, title in [(self.report_view, 'Actual report / timings / resources / quality'),
                              (self.logs, 'Bounded process log tail'), (self.baseline, 'Baseline report (independent)')]:
            widget.setReadOnly(True)
            tabs.addTab(widget, title)
        layout.addWidget(tabs)
        self.logs.setPlaceholderText('Actual CLI output only; no invented stage or progress values.')
        self.baseline.setPlaceholderText('Select an existing report. Raw conditions/results remain separate; no speedup is inferred.')

    @property
    def busy(self) -> bool:
        return self.waiting or (self.task is not None and not self.task.done.is_set())

    @property
    def io_busy(self) -> bool:
        return self.inspection_busy or self.dossier_panel.io_busy

    @property
    def inspection_busy(self) -> bool:
        return (self.inspection is not None and self.inspection.is_alive()) or not self.inspected.empty()

    @property
    def dossier_busy(self) -> bool:
        return self.busy or self.inspection_busy or getattr(self.parent(), 'closing', False)

    def choose_suite(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select benchmark manifest', '', 'JSON (*.json)')
        if path:
            self.manifest.setText(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Select benchmark output parent')
        if path:
            self.output.setText(path)

    def request_run(self):
        if self.busy or self.io_busy:
            self.status.setText('A benchmark is already running or waiting for worker shutdown.')
        elif not self.manifest.text().strip() or not self.output.text().strip():
            self.status.setText('Select a manifest and output parent first.')
        else:
            self.run_requested.emit(Path(self.manifest.text()), Path(self.output.text()))

    def begin(self, manifest: Path, parent: Path):
        self.waiting = False
        self.report_shown = False
        self.revision = -1
        self.task = self.factory(manifest, parent)
        self.report.setText(str(self.task.report_path))
        self.report_view.clear()
        self.logs.clear()

    def inspect(self, path: Path, role: str = 'report'):
        if self.io_busy:
            self.status.setText('A report inspection/export is still running.')
            return
        if self.busy and role != 'baseline':
            self.status.setText('Wait for the active benchmark before replacing/exporting its report view.')
            return
        self.status.setText('Reading local report…')
        self.inspection = threading.Thread(target=self._inspect, args=(path, role), daemon=True)
        self.inspection.start()

    def _inspect(self, path: Path, role: str):
        try:
            data, _ = read_report(path)
            view, error = report_view(data), None
        except Exception as exc:
            logging.getLogger(__name__).exception('Benchmark report inspection failed')
            view, error = '', f'{type(exc).__name__}: {exc}'
        self.inspected.put((role, str(path), error, view))

    def choose_report(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Inspect actual benchmark report', '', 'JSON (*.json)')
        if path:
            self.inspect(Path(path))

    def choose_baseline(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Inspect independent baseline report', '', 'JSON (*.json)')
        if path:
            self.inspect(Path(path), 'baseline')

    def choose_export(self):
        if self.busy or self.io_busy:
            self.status.setText('Wait for benchmark/inspection completion before exporting.')
            return
        if not self.report.text():
            self.status.setText('No report selected.')
            return
        destination, _ = QFileDialog.getSaveFileName(self, 'Export exact report to a new file', '', 'JSON (*.json)')
        if destination:
            self.inspection = threading.Thread(target=self._export, args=(Path(self.report.text()), Path(destination)), daemon=True)
            self.inspection.start()

    def _export(self, source: Path, destination: Path):
        error = 'Report export did not reach a terminal result.'
        try:
            export_report(source, destination)
            error = None
        except BaseException as exc:
            logging.getLogger(__name__).exception('Benchmark report export failed')
            error = f'{type(exc).__name__}: {exc}'
        finally:
            self.inspected.put(('export', str(destination), error, ''))

    def refresh(self):
        self.dossier_panel.poll()
        if self.task is not None:
            snapshot = self.task.snapshot()
            if snapshot['revision'] != self.revision:
                self.revision = snapshot['revision']
                self.status.setText(f"{snapshot['status']} · PID {snapshot['pid']} · exit {snapshot['returncode']} · {snapshot['error'] or snapshot['cleanup_error'] or ''}")
                self.logs.setPlainText(snapshot['log_tail'])
                if snapshot['report_available'] and not self.report_shown:
                    self.report_view.setPlainText(snapshot['report_view'])
                    self.report_shown = True
        try:
            role, path, error, view = self.inspected.get_nowait()
        except queue.Empty:
            return
        if error is not None:
            self.status.setText(error)
        elif role == 'baseline':
            self.baseline.setPlainText(path + '\n' + view)
            self.status.setText('Baseline loaded independently; inspect conditions before comparing.')
        elif role == 'export':
            self.status.setText('Exact report exported: ' + path)
        else:
            self.report.setText(path)
            self.report_view.setPlainText(view)
            self.status.setText('Actual local report loaded: ' + path)
