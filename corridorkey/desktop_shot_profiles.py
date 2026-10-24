"""Explicit off-thread source analysis and future-only shot profile application."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from .control import CancellationToken
from .shot_profiles import (
    MAX_BYTES,
    ProfilePlan,
    analyze_job,
    apply_profiles,
    load_profiles,
    save_profiles,
)


class ShotProfilesPanel(QDialog):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.setWindowTitle('Shot profiles · explicit source heuristics')
        self.resize(850, 650)
        self.io_thread: threading.Thread | None = None
        self.token: CancellationToken | None = None
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self.revision = 0
        self.closing = False
        layout = QVBoxLayout(self)
        note = QLabel('CPU source analysis only; no model loaded. Adjacent RGB change and image-edge detail are uncalibrated proxies, not subject edges, quality or VRAM fit. Proposed feathering may harm detail: inspect/edit before Apply. Rows use absolute [In,Out), optional enabled flags and only Stage 2 caps (≤ global), expansion and feather. One original timeline, hints/corrections/resets and queued jobs stay unchanged. Save to a new local JSON; reopen with its matching base recipe. Disable clears profiles from future settings only.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.document = QPlainTextEdit()
        self.document.setPlaceholderText('Analyze the current selected source range, or open saved recommendations. Edit row controls/ranges/enabled, then explicitly Apply to future settings.')
        self.document.textChanged.connect(self.invalidate)
        layout.addWidget(self.document)
        buttons = QHBoxLayout()
        for title, callback in [('Analyze selected range', self.analyze), ('Open…', self.choose_open),
                                ('Save as…', self.choose_save), ('Inspect measured candidate…', self.choose_recommendation),
                                ('Apply to future job', self.apply),
                                ('Disable future profiles', self.disable), ('Cancel', self.cancel)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel('Nothing analyzed or applied. Source budgets: ≤100000 frames, ≤256 rows, conservative encoded reads ≤2 GiB; profile metadata ≤1 MiB. Sequence images: ≤16 Mi-pixels /256 MiB encoded each, including UHD/DCI 4K. Native decoding still allocates a source frame; no RAM cap.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @property
    def busy(self):
        return (self.io_thread is not None and self.io_thread.is_alive()) or not self.results.empty()

    def invalidate(self, *_):
        self.revision += 1
        if self.token is not None:
            self.token.cancel('Source/settings/document changed')

    def cancel(self):
        self.invalidate()
        self.status.setText('Cancellation requested; waiting for owned I/O. A committed Save remains on disk.')

    def _start(self, kind, operation):
        if self.closing or self.busy:
            raise ValueError('Wait for the current shot-profile operation to retire')
        token = CancellationToken()
        self.token = token
        revision = self.revision
        def run():
            try:
                result, error = operation(token.check), None
            except Exception as exc:
                logging.getLogger(__name__).exception('Shot profile operation failed')
                result, error = None, f'{type(exc).__name__}: {exc}'
            self.results.put((kind, revision, token, result, error))
        self.io_thread = threading.Thread(target=run, name='shot-profile-io', daemon=True)
        self.io_thread.start()
        self.status.setText(kind + ' running off-thread; no inference or queue mutation.')

    def analyze(self):
        try:
            job = self.workspace.job(preview_only=True)
            self._start('Analyze', lambda check: analyze_job(job, check=check))
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def open_path(self, path: Path):
        try:
            def load(check):
                check()
                _, plan = load_profiles(path)
                check()
                return plan
            self._start('Open', load)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def choose_open(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open shot profiles', '', 'JSON (*.json)')
        if path:
            self.open_path(Path(path))

    def _plan(self):
        text = self.document.toPlainText()
        if len(text.encode()) > MAX_BYTES:
            raise ValueError('Shot profile editor exceeds 1 MiB; nothing parsed or applied')
        return ProfilePlan.from_bytes(text.encode())

    def save_path(self, path: Path):
        try:
            job = self.workspace.job(preview_only=True)
            plan = self._plan()
            def save(check):
                save_profiles(path, plan, job, check=check)
                return str(path)
            self._start('Save', save)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def choose_save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save new shot profile JSON (existing files refused)', '', 'JSON (*.json)')
        if path:
            self.save_path(Path(path))

    def recommend_request(self, request):
        """Inspect existing evidence off-thread; never apply a candidate automatically."""
        from .shot_profile_recommendations import recommend
        try:
            if not isinstance(request, dict):
                raise TypeError('Measured recommendation request must be an object')
            job = self.workspace.job(preview_only=True)
            self._start('Measured recommendation', lambda check: recommend(job, check=check, **request))
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def choose_recommendation(self):
        from .diagnostics import finite_json_float
        from .local_json import _depth, _pairs
        text, accepted = QInputDialog.getMultiLineText(self, 'Inspect local measured whole-plan candidate',
            'Explicit local paths + expected lowercase SHA256 receipts. Existing pair/numerical/resource gate must pass. '
            'Optional visual_evidence_path + visual_evidence_sha256 preserve visual requirements. Nothing applied.',
            json.dumps({key: '' for key in ('report_path', 'policy_path', 'policy_sha256',
                'numerical_evidence_path', 'numerical_evidence_sha256',
                'resource_evidence_path', 'resource_evidence_sha256')}, indent=2))
        if not accepted:
            return
        try:
            raw = text.encode()
            if len(raw) > 64 * 1024:
                raise ValueError('Measured recommendation request exceeds 64 KiB')
            _depth(raw, 8)
            request = json.loads(raw, object_pairs_hook=_pairs, parse_float=finite_json_float,
                                 parse_constant=finite_json_float)
            self.recommend_request(request)
        except (ValueError, TypeError, OSError) as exc:
            self.status.setText(str(exc))

    def apply(self):
        try:
            job = self.workspace.job(preview_only=True)
            plan = self._plan()
            self._start('Apply', lambda check: apply_profiles(job, plan, check=check))
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def disable(self):
        self.invalidate()
        if self.workspace.update_settings({'shot_profiles': None}):
            self.status.setText('Profiles disabled in future settings. Saved recommendations and queued snapshots preserved.')

    def poll(self):
        if self.io_thread is not None and self.io_thread.is_alive():
            return
        try:
            kind, revision, token, result, error = self.results.get_nowait()
        except queue.Empty:
            return
        self.io_thread, self.token = None, None
        stale = revision != self.revision or token.cancelled or self.closing
        if kind == 'Save':
            self.status.setText(error or f'Saved new local profile document: {result}. Nothing applied; committed saves survive cancellation.')
        elif stale:
            self.status.setText('Stale/cancelled completion rejected. Analyze/open/apply explicitly again.')
        elif error:
            self.status.setText(error)
        elif kind == 'Measured recommendation':
            plan, summary = result
            self.document.setPlainText(json.dumps(plan.to_dict(), indent=2))
            self.status.setText(json.dumps(summary, sort_keys=True))
        elif kind in ('Analyze', 'Open'):
            self.document.setPlainText(json.dumps(result.to_dict(), indent=2))
            self.status.setText('Recommendations captured; inspect/edit row settings and enabled flags, Save or explicitly Apply. No form/queue changes.')
        elif kind == 'Apply':
            config, _ = result.configurations()
            self.workspace.update_settings({'shot_profiles': config.shot_profiles})
            self.status.setText('Validated profiles applied to future settings only. Render explicitly; existing queued requests unchanged.')

    def shutdown(self):
        self.closing = True
        self.invalidate()
        self.poll()
        return not self.busy
