# Desktop, video, packaging and native OFX audit — lane C

Date: 2026-10-24. Read-only application audit; only this report was written.
Target: dirty working tree based on `7d808e61d8a42d77e30792ded6911aabca96a08f`.
The coordinator's `evidence/initial-state.json` records the starting file identities.
No application files, tests, authoritative ledgers or runtime defaults were changed.

## Code Review Summary

**Files Reviewed:** 25 primary source/build files, plus contextual documentation,
callers and test fixtures (scope below; not a claim that every desktop module was
exhaustively audited).
**Total Issues:** 4 confirmed root-cause findings.

### By Severity

- CRITICAL / P0: 0
- HIGH / P1: 1
- MEDIUM / P2: 3
- LOW / P3: 0

**Recommendation: REQUEST CHANGES.** In particular, invalid timing must not become
fabricated source FPS, and cancellation-delivery errors must not disappear.
Windows/Resolve unavailability is separately recorded, not counted as a bug.

## Scope and specification compliance

Primary source/build inspection:

- `corridorkey/desktop.py`, `desktop_window.py`, `desktop_bridge.py`,
  `desktop_loader.py`, `desktop_playback.py`, `desktop_preview.py`,
  `desktop_comparison.py`, `desktop_viewport.py`, `desktop_state.py`,
  `desktop_corrections.py`, `desktop_progressive.py`, `desktop_ofx_capture.py`,
  `desktop_bundle.py`, `desktop_recipe.py`, `desktop_background.py`,
  `desktop_resources.py`, `app_process.py`, `playback.py`, `video_io.py`,
  `correction_canvas.py`, `correction_editor_service.py`;
- `native/ofx/adapter.cpp`, `cache.hpp`, `capture.hpp`, `CMakeLists.txt`.

Contextual inspection included readiness/calibration/portable-project shutdown
and submission boundaries, `session.py`, `subject_guidance_process.py`, progressive
scheduler/recipe interfaces, bundle entry, native host fixtures and the desktop,
playback, progressive, capture and packaging test fixtures. Read the supplied
AGENTS contract, current product brief, architecture/status/options/code-health,
issue-ledger context, development-brief requirements, and relevant packaging,
playback, subject-click, OFX capture and progressive-preview documents.

This is a scoped audit of existing functionality as well as the dirty additions,
not a claim all findings originated in the latest diff. UI-001 affects the new
capture dialog and existing dialogs; UI-002–004 are existing code-path defects.

### Supported execution paths / feature classification

| Area | Observed implementation | Classification and limit |
|---|---|---|
| Source desktop | Optional Qt entry, lazy supervised worker, bounded commands/events, immutable queued job snapshots, terminal/lost ownership | Connected source implementation; actual offscreen Qt and CPU-fixture evidence, not CUDA/Windows acceptance |
| Readiness | Explicit checks and reviewed short-render submission through ordinary worker; separate completed decode/display | Connected; real model loading/inference/quality requires actual assets/hardware; no opening-time inference |
| Frame inspection / playback | Source and EXR exports; source offsets mapped explicitly; paired A/B publication; shared zoom/pan; declared sequence FPS; bounded reusable indexes | Connected image-sequence subset; video/VFR playback explicitly refused, not supported timestamp playback |
| Corrections / guidance | Background editor owner; native brush coordinates, dirty guards, immutable saves, tracked-mask/clean-plate/subject controls; heuristic diagnostics | Connected source workflow; optional provider actual execution and photographic quality remain unverified; Escape close contract incomplete |
| Progressive previews | Explicit draft/final ladder, bounded scheduling/cache, existing worker, stale-result fences, separate scratch/export | Partial integration: live playback focus and source/export coordinate mapping broken (UI-003); silent cancellation failure (UI-004) |
| Video I/O | OpenCV uint8 assumed-sRGB source; explicitly linear sequences retain float path; FFmpeg writer; EXR precision choice; coverage separated from RGB transfer | Legacy video path is not HDR/high-bit/VFR acceptance. Missing FPS currently invents 24 (UI-002). No measured timestamp correspondence established |
| Projects / import-export | Strict schema/wire validation, protected source/correction paths, explicit portable workflow; queued work not replayed by opening | Connected local paths; broader cross-platform collection/relink core reviewed in other lane, not certified here |
| OFX | Cached effect; optional progress V1/V2; explicit opaque linear709 upstream capture; signed host offset; fresh bundle, per-frame leases, checksums and COMPLETE-last commit | Experimental prepared-cache/capture implementation, **not live neural OFX**. Companion submission uses existing worker; manual cache attachment; actual Resolve acceptance absent |
| Windows bundle | Target-gated recipe, source-only engine collection, explicit frozen module dispatcher, no weights/FFmpeg bundled | Recipe only, not built/validated Windows release. Frozen subject guidance explicitly raises unavailable; it is not merely an unexecuted supported path |

Full brief compliance is not established: video timestamp handling, real GPU and
quality validation, native Windows packaging and actual Resolve acceptance remain
open. The full brief also asks for an optional developer-mode interface, whereas
`desktop_window.py:180` unconditionally adds the benchmark button; this is a
small remaining specification gap, not included among the four prioritized bugs.
No architecture redesign was undertaken; final architecture coverage belongs to
the coordinator's independent architecture/review lane.

## Findings

### UI-001 — P2 / MEDIUM: Escape skips the operation-owning close path

**Locations:** `corridorkey/desktop_ofx_capture.py:429–436`,
`corridorkey/desktop_progressive.py:444–447`; same missing rejection routing is
present in `desktop_corrections.py:872–885` and `desktop_background.py:89–92`.

**Evidence/reproduction:** Actual PySide6 offscreen widgets, not a guessed Qt
behavior: submit-state capture dialog, then `QTest.keyClick(dialog, Qt.Key.Key_Escape)`.
Observed `visible=False`, `closing=False`, `rendering=True`, cancellation signals
`[]`. On enabled progressive panel, Escape produced `visible=False`,
`enabled=True`, `timer_active=True`. No real model was started by this probe.

**Impact/root cause:** These dialogs implement cancellation, stale-result fences,
and dirty-draft checks only in `closeEvent`. QDialog Escape calls `reject`, bypassing
those overrides. The window disappears while inference/file/provider work remains
active; progressive scheduling can continue hidden. The correction dirty guard is
also bypassed, although the parent still retains its draft (no proven data loss).
This conflicts with the implemented window-close cancellation semantics.

**Fix:** Route `reject()` through the existing `close()` path in operation-owning
dialogs, as `desktop_readiness.py:612–614` already does. Retain existing ownership
and dirty guards; do not force-close/kill operations or introduce a second owner.

**Effort:** S (targeted dialog changes and coverage). **Confidence:** high; actual
Qt reproduction. **Regression test:** Escape during gated capture I/O/render,
progressive work and a dirty correction draft; assert identical cancellation,
visibility and retirement behavior to the title-bar close path.

### UI-002 — P1 / HIGH: Missing video FPS silently becomes an invented 24 fps

**Location:** `corridorkey/video_io.py:211` (downstream metadata:252;
FFmpeg raw input rate:818).

**Evidence/reproduction:** A decoder fixture reporting valid 64×64 geometry,
10 frames and FPS `0` was passed to the real `probe_video_meta`. It succeeded and
printed `fps 24.00`; returned `native_fps == 24.0`. The later positive/finite
validation at line217 cannot catch the missing value because `or 24.0` already
replaced it. No real clip decode/inference was claimed.

**Impact/root cause:** Unknown timing is represented as observed native timing.
Writers, duration/range metadata and hint FPS checks then consume an arbitrary
rate. A 30/60-fps source whose decoder cannot report timing can be exported at
24 fps with wrong duration and downstream alignment. This is a silent fallback
that repairs neither the primary timing contract nor its diagnostics.

**Fix:** Remove the fallback and reject unavailable/nonpositive/nonfinite FPS
with an actionable message. An explicit declared override could be a separately
specified future feature; do not add another inferred/default rate. Keep declared
image-sequence FPS behavior separate.

**Effort:** S. **Confidence:** high; executed exact branch. **Regression test:**
zero/missing/NaN FPS refuses before writer/model work; valid fractional FPS remains
unchanged. Audit metadata consumers so inferred timing cannot masquerade as native.

### UI-003 — P2 / MEDIUM: Progressive focus is not the displayed source playhead

**Locations:** `corridorkey/desktop_progressive.py:188`, `:323–329`, `:387–393`;
`corridorkey/desktop_playback.py:363–364`;
`corridorkey/desktop_window.py:1570–1605`.

**Evidence/reproduction:** Using existing real Qt widgets, local generated frames
and the existing explicitly test-only CPU worker fixture, render preview frame1,
start Source playback, and wait for displayed frame3. Observed:

```text
playback_displayed_frame=3
workspace_frame=1
progressive_desired_frame=1
scheduler_frame=1
scheduler_playing=False
```

A second probe configured source range `[2,4)` with the normal workspace frame0,
then captured a progressive recipe. Calling its completed I/O poll raised uncaught
`ValueError: source_frame requires an exact integer in [2,3]` at line327, leaving
status `Capturing bounded source/settings identity; no model loaded yet.`, enabled
true and failure unset at that boundary.

**Impact/root cause:** The progressive adapter treats the manual spinbox as an
absolute source playhead. Playback only updates its own displayed-frame state;
export stills use output-local indices. Playback never causes focus or playing
state updates, so advertised upcoming-frame scheduling is not connected to actual
playback. Trimmed export/local indices can address the wrong source frame or raise
outside the poll error boundary.

**Fix:** Bind progressive focus to a validated absolute source coordinate: use
accepted playback frame/time and play/pause changes when playing, otherwise map
manual export-local indices through the selected render offset. Refuse incompatible
source bindings explicitly. Validate the initial focus before accepting the recipe
and handle capture completion failures within the existing visible error path;
do not silently clamp to a different frame.

**Effort:** M (one boundary repair plus integration coverage). **Confidence:** high;
both behaviors reproduced without pretrained inference. **Regression test:** live
source playback updates scheduler focus/playing; play/pause updates even without
spinbox edits; partial-render export index0 maps to its source start; out-of-range
focus produces an actionable UI error and no unhandled Qt callback exception.

### UI-004 — P2 / MEDIUM: Progressive cancellation delivery failure is swallowed

**Location:** `corridorkey/desktop_progressive.py:220–226`, early-return:380–381.

**Evidence/reproduction:** Real panel, active `worker_id`, a bridge whose `send`
raises `queue.Full`, then disable. After five `poll()` calls:

```text
cancel_attempts=1
cancel_sent=False
worker_id='running-preview'
visible_status='Disabled; waiting for owned preview cleanup.'
```

**Impact/root cause:** `_cancel_worker` catches and discards every delivery error.
Disabling also clears the recipe; subsequent polls return before another attempt.
The active job therefore has no delivered cancellation request, but the interface
suggests retirement is in progress. A long-running native/inference job can keep
using resources and delay an export. Ownership is retained correctly, so this is
not a proven process leak, but retained ownership does not substitute for delivering
or accurately reporting cancellation.

**Fix:** Preserve and visibly report delivery failure; retain the existing target
and allow an explicit retry or bounded retry of the same cancellation request
without releasing ownership. The new capture dialog's visible, retryable cancel
path (`desktop_window.py:821–828`) is an existing local pattern. Do not silently
stop the whole worker or create an alternate inference path.

**Effort:** S. **Confidence:** high; executed queue-full failure path. **Regression
test:** first cancellation delivery fails, second succeeds; terminal ownership is
retained throughout; unrecoverable send failure remains visible.

## Reproducible local proofs

Use the existing compatible interpreter, without installing anything:

```sh
PY=/private/tmp/corridorkey-compatible-venv/bin/python
QT_QPA_PLATFORM=offscreen "$PY" - <<'PY'
import queue, threading
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from corridorkey.desktop_ofx_capture import OfxCaptureDialog
from corridorkey.desktop_progressive import ProgressivePreviewPanel
app = QApplication([])
c = OfxCaptureDialog('{}', '{}'); events = []
c.cancel_requested.connect(events.append); c.mark_submitted('job'); c.show()
QTest.keyClick(c, Qt.Key.Key_Escape)
print('capture Escape', c.isVisible(), c.closing, c.rendering, events)
c.observe({'kind':'cancelled','request_id':'job','payload':{}}); c.close()
w = QWidget()
w.bridge = SimpleNamespace(send=Mock(side_effect=queue.Full('capacity')), done=threading.Event())
p = ProgressivePreviewPanel(w); p.timer.stop(); p.enabled.setChecked(True)
p.show(); QTest.keyClick(p, Qt.Key.Key_Escape)
print('progressive Escape', p.isVisible(), p.enabled.isChecked())
p.worker_id = 'running-preview'; p.enabled.setChecked(False)
for _ in range(5): p.poll()
print('cancel', w.bridge.send.call_count, p.cancel_sent, p.status.text())
p.worker_id = None; p.shutdown()
PY

"$PY" - <<'PY'
from unittest.mock import patch
import cv2
from corridorkey.video_io import probe_video_meta
class Capture:
    def isOpened(self): return True
    def get(self, key):
        return {cv2.CAP_PROP_FPS:0, cv2.CAP_PROP_FRAME_WIDTH:64,
                cv2.CAP_PROP_FRAME_HEIGHT:64, cv2.CAP_PROP_FRAME_COUNT:10}[key]
    def release(self): pass
with patch('corridorkey.video_io.cv2.VideoCapture', return_value=Capture()):
    result = probe_video_meta('no-fps.mp4', num_frames=10, shape_quantum=64,
        native_resolution=True, start_mode='begin', seed=1, model_scale=1,
        linear_brightness=1, linear_contrast=1, linear_contrast_pivot=.18)
    print(result['native_fps'])
PY

PYTHONPATH=tests QT_QPA_PLATFORM=offscreen "$PY" - <<'PY'
from test_desktop_progressive import ProgressiveDesktopTests
ProgressiveDesktopTests.setUpClass(); t = ProgressiveDesktopTests(); t.setUp()
try:
    t.configure(); p, w = t.panel, t.window
    p.enabled.setChecked(True); p.begin()
    t.until(lambda:p.scheduler is not None and p.scheduler.snapshot(1).status=='final')
    w.view.setCurrentText('Source'); w.playback.play()
    t.until(lambda:w.playback.displayed_frame==3)
    print(w.playback.displayed_frame, w.frame.value(), p.desired_frame,
          p.scheduler._frame, p.scheduler._playing)
    t.stop()
    p.timer.stop(); w.timer.stop()
    w.update_settings({'start_frame':2,'num_frames':2}); w.frame.setValue(0)
    p.enabled.setChecked(True); p.begin(); p.io_thread.join(5)
    try: p.poll()
    except Exception as error: print(type(error).__name__, str(error))
    p.enabled.setChecked(False)
finally:
    t.doCleanups()
PY
```

## Diagnostics and limitations

Executed locally:

- Above isolated Qt/CPU-fixture and metadata proofs: reproduced as reported.
- `clang++ -std=c++17 -Wall -Wextra -Werror -pedantic -fsyntax-only -I third_party/OpenFX/include native/ofx/adapter.cpp native/ofx/host_test.cpp native/ofx/capture_host_test.cpp`: **PASS**. This covers native syntax/header diagnostics, not Windows linkage or host behavior.
- AST parse of modified `desktop_window.py`, `desktop_ofx_capture.py`,
  `desktop_bundle.py`, `video_io.py`: **PASS**.
- `python -m ruff check --output-format json` on those four files: exit1;
  desktop files zero findings; video_io five findings (F401/I001/RUF046).
  Not a clean repository lint claim. Coordinator full static report
  `evidence/static-final/health.json` reports success=true and no new findings:
  512 Ruff / 701 Pyright diagnostics (684 errors + 17 warnings). The initial
  `evidence/static/health.json` failed with 3 new Ruff findings and is superseded;
  do not cite that initial report as a passing gate. Existing backlog is not clean.
- Full suite was intentionally not repeated by this lane. Coordinator owns its
  complete applicable CPU/static verification and sandbox-boundary disposition.

Root-cause guard: UI-002 substitutes success-shaped timing for missing evidence;
UI-004 suppresses cancellation diagnostics. Neither is acceptable as a hidden
compatibility fallback. No hardcoded secrets, shell construction/injection, neural
precision downgrade or source overwrite was confirmed in this scoped inspection.
That statement is not a security certification of uninspected modules.

Native capture's V1/V2 progress selection is a narrow explicit suite-version
boundary, not an inference fallback. Missing progress refuses capture while
existing cache rendering remains independently available. Host image handles are
leased per frame/action; previous CacheFile/CacheStart are not mutated by capture.
Bound cache rendering revalidates all selected source frames for every output
frame; this deliberately conservative O(range) validation is not a demonstrated
real-time path. Architecture/performance tradeoffs require the independent lane,
not an unsupported optimization or weakened source witness in this audit.

**Not checked:** actual Windows/MSVC/PyInstaller binary, native GUI/accessibility,
CUDA/pretrained models, optional SAM2 execution, representative video/HDR/VFR,
real Resolve viewer/export/proxy/retime/persistence and reference-footage quality.
**Risk:** local fixtures and compilation do not satisfy any of those external
acceptance criteria; full product acceptance remains pending.
