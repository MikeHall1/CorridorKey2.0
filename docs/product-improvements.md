# Product improvements: implementation and dependency map

The current user objective is [the ten-feature extension](product-improvements-brief.md).
The earlier [repository review](review-objective.md) is historical accepted local
work, not acceptance of these extensions or the full [development brief](development-brief.md).
Starting snapshot: clean `1c3f9f2`, branch `review/corridorkey-audit-20261023`,
verified origin `MikeHall1/CorridorKey2.0`. The existing compatible Python3.12/
NumPy1.26.4 environment remains available; the user's `.venv` is unchanged.
The unchanged starting-state 985-test/static evidence is reused rather than
repeating a baseline. New changes need their own integration verification.

| ID | Existing foundation | Required new work / dependency |
|---|---|---|
| EXT-01 | Asset presence, explicit structural audit, backend probe, shared job/worker and export inspection | Unified readiness, explicit short render and staged execution/decode/display evidence. First priority. |
| EXT-05 | Atomic A/B thumbnails, recorded source offsets, latest-only decoder | Playback clock/readers/loops/speed/zoom/pan and bounded diagnostics. Builds on explicit render. |
| EXT-02 | Motion flow/reliability, immutable correction documents/history | Tracked derived assets, cut/reliability policies, anchors, preview/commit and invalidation. Reuse playback inspection. |
| EXT-03 | Native decoding, trimap/refinement, correction import | Plate identity/alignment/exposure/screen field and reversible guidance. Reuse correction attachment. |
| EXT-04 | Correction import and optional-dependency conventions | Segmentation provider/click lifecycle, coarse preview/provenance and optional tracking. Requires explicit provider evaluation. |
| EXT-06 | Latest-only decode serials and supervised inference | Prioritized progressive scheduler, provenance, resource budget and state labels. Reuse playback. |
| EXT-07 | Motion arrays and diagnostic range/rerender preparation | Spatial maps/legend/provenance/navigation and correction actions. Reuse tracking/playback. |
| EXT-08 | Controlled cold/warm benchmark runs, recorded settings and identities | Configuration search, failures, recommendations, expiry and duration uncertainty. Reuse readiness. |
| EXT-09 | Strict absolute-path schema1 projects and safe transfer patterns | Portable inventory/relative paths, preview/collection and deliberate relinking with compatibility checks. |
| EXT-10 | Prepared-cache native adapter and shared-engine cache sealer | Supported host selected-range capture and preparation interaction; investigate SDK early in parallel. |

This map is based on read-only independent source audit, not reported feature
summaries. Directional stationary correction masks are not tracking. Thumbnail
inspection is not playback. Cold/warm benchmark controls alone are not automatic
calibration. Existing absolute-path projects are not portable manifests.

Readiness increment: **Readiness / first render…** opens without running work.
Explicit **Check readiness** runs local metadata/source/output checks off-thread;
asset integrity, backend, model loading, inference and output decoding/display
remain individually unverified where they have not executed. FFmpeg lookup does
not prove encoder capabilities; EXR does not require FFmpeg. Dependency versions
do not prove native imports or the pinned utils3d source identity. Permissions
are advisory and do not reserve space or writes.

**Prepare short render** snapshots the selected local source, absolute In and
1–120 frames into a unique output child. It preserves requested quality, backend,
precision and writer settings, rebases supplied cut/reset markers and starts fresh
temporal state. A first-frame hint cannot silently move; correction subranges
require deliberate rebasing through the existing diagnostic workflow. **Run
displayed snapshot** explicitly submits through the existing worker and queue.
Preparation creates no directories and changes no queued jobs or form settings.

The local report is an explicit fresh JSON file; exporting into source, hint,
correction-asset or selected model trees is refused. No automatic large copies,
downloads, installation or environment changes are introduced.

EXT-01 now has first-run introduction, explicit backend/encoder metadata probes,
matching engine-reported execution and selected output decoding/Qt presentation.
Matching failed/cancelled engine records now preserve contextual checkpoints and
partial work separately from success. Remaining inspection roles and actual
CUDA/local-footage acceptance remain pending. See [readiness](readiness.md). The other workflows remain incomplete; initial sequence playback is described below.
Required external evidence remains CUDA/pretrained weights/permitted footage,
optional segmentation provider/hardware, Windows packaging and actual Resolve
SDK/host execution. Nothing here establishes quality or speed improvement.

Focused validation: `python -m unittest tests.test_readiness tests.test_desktop_readiness -v`.
Full CPU/static validation uses `python scripts/ralph_loop.py verify --python PATH`
(verification only; never launch a second implementation supervisor).


Playback increment: select a completed source-image-sequence render with EXR
exports, choose Source/Alpha/Foreground/checker/solid composite and **Play
sequence**. Source In/Out fields bound the loop; blank values use the selected
common range. Speeds are¼×,½×,1×,2×,4×. Synchronized A/B uses the recorded source
coordinate and independent export offsets. Displayed source frame/time is
updated only for accepted decoded pixels. Frame/fps is declared sequence time,
not observed video PTS. Manual preview, navigation/view/selection/settings changes
pause and fence obsolete results. The existing background decoder remains the
sole owner, with one pending playback token; no new decode thread or cache.

The existing50ms poll admits at most20poll opportunities/second, so playback may
drop opportunities even for24fps sources. Counters describe scheduling/buffering,
not calibrated decoder performance. Readers reopen for each frame. Video/VFR,
persistent readers/cache and photographic backgrounds remain
pending; existing checker and solid colour backgrounds work. No real-time claim.

Source-only sequence playback now works without a completed render. Source → Play
prepares a bounded frozen natural-order path/stat index off-thread on the sole
existing decoder; each frame reads its indexed file without directory rescans.
Explicit Refresh source index reinspects membership on next Play. Entries/frames
are bounded at100,000 and path bytes at32MiB; stat checks refuse changed indexed
files, without claiming source-content authentication. Declared FPS/absolute
offsets, pause/resume, loops, cancel and stale-preparation fences are tested.
Native decode memory is unchanged; photo backgrounds/video/VFR remain pending.
A/B surfaces share decoded-pixel zoom and normalized source centre. Wheel zoom,
left-drag pan and visible Zoom in/out/Fit controls alter display only. The label
explicitly states decoded-preview scale (normal cap1280pixels), not native-pixel
inspection. Different-sized views fit using the smaller available scale; each
retains only its bounded pixmap. Six Qt tests cover transforms, source mapping,
fit, native child destruction and immutable image snapshots.

Thirteen actual PNG/nativeEXR offscreen tests exercise playback, pause/manual seek,
loops, speed, A/Bsource3→Aexport2/Bexport1, stale native decode retirement and
source preservation. The headless planner has12deterministic tests.

Independent readiness review inspected Git blobs `readiness.py`
`dc426f1b487c7042d7d21d33882c5161860da373`, `desktop_readiness.py`
`16b0770b805772e1f0d59f1559607d2f42a3d74f`, and pre-playback window
`60f51b6b972a06a456c5fb3b406b79d4d8ad441d`. Main subsequent self-review found
HEALTH-READINESS-256 report path-list growth (2/20/200paths after1/10/100 fake
submissions). The fix copies the scan dependency list and retains/deduplicates
only current render protection. A100-render regression verifies bounded retention;
this is fixture evidence, never actual neural inference.


Tracked-correction initial design audit (EXT-02, now implemented as described in
[tracked corrections](tracked-corrections.md)): appending
ordinary per-frame keep/remove entries beside the original stationary range
would double-apply strength and retain the stationary mask. Use explicit schema3
track metadata to replace participating anchors' final constraints within the
tracked span at their original composition position, preserving authored entries
and mask bytes. Keep schema1/2 meaning unchanged. Apply derived masks to final
constraints initially; existing authored neural hints remain separate.

A bounded tracking preview bundle should bind source, anchor IDs/revisions/hashes,
shots/resets/cuts, motion/decode contract and settings to separately stored float32
EXR masks/diagnostics. Hash that input binding without the accepting revision to
avoid self-reference. Preview never mutates history; explicit Accept creates one
undoable revision. Anchor/boundary edits invalidate derivatives. Range crossing
cuts/resets refuses; unreliable propagation stops/requests another anchor by
default, with visibly explicit approximation as an alternative. Backward tracking
needs its own bounded reader because DraftSourceReader is increasing-only. Reuse
motion.propagate_alpha/interpolate_alpha; refuse trimap-class interpolation until
its semantics exist. Minimum connected acceptance includes translating flow,
keep/remove once-only ordering, save/reopen/undo, cancellation/publication failure,
identity invalidation and bounded memory/disk. Connected preview/inspect/accept controls now use this contract; video/production-footage acceptance remains pending.


Independent playback review found HEALTH-PLAYBACK-257 (resume returned to the
unchanged still-navigation field) and258 (obsolete decoder errors stopped newer
playback). Both were reproduced before correction. Pause now freezes the last
accepted displayed frame and resumes only an unchanged binding; explicit manual
preview/navigation resets it even when the still index is unchanged. Completion
always retires ownership, while only current-generation errors stop/report.
Thirteen real-decoder cases cover these paths, including paused buffering and
old Alpha failure followed by valid newer Source pixels. Independent final
review passes13cases2.603s with no remaining blocker in this increment.
