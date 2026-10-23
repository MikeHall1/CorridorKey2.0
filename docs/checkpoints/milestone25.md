# Run checkpoint — milestone25, 2026-10-22

Five implementation/verification cycles completed; the full goal remains active and incomplete. The previous response prepared the follow-up goal without runtime changes (**no progress** for implementation). This run revalidated current source/ledgers and established a fresh464-test/207-file baseline before editing. Archived milestone24: `checkpoints/milestone24.md` and `.ralph/checkpoints/milestone24.json`. No competing supervisor was launched.

## Completed cycles

1. Added actual completed-render source-coordinate mapping and paired decode. Each export gets its own relative index; missing/incomplete metadata, incompatible geometry/FPS/source path and disjoint selections reject. Current exports must match recorded count/geometry.468 tests/full no-new-findings gate pass.
2. Connected **Selected render as synchronized A** to the latest-only background preview owner. A retains observed range/paths/view; B can be another selected completed export or the current source. Both thumbnails publish together. Invalid/new selections invalidate stale in-flight pairs. Generated AVI/EXR tests demonstrate source frame2 mapping to export1 versus export0, and actual source decode.470 tests/full gate pass.
3. Added optional core/CLI `start_frame` shared by video/sequence probes, runtime and correction binding. Explicit starts require begin; legacy null-start video modes remain. Full hints use the selected offset; all engine temporal indices remain selected-clip-relative. Real CPU exports in both memory paths verify selected source RGB/count.474 tests pass; first static gate failure was classified/archived, then corrected with0 new findings.
4. Added Source In/Out (exclusive), Mark In/Out and explicit Apply for future jobs, project roundtrip and translated export marks. Invalid ranges leave settings unchanged; old queued snapshots remain unchanged. Legacy middle/random selection clears explicit start.475 tests/full gate pass.
5. Fixed source/prepass/draft/hint readers ignoring decoder seek refusal: release capture and fail before decoding a mislabeled range. Real generated workflow renders two trims and compares them/source without modifying earlier outputs. Screenshot inspection found clipped long-path labels; compact indices/reserved height and full-path tooltips fix the observed layout. Regenerated capture inspected. Final476-test gate passes; repeated CPU benchmark recorded below.

Contracts: [source-ranges.md](source-ranges.md), [desktop.md](desktop.md), [options.md](options.md). No inference approximation/precision/backend default changed. No dependency installs, user environment modifications, large downloads, publication or upload.

## Verification

- Fresh baseline: `benchmark-results/m25-resume-baseline.txt` and archived `m25-resume-baseline/`.
- Per-cycle archives: `m25-cycle1/`, `m25-cycle2/`, `m25-cycle3/`, `m25-cycle4/`. Initial cycle3 failure retained in `m25-cycle3-first-failure/`; fixes preserve all cases/checkers. Initial final gate before screenshot label fix retained in `m25-final-before-labels/`.
- Final: `benchmark-results/m25-final-verification.txt`, `.ralph/runs/manual-check/verification.json`: success true, all4 checks return0. **476 tests PASS/no skips;210 Python files,0 syntax errors.** Runtime imports, requirements/VCS identity, installed dependency consistency, compilation and whitespace pass.
- Ruff520 findings; Pyright696 errors/17 warnings (713 diagnostics), **0 new fingerprints** against unchanged baseline. Repository is **not static-clean**. The104-error reduction removes inferred constructor-kwargs diagnostics by dynamically assigning named invalid fields in two negative-test methods before validation; same rejection cases remain. No checker weakening, ignore or baseline refresh.
- Comparison/preview/loader/window and new range/comparison/UI tests have focused Ruff/Pyright0 findings. Legacy runtime/video/config/infer findings remain visible in whole-repository reports.
- Existing dataclass instance/class and heterogeneous-set serializer guards/callers/tests inspected; contracts remain intact. Dependency-free desktop and inference help pass, including new CLI flag. Pylance editor diagnostics NOT VERIFIED.
- Actual `python -m corridorkey.benchmark --output benchmark-results/m25-cpu-benchmark.json` completed successfully:3 repetitions, median30.306669083889574s, min30.04973508301191s, max30.38384341704659s. All476-test repetitions pass. These are CPU correctness-suite elapsed times, not inference FPS, GPU speedup or neural-quality evidence; no controlled baseline/candidate speedup is claimed.

## Actual generated workflow

`benchmark-results/m25-generated-summary.json`, script `m25-demo.py`, log `m25-generated-demo-final.txt`, final artifacts `benchmark-results/comparison-m25-swjol2gk`:

- Actual Qt window, shared worker/session and native EXR writer with explicit **constant-alpha CPU fixture**, not pretrained inference.
- Explicit [1,5) and [2,5) source trims produce4 and3 frames. Synchronized source frame3 addresses A export2 and B export1. Source B addresses original source frame3. Labels/captures inspected locally after layout correction.
- Exactly two explicit renders; viewing/range operations add no jobs. Out-of-overlap frame5 rejects. First render EXR hashes remain unchanged. Owned worker cleanup has no error; preview thread stopped.
- No GPU FPS or neural-quality claim; real model visual acceptance remains NOT VERIFIED.

## Issues and remaining scope

HEALTH-VIEWER/TEST/TYPE/RANGE138–145 were classified before correction and resolved with preserved failed evidence. **HEALTH-FORMS-126 native Qt crash mechanism remains open**: these passing runs are not a native backtrace or target interactive acceptance. Existing broad static backlog remains.

Source synchronization is path/frame-index alignment, not authenticated source content or exact timestamp/VFR alignment. Current bytes are decoded; keep sources/exports stable. Explicit Out extending past EOF caps under the existing limit semantics. In at/past EOF rejects. Nonzero selection starts fresh RVM state; no direct-seek recurrence equivalence is claimed. Correction documents stay bound to their selected start/count and must match/rebind. UI EOF validation occurs in actual source probing, not guessed metadata. Fixed pinning remains available; no playback performance or blinded quality approval is implied.

Actual current prerequisites: `benchmark-results/m25-prerequisites.json` reports CUDA unavailable, all5 configured model/foundation asset paths absent, FFmpeg not on PATH. No large asset download was attempted. Permitted representative/held-out footage, target Windows packaged application and Resolve SDK/host acceptance remain missing or unverified. Production GPU performance, neural quality and Resolve integration remain **NOT VERIFIED**.

Next: build a maintainable target desktop packaging recipe with explicit dependency/asset/notices handling and smoke checks. Continue official Resolve SDK capability investigation/adapter work, total resource/disk budgets, retention deletion/portable backup, advanced corrections, high-bit video/managed colour, direct-seek/Stage2/writer recovery, controlled alternating/stress/held-out evaluation and actual GPU/Windows/host acceptance. Independent work remains actionable; goal is neither complete nor blocked.

Machine checkpoint `.ralph/checkpoint.json`; task ledger `.ralph/plan.json`; issues `.ralph/issues.json`.
