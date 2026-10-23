# Run checkpoint — milestone 5, 2026-10-21

Five implementation/verification cycles completed. This is the requested run boundary, not product completion. The preceding turn was verified progress (milestone 4); its checkpoint is archived in `checkpoints/milestone4.md`. The task ledger is `.ralph/plan.json`; the machine-readable resume point is `.ralph/checkpoint.json`. No standalone supervisor or competing editor ran concurrently.

## Completed this run

1. Implemented CPU offline bidirectional matte propagation with target-to-anchor flow, reverse consistency, RGB residual weighting and explicit uncalibrated reliability/disagreement/uncertainty. Seven motion tests include actual Farneback on a textured translation and deterministic known-field scaling checks.
2. Added bounded native-rate frame assembly with current source RGB, two retained anchor sources, shot-local brackets, look-ahead and explicit stop/approximate/additional-neural callbacks. Four CPU tests verify counts, signed/HDR RGB, fallback and interval-1 no-motion semantics.
3. Integrated Quick Key through the shared CLI/session/batch/runtime. Sparse Stage 1 consumes contiguous native contexts; Stage 2 predicts anchors; existing full-rate interval 1 bypasses the draft path. Supplied shot/reset boundaries reset carry and chronological RVM. Real CPU decode/maps/EXR export with neural substitutes checks both memory paths, extra neural fallback and interval-1 equality.
4. Hardened gap reads, mapping/error/cancellation cleanup and native source precision. Sparse readers avoid retaining/preprocessing unused gaps; low-VRAM Stage 2 gathers only targets. RVM reset tests inspect actual chronological feature values across chunk splits. Draft foreground retains float32 signed HDR without a model roundtrip. Per-frame JSONL traces identify rendered attempts, brackets, uncertainty and fallback. Tests cover carried-hint reset, cancellation, partial failure and labelled approximations.
5. Documented settings, colour/temporal/resource/cost limits and unfinished work; verified the actual missing-CUDA CLI failure; ran repeated CPU correctness reports and final compilation/whitespace verification. Final review corrected preview linear-cutout despill-before-clipping, with a dedicated HDR regression test.

## Verification and limits

- Numeric CPU environment: **108 tests pass**, no skips.
- Dependency-free environment: **51 pass, 57 explicitly skip** optional numeric checks.
- Three repetitions each: `benchmark-results/cpu-adapters-milestone5.json` and `benchmark-results/cpu-stdlib-milestone5.json`.
- Full verification: `.ralph/runs/manual-check/verification.json` (unittest, compilation, whitespace).
- Actual CLI prerequisite report: `benchmark-results/quick-prerequisite.json` is failed with a CUDA requirement and null FPS. This is expected environment rejection, not successful inference.

Reports and tiny generated fixtures remain local ignored artifacts. Tests use deterministic neural substitutes; real Farneback, OpenCV decode, memmaps and EXR writers are exercised. No real model output, FFmpeg encode, GPU speed or professional visual quality is proven. The temporary CPU environment's test-only NumPy 2 deviation remains documented in `baseline.md`.

The 9-frame/interval-4/radius-1 fixture evaluates 7 Stage 1 frame instances and 3 Stage 2 frames, versus 16/9 full-rate. Four neural fallbacks raise sparse work to 19/7. These are actual fixture work counts, not a speedup claim. Low-VRAM IID precomputation still covers all nine frames. All anchors/RVM finish before draft output, so the bracket look-ahead metric excludes whole-clip startup dependency. Source-RGB foreground substitution is explicitly labelled. Uncertainty thresholds remain uncalibrated.

## Remaining tasks and next action

Resume `quick-key`: implement automatic shot-boundary analysis and integrate cut events into the scheduler/RVM reset path. Current cuts/resets must be supplied explicitly. Then implement adaptive anchor insertion/maximum-gap policy, hold/causal comparison and authored correction conditioning/invalidation. Keep native context and recurrent chronology explicit, count extra work, and retain the original full-rate interval-1 route.

Quick Key remains in progress: correction points exist in the scheduler but authored corrections are not wired into runtime, automatic/adaptive/causal modes and diagnostic overlays are unfinished, and visual acceptance is unavailable. Other ready tasks include high-bit/colour/refinement controls, quality metrics/A-B reports, durable verified recovery, desktop UI, packaging and official SDK/worker contracts.

## External prerequisites

This Mac has no validated neural inference backend. CUDA hardware, model/foundation weights, permitted development/reference/held-out footage and real export validation remain missing. No training dataset is available for future training work. Windows cleanup/package tests and licence review, plus the Resolve SDK/Windows/NVIDIA host are external requirements. None is marked passed. Python cancellation cannot forcibly interrupt current native I/O/model/GPU calls; total RAM/disk budgets and variable-rate timestamp validation remain unfinished.

No media was uploaded, no code pushed/published/deployed, and no assets or footage downloaded. Existing work remains uncommitted.
