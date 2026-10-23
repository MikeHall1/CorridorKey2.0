# Run checkpoint — milestone 6, 2026-10-21

Five implementation/verification cycles completed. The preceding goal turn was verified progress (milestone 5), now archived in `checkpoints/milestone5.md`. This checkpoint ends the bounded run; the product goal remains active and unfinished. `.ralph/plan.json` is authoritative for task acceptance/evidence. No competing supervisor or background editor ran concurrently.

## Completed this run

1. Streaming CPU shot analysis: reduced adjacent display proxies, per-channel histogram distance/RGB change, native cut-candidate indices, bounded retained frames and optional per-frame diagnostics. Five tests cover static/ramp scenes, hard/adjacent changes, flashes, output ordering and cancellation. The heuristic is explicitly uncalibrated, can miss cuts and can flag flashes/exposure changes.
2. Integrated opt-in cut analysis into CLI/session/batch/runtime before context/RVM scheduling. Supplied and detected cuts are unioned without modifying requested settings. Runtime tests inspect actual CPU RVM state resets at detected boundaries and interrupted analysis records. No successful GPU inference occurred.
3. Maximum-gap scheduling supplements existing regular/final/correction/cut anchors. Adaptive mode explicitly requests neural fallback and promotes the resulting anchor for subsequent frames, logging motion/uncertainty reasons and planned/actual positions. On the synthetic flashing fixture this reduces additional Stage 2 requests from four to two; it is a fixture work count, not GPU speedup or quality evidence.
4. Added hold and causal propagation with current RGB and explicit fallback. Causal native contexts use only prior/current frames, each context owns its anchor, and later windows cannot change earlier batons. Causal planning omits final-frame anchors derived from future boundary/end knowledge; RVM uses one-frame chunks. Tests alter future RGB so it introduces a new automatically detected cut yet preserve every earlier alpha in both memory paths using context-sensitive neural substitutes. Causal flow uses two fields and logs anchor age/reliability deficit; hold uses no flow. Whole-clip precompute remains offline, not live/zero-startup processing.
5. Updated option/temporal/benchmark documentation, validated a 26-case local matrix example through the real configuration loader, recorded the actual unavailable-CUDA CLI result, and ran repeated CPU correctness plus compilation/whitespace checks.

## Verification

- Numeric CPU stack: **126 tests pass**, no skips, in each of three repetitions.
- Dependency-free stack: **53 pass, 73 explicitly skip** optional numeric checks, in each of three repetitions.
- Reports: `benchmark-results/cpu-adapters-milestone6.json` and `benchmark-results/cpu-stdlib-milestone6.json`.
- Full verification: `.ralph/runs/manual-check/verification.json` (unittest, compilation, whitespace), all successful.
- Actual CLI prerequisite: `benchmark-results/quick-m6-prerequisite.json` records failure with the CUDA requirement and null FPS. New options parse/snapshot; no neural CPU fallback was introduced.
- `benchmarks/suites/quick-key-matrix.example.json` declares 26 interval/approach/adaptive cases; `quick-key-warm.example.json` uses the existing three-repetition harness. Only configurations were validated here, not inference or an alternating A/B/quality study.

Real Farneback, OpenCV decode, EXR writes and memmap cleanup are covered locally; neural modules in runtime fixtures are deterministic substitutes. No real-model quality, GPU performance, target-platform FFmpeg export or Resolve compatibility is proven. Reports and generated fixtures remain local ignored artifacts. The temporary CPU environment's NumPy 2 deviation from runtime requirements is documented in `baseline.md`.

## Remaining and next action

Implement the ready `corrections` task: read `docs/corrections.md`, implement versioned authored matte/keep/remove/reset persistence and working conditioning/refinement, wire correction/reset frames into Quick Key anchors, and invalidate the full dependent shot range where recurrence/hint carry requires it. Do not expose unsupported correction semantics or limit invalidation to an arbitrary local neighbourhood. Keep tests/evidence CPU-compatible and preserve existing first/full-hint behavior.

`quick-key-adaptive` is locally implemented with explicit validation limits; `quick-key` remains in progress pending authored correction integration and external acceptance. Other ready tasks remain colour/high-bit/precision/refinement contracts, reference quality metrics/A-B reports, verified durable recovery, desktop UI, packaging and official SDK/worker contracts. Diagnostic GUI overlays, streaming/live execution, total RAM/disk budgets and broader I/O/timebase validation remain unfinished.

## External prerequisites

This Mac has no validated inference backend. Actual CUDA hardware, model/foundation weights and permitted development/reference/held-out footage remain missing, along with any training dataset for future training work. GPU visual/performance comparisons, real export/timing, Windows cleanup/package testing, distribution/licensing review and the Resolve SDK/Windows/NVIDIA host are not marked passed. No downloads, pushes, deployments, publications or media uploads occurred. Existing work remains uncommitted.
