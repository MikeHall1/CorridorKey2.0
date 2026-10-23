# Run checkpoint — milestone 7, 2026-10-21

Five implementation/verification cycles completed. The preceding goal turn was verified progress (milestone 6), archived in `checkpoints/milestone6.md`. This run boundary does not declare the complete product done. The active chat remains the implementation owner; no competing supervisor ran. `.ralph/plan.json` and `.ralph/checkpoint.json` hold the resumable ledger and next task.

## Completed this run

1. Strict immutable schema-1 correction documents bind source content/selected timeline/native geometry and ordered local hashed assets. Added canonical portable identity, atomic save/reopen, revision history/undo/redo and conservative whole-shot invalidation. Five dependency-free format/history tests pass.
2. Native raster operators implement real soft-strength matte/keep/remove math, ordering, static directional ranges, grayscale/RGBA coverage and uint16/float precision validation. Unknown partial hints are not fabricated. Six CPU raster tests pass; at most two native decoded masks are retained.
3. Shared CLI/session/batch/runtime integration adds `--corrections` / `settings.corrections_path`, preflight hashes before model load, supported authored hint/auxiliary carry conditioning, float32 output constraints, shot/reset segmentation, actual RVM reset behavior and mandatory Quick Key correction anchors. Both memory paths and corrected interval-1 parity are exercised with real decode/maps/EXR and deterministic neural substitutes.
4. Hardened preflight/session reuse, benchmark identities, incompatible geometry/cut detection, cancellation and ordered strength semantics. Raw neural alpha, both-direction flow conditioning and final output constraints are separate: forward/backward static raster values cannot enter flow state; neural frames receive final ordered constraints exactly once. Eight correction runtime tests pass, including mixed operations and all draft approaches. Hash/schema failures preserve an untouched healthy session; ranges crossing effective cuts fail before pipeline stages.
5. Documented accepted semantics and limits, generated a tiny local correction prerequisite fixture, recorded the actual missing-CUDA failure with validated correction identity/asset refs and null FPS, and ran repeated CPU correctness/compilation/whitespace checks. No neural CPU fallback was introduced.

## Verification

- Numeric CPU environment: **147 tests pass**, no skips, in all three repetitions.
- Dependency-free environment: **60 pass, 87 explicitly skip** numeric checks, in all three repetitions.
- Reports: `benchmark-results/cpu-adapters-milestone7.json` and `benchmark-results/cpu-stdlib-milestone7.json`.
- Full verification: `.ralph/runs/manual-check/verification.json` records actual unittest, compilation and whitespace checks.
- Actual CLI prerequisite record: `benchmark-results/corrections-m7-prerequisite.json` fails with a CUDA requirement, validated correction identity/asset refs, preflight timing and null FPS. It is not successful inference.

Reports and generated tiny masks/fixtures remain local ignored artifacts. Runtime tests use deterministic neural substitutes; actual OpenCV decode, coverage math, source/asset hashing, EXR writing, memmap cleanup and RVM state plumbing are tested. No trained-model visual improvement, GPU speedup, real encoded timing or Resolve compatibility is proven. The temporary CPU environment's NumPy 2 test-only deviation remains documented in `baseline.md`.

## Semantics and limits

Ordered matte/keep/remove operations have exact direct alpha effects and meaningful soft strength. Both-direction corrections participate in existing hints/auxiliary carry/Quick Key flow; partial keep/remove needs a known hint base. Forward/backward ranges repeat static native rasters as direct output constraints, not motion tracking or reverse neural inference. Direct alpha edits do not recompute reconstructed foreground colour. Every active authored/reset frame is a required anchor. Changing temporal context/anchor placement can affect the whole shot beyond a direct mask range; invalidation conservatively returns full affected declared shots.

Current jobs recompute rather than reuse durable output caches. The invalidation/history API supports future editor/recovery integration; durable checkpoint recovery, a multi-revision project journal, brushes/trimaps and the desktop correction UI remain unfinished. Whole-clip Quick Key staging is still offline. Scalar FPS binding does not validate variable-rate timestamp alignment. Actual Windows and native/GPU cancellation behavior remain external tests.

## Next action and remaining work

Proceed with the ready `colour-precision` task. Audit the input/colour/association/invalid-value contracts, implement higher-precision input handling and only refinement controls with working effects, and add HDR/coverage fixtures without silent transfer/precision changes. Then implement reference quality metrics/A-B evaluation, verified durable recovery, desktop UI/queue/editor, packaging and official Resolve SDK/worker contracts.

`corrections` is locally implemented with explicit limits. `quick-key` is now recorded as external acceptance: its local runtime/correction integration is implemented, but actual model quality/performance and encoded timing still require external evidence. No change narrows the complete brief or declares whole-goal completion.

## External prerequisites

This Mac has no validated neural inference backend. CUDA hardware, model/foundation weights and permitted development/reference/held-out footage remain missing; no training dataset is available for future training work. Real GPU quality/performance comparisons, native encoding/timing, Windows cleanup/package smoke tests, distribution/licensing review and the Resolve SDK/Windows/NVIDIA host are not marked passed. No footage/assets were downloaded, no media uploaded, and no code pushed/published/deployed. Existing changes remain uncommitted.
