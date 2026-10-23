# Run checkpoint — milestone 14, 2026-10-22

Five implementation/verification cycles completed; no second loop/supervisor started. Previous turn is classified **progress** from milestone13 source/checkpoint and the fresh280-test baseline. Milestone13 is archived in `checkpoints/milestone13.md` and `.ralph/checkpoints/milestone13.json`. Work stays local and uncommitted; no push/publish/upload, source/weight downloads, dependency installs or user `.venv` modification occurred.

## Completed cycles

1. Established fresh full baseline (280 tests,131 source files, import/dependency/compile/whitespace pass; Ruff523/Pyright826errors17warnings, no new findings). Added strict canonical typed IID frame identities and generic safe bundle identity validation. Four new tests;284 tests passed. Evidence `benchmark-results/m14-cycle1-*`.
2. Added durable normalized float32 tensor adapter, original float32/bfloat16 proxy bit hashing, geometry/channel checks, closed independent mmap copies, owned tensor staging, budget rejection and RAM admission of durable hits. Five focused tests passed. The broad cycle2 check overlapped the following integration edits and is **not** a passing independent closure: it reproduced9 low-VRAM cleanup failures and new diagnostics, resolved under031/032 and superseded by cycle3/final gates. Evidence remains `m14-cycle2-*`.
3. Bound actual session-loaded model assets and current dependency identity before runner access. Diagnostics carries the adapter but serializes counters only. Both normal Hann and low-VRAM prepass share the existing cache boundary, preserve original input/feature precision and record actual evaluated work. Fixed ResourceScope API integration, scope narrowing, import layout and fixture contracts after classification.291 tests and no-new-finding gate passed; `m14-cycle3-*`.
4. Added working recovery `--durable-foundations` (default off), policy identity, failure diagnostics and fresh-session failed-job reuse. Full292 tests and no-new-finding gate passed; `m14-cycle4-*`. An actual2-frame CPU export fixture seals1 frame before failure, resumes with1 hit/1 new foundation evaluation, loads a fresh session and preserves the old output path.
5. Reproduced and fixed payload mutation between lookup validation and mmap loading: now rehash after independent copy/map close before accepting a hit. Added normal/low-VRAM equality, tensor-write disk-full/cancel cleanup and actual child exit75/reopen coverage. Fixed new fixture typing/style after classification. Final full gate passes.

## Final actual verification

- **297 tests PASS**, no skips; `.ralph/runs/manual-check/check-0.txt`.
- Compilation and whitespace PASS; `.ralph/runs/manual-check/verification.json`, all4 command return codes0.
- **134 Python files /0 syntax errors**; selected runtime imports and declared requirements/VCS/installed dependency checks PASS.
- Ruff **520 findings**; Pyright **822 errors /17 warnings**. **Zero new diagnostic fingerprints** against unchanged reviewed baseline; repository is **not static-clean**. No suppressions, weakened configuration or baseline refresh.
- Focused feature/recovery modules and new identity/adapter/runner test files: Ruff/Pyright **zero findings** (`benchmark-results/m14-focused-*`). Pylance editor **NOT VERIFIED**. JS/TS/native diagnostics not applicable to current source scope.
- CLI help actually passes with `python3 -S` (no site packages/model import), including the new option; `benchmark-results/m14-cli-help-stdlib.txt`.
- Same isolated Python3.12.15, NumPy1.26.4, torch2.9.1/torchvision0.24.1, OpenCV4.11 and pinned Ruff0.17.1/Pyright1.1.414. Temporary env may need recreation. No installation in this run.
- Final command `benchmark-results/m14-final-verification.txt`; run-start gate preserved in `benchmark-results/m14-start-baseline/checks/` and `m14-start-verification.txt`.

## Actual fixture work counts and limits

`benchmark-results/m14-feature-fixture-summary.json` records counts established by deterministic CPU tests: partial3/5 frame reuse produces3 hits/2 extra evaluations; a second5-frame low-VRAM prepass makes0 evaluations with identical float32 files; a second7-frame normal Hann stream makes0 foundation evaluations with identical batons; failed2-frame export resumes with1 hit/1 evaluation in a fresh session. Actual hard process exit75 releases ownership and removes the recognized private tensor stage on reopen. Cancellation and disk-full leave no partial sealed entries. These are **protocol/control fixture counts, not GPU FPS, visual-quality or real-model parity measurements**.

## Problems and root fixes

`.ralph/issues.json` records environment invocation030 (Pyright target interpreter), potential/confirmed integration031 (nonexistent cleanup API/default telemetry), typing032, fixture contracts033, heterogeneous kwargs typing034, confirmed integrity035 (post-lookup mutation) and fixture narrowing/style036, with pre-fix evidence/action/verification. Existing closure false-positive008 is now explicit `offset=offset` binding. Original broad static backlog005 remains open. Default inference behavior, backend selection, model precision and chronological RVM remain preserved.

## Operational scope

Use the recovery commands in `docs/recovery.md`, adding `--durable-foundations` to both initial run and resume. Production remains explicit-scheduling EXR/CUDA with local assets; no CPU/MPS inference fallback. Original proxy bits/dtype, native index, channel/grid and complete parent job provenance conservatively select typed IID entries. Source/hints/settings/assets/backend/implementation changes invalidate reuse. Actual loaded asset paths/context bind before any runner feature access. No RVM/baton/recurrent state is persisted.

Immutable features and completed EXR bundles share owned sealed-entry storage/LRU budget. Transient NPY/copy/metadata files, journals/reports, outputs, physical allocation and RAM/VRAM are outside that logical budget. Unknown files/symlinks prevent deletion. Features are finite exact normalized float32, including signed/large latent values; loading forbids pickle and returns independent tensors after closing maps. Inputs/dependencies/cache must stay stable; checks are not globally atomic filesystem snapshots. GPU producer batch shape can change on misses, so real numerical roundoff/parity/quality/throughput are **NOT VERIFIED**. Windows directory flush/power-loss and target-platform behavior remain **NOT VERIFIED**.

## Remaining work and next action

The full brief remains **in progress**. Next: inspect RVM/Stage1/Hann carried state and implement a versioned per-shot checkpoint contract with safe serialization, complete compatibility/invalidation and CPU fault/restore tests before exposing recurrent/range resume. Preserve chronological RVM and require real-model numerical acceptance. Then integrate recoverable queue/cache into the desktop worker.

Other open work includes richer benchmark/resource/timing/held-out evaluation, high-bit video/managed colour, desktop editor/queue/previews/brushes, packaging and worker/OFX integration. Actual CUDA hardware, local weights and permitted representative/held-out footage, Windows target validation, Resolve SDK and a real host are missing prerequisites. No real GPU performance/quality or Resolve acceptance is claimed; product_complete remains false in `.ralph/checkpoint.json` and `.ralph/plan.json`.
