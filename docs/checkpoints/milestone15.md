# Run checkpoint — milestone 15, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, confirmed by milestone14 source/checkpoint and fresh297-test baseline. Milestone14 archived at `checkpoints/milestone14.md` and `.ralph/checkpoints/milestone14.json`. No competing supervisor/second loop started. All work remains local/uncommitted; no push, publish, upload, model/source downloads, dependency installs or user `.venv` modifications.

## Completed cycles

1. Established actual baseline (297 tests,134 files, full gate pass). Inspected RVM state, exported Stage1 recurrence configuration and Hann carried hints/accumulators. Added strict immutable typed/versioned exact-call RVM identity: parent, proxy dtype/shape/bits, incoming state hash, native chunk bounds and shot start. Shot start requires None; continuation requires a state hash. Three identity tests; full300 tests/no-new-finding gate passed. `benchmark-results/m15-cycle1-*`.
2. Added paired RVM float16 features/exact floating outgoing state in an atomically sealed safetensors bundle. Loader hashes the immutable bytes it deserializes, checks names/shapes/dtypes/finite values and owns independent tensors. Extracted existing loaded-model binding into a shared base used by both durable adapters. Actual float32/bfloat16/float16 state bits survive reopen; corrupt/incomplete state and zero budgets reject safely. Fixed a stale import after classification. Full304 tests/no-new-finding gate passed; `m15-cycle2-*`.
3. Bound RVM adapter before shared-session runner, carried counters through diagnostics and integrated the existing chronological prepass. Whole chunks hit or invoke the same complete model call; missing chunks consume verified restored state, cuts reset it. Both memory modes consume the reconstructed float16 prepass. Interrupted6-frame fixture restores2 frames and evaluates remaining4; missing-middle fixture evaluates1 and reuses5 with identical files. Fixed new import ordering after classification. Full307 tests/no-new-finding gate passed; `m15-cycle3-*`.
4. Added working `--durable-rvm` (default off, independently combinable with IID caching), policy identity and reports. Actual shared-session/native EXR export fixture cancels after sealing1 RVM frame, resumes into a preserved fresh destination with1 restored/1 computed frame and a fresh session. Fixed duplicate test import after classification. Full308 tests/no-new-finding gate passed; `m15-cycle4-*`.
5. Reproduced null-provenance loader crash before remediation; now validates the complete canonical producer/provenance contract and recomputes invalid entries. Added wrong metadata, post-lookup mutation, small/zero budgets, disk-full/cancel cleanup and actual child exit76/state restore. Tiny randomly initialized **actual RVM code** on CPU produces identical float16 prepass files for full replay and missing-middle recomputation from restored state. Final full verification passes.

## Final actual verification

- **314 tests PASS**, no skips; `.ralph/runs/manual-check/check-0.txt`.
- All4 verifier commands return0: tests, compilation, full code-health gate, whitespace; `.ralph/runs/manual-check/verification.json`.
- **140 Python files /0 syntax errors**; selected runtime imports, declared requirements/VCS and installed dependency consistency PASS.
- Ruff **520 findings**, Pyright **821 errors /17 warnings** remain. **0 new diagnostic fingerprints** against unchanged reviewed baseline. Repository is **not static-clean**. No ignore directives, checker weakening or baseline refresh.
- New checkpoint modules and identity/adapter/prepass/runner tests: focused Ruff/Pyright **0 findings**, `benchmark-results/m15-focused-*`.
- CLI help including `--durable-rvm` passes with `python3 -S`, no site packages/model import; `benchmark-results/m15-cli-help-stdlib.txt`.
- Same isolated Python3.12.15/NumPy1.26.4/OpenCV4.11/Torch2.9.1/torchvision0.24.1, Ruff0.17.1/Pyright1.1.414. No installs; temporary environment may need recreation. Pylance editor NOT VERIFIED; JS/TS/native diagnostics not applicable to current owned source scope.
- Fresh baseline archived `benchmark-results/m15-start-baseline/checks/`; final command `m15-final-verification.txt` and per-cycle logs retained.

## Actual measurements and problems

`benchmark-results/m15-recurrent-fixture-summary.json` records **CPU fixture work counts**, not a throughput benchmark. Deterministic6-frame prepass resumes2 cached/4 computed frames; missing middle recomputes1/reuses5; cancelled2-frame export resumes1/1 in a fresh session. Native RVM code with tiny random weights (D16,1 encoder layer, seed1525) replays6 frames with0 model evaluations and recomputes one missing frame with identical float16 files. Actual hard exit76 restores the completed state/features bundle. No pretrained weights, GPU FPS, visual-quality or Resolve acceptance was measured.

`.ralph/issues.json` records resolved HEALTH-TYPE-037 (obsolete Path import after context extraction),038 (new import order),039 (duplicate helper import) and HEALTH-RVM-040 (null/invalid provenance crash; canonical exact producer contract now required), each with pre-fix evidence/root/action/checks. Existing broad static backlog005 remains open.

## Operational scope and limits

See `docs/recovery.md` for commands and `docs/recurrent-checkpoints.md` for inspected state inventory and version1 identity/payload contract. Production remains local CUDA/EXR with explicit scheduling; normal commands/defaults retain their behavior. No CPU/MPS model fallback or precision downgrade. Model paths/dependency identity bind before adapter access; source/hints/corrections/settings/assets/implementation/backend/policy and exact input/state bits participate conservatively.

RVM entries seal existing normalized float16 features and exact float32/bfloat16/float16 producer state **together**. Chunk call grouping and shot resets are unchanged. Recomputed state must match a suffix's incoming digest before that suffix can hit. Corrupt recognized entries recompute; unknown files stop deletion. SQLite ownership and strict staging/budgets are shared with IID/output bundles. Sealed logical bytes are bounded; CPU/GPU copies, serialization/read buffers, temporaries, journals/reports/render copies and physical allocation are outside that limit.

Every selected frame still decodes chronologically and reconstructs the full RVM memmap. This is RVM **prepass** recovery: Hann carried hints/live blending/pending Stage2 frames, sparse fallback state and writer output restart. It is not direct range seek, full mid-job continuation, reverse scrubbing or an OFX random-access solution. Stage1's exported recurrent sites remain disabled by its existing loader. Sources/dependencies/cache must remain stable; fingerprints are not a global atomic snapshot.

Pretrained CUDA numerical parity/repeatability/tolerances across restart, transfer/memory/performance effects, permitted reference visual-quality review, Windows directory flush/power-loss and Resolve SDK/host validation remain **NOT VERIFIED**. Keep explicit opt-in. No real-time/professional-quality/macOS acceleration claim.

## Next action and remaining brief

Implement a consistent **Hann window checkpoint** binding next-window cursor/schedule, carried hints, live accumulators/weights, source-frame dependencies and finalized baton ranges. Verify interruption/reopen equality plus sparse/correction/reset invalidation with CPU fixtures before exposing range/output resume; real-model acceptance remains external. Then integrate recoverable queue/cache into desktop worker/editor.

Broader evaluation/resource/alternating benchmark/stress tooling, high-bit video/managed colour, desktop previews/brushes/queue, packaging and OFX/worker integration remain open. Actual CUDA hardware, model weights, permitted representative/held-out footage, Windows target validation and Resolve SDK/host remain missing prerequisites. The full product goal is **in progress**; `.ralph/plan.json` and `.ralph/checkpoint.json` keep product_complete false.
