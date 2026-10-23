# Run checkpoint — milestone 16, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, confirmed by milestone15 source/checkpoint and fresh314-test baseline. Milestone15 archived in `checkpoints/milestone15.md` and `.ralph/checkpoints/milestone15.json`. No competing supervisor or second loop started. All work remains local/uncommitted; no push/publish/upload, media/weight downloads, dependency installs or user `.venv` changes.

## Completed cycles

1. Established fresh full baseline (314 tests,140 files, code-health/import/dependency/compile/whitespace pass). Added strict typed Hann event identity: parent, full schedule, window cursor/bounds/segment, next boundary/carry, incoming replay state and complete exact Stage1 input precision/shape/bits. Two tests; full316 tests/no-new-finding gate passed. `benchmark-results/m16-cycle1-*`.
2. Added safe raw Stage1 baton/used auxiliary-alpha event adapter, exact float32/bfloat16/float16 preservation, canonical producer metadata, immutable verified-byte safetensors loading, shared bounded storage and full replay-state digest (carry/accumulators/weights/pending frames/previous segment). Invalid inventory/provenance, zero budgets and active outgoing recurrence reject safely. Fixed new loop diagnostic after classification. Full320 tests/no-new-finding gate passed; `m16-cycle2-*`.
3. Bound adapter before shared runner and replayed verified events through existing Hann carry/correction/blend/flush code. Both streaming and low-VRAM baton-file paths reconstruct state; counters increment only for actual Stage1 calls. Interrupted15-frame/three-window fixture resumes1 hit/2 calls and matches uninterrupted batons; repeat makes0 calls. Fixed optional dependency typing after classification. Full321 tests/no-new-finding gate passed; `m16-cycle3-*`.
4. Added working `--durable-hann` default-off policy/report wiring and fresh-session failed-export prefix reuse. Native EXR/runtime matrix covers full/offline/hold/causal/cuts/corrections/full-rate resets/neural fallback in both paths, replaying with0 Stage1 calls and identical outputs. Corrected substitute auxiliary /4 geometry to the actual /2 contract; repaired full-rate fixture sparse settings and module/diagnostic typing after classification. Full323 tests/no-new-finding gate passed; `m16-cycle4-*`.
5. Added corruption-middle recovery with verified suffix reuse, exact signed/large raw values, post-lookup mutation, capacity/disk-full/cancel cleanup, shared-session asset mismatch and actual hard child exit77/reopen. Added runtime rejection of active Stage1 recurrent sites before cache lookup. Final328 tests and full gate pass. Final helper import diagnostics were classified and corrected without suppressions.

## Final actual verification

- **328 tests PASS**, no skips; `.ralph/runs/manual-check/check-0.txt`.
- All4 verifier commands return0: tests, compilation, whole code-health gate and whitespace; `.ralph/runs/manual-check/verification.json`.
- **145 Python files /0 syntax errors**; selected runtime imports, declared requirements/VCS and installed dependency consistency PASS.
- Ruff **520 findings**, Pyright **803 errors /17 warnings** remain. **0 new diagnostic fingerprints** against unchanged reviewed baseline. Repository remains **not static-clean**. No global ignores, checker weakening or baseline refresh. Helper imports/narrowing improve dependency/type lifetimes and remove existing occurrences without behavior changes.
- New Hann identity/adapter/runtime test files and runner: focused Ruff/Pyright **0 findings**, `benchmark-results/m16-focused-*`.
- CLI help with all cache flags passes using `python3 -S`, without site packages/model import; `benchmark-results/m16-cli-help-stdlib.txt`.
- Same isolated Python3.12.15/NumPy1.26.4/OpenCV4.11/Torch2.9.1/torchvision0.24.1 and Ruff0.17.1/Pyright1.1.414. No installs; temporary environment may need recreation.
- Fresh runtime prerequisite check: CUDA **false**; all5 configured runtime/foundation/stat asset paths **absent**; FFmpeg **not on PATH**. No inference backend fallback. Pylance editor, pretrained GPU/quality, Windows power-loss and Resolve SDK/host **NOT VERIFIED**; JS/TS/native diagnostics not applicable to current owned source scope.
- Run-start baseline archived `benchmark-results/m16-start-baseline/checks/`; final command `m16-final-verification.txt`, per-cycle logs retained.

## Actual fixture work counts and problems

`benchmark-results/m16-hann-fixture-summary.json` records CPU deterministic work counts, **not a throughput benchmark**. A15-frame/8-frame-window/4-frame-stride stream has3 windows/24 frame instances. After sealing1 window, resume has1 hit/2 new calls (16 frame instances); full replay has0 calls. Corrupt middle recomputes1 call and matches all batons in both memory paths. A14-subcase matrix covers7 modes across2 paths and matches36 native EXR outputs per case with0 repeat Stage1 calls. A failed2-event export resumes1 hit/1 new event in a fresh session. Actual hard exit77 restores a sealed event/releases ownership. These use deterministic neural substitutes; pretrained GPU parity, FPS and professional/visual quality are not measured.

`.ralph/issues.json` records resolved HEALTH-TYPE-041 (new dictionary iteration),042 (optional helper dependency lifetimes), HEALTH-HANN-043 (substitute auxiliary geometry), HEALTH-TEST-044 (full-rate fixture used rejected sparse cuts), HEALTH-TYPE-045 (dynamic module/list and optional diagnostics narrowing). Each has pre-fix evidence/root/action/checks. Existing broad static backlog005 remains open.

## Operational scope and limits

See `docs/recovery.md` for commands and `docs/hann-recovery.md` for the identity/payload/replay contract. `--durable-hann` is explicit/default-off and independently combines with IID/RVM flags. Production remains CUDA/EXR with explicit scheduling and local assets. Shared-session model/dependency binding happens before adapter use; active Stage1 recurrent sites are rejected. Existing exported runtime already disables them.

Raw baton and used auxiliary alpha preserve exact producer precision in one sealed entry; unused auxiliary foreground/bottleneck are excluded. The same source/conditioning/carry/correction/interpolation/blending/finalization code runs after verified replay. Every prior event is replayed or recomputed, reconstructing carried hints, sums/weights, pending source frames and finalized baton ranges. Entries bind actual incoming state and complete model input values, so a changed prefix/schedule/correction/cut cannot authorize stale suffix reuse. Payload/provenance/shape/dtype/finite checks use verified immutable bytes; no pickle/mapped alias or partial writer completion is trusted.

This is **chronological prefix event reconstruction**, not a direct arbitrary-window snapshot seek. Source decode, exact input hashing, foundation work unless independently cached, and Hann replay arithmetic still run. Stage2/propagation/writers export again into fresh recovery destinations. Partial outputs remain preserved and incomplete. Features/RVM/Hann/output entries share logical sealed-byte/LRU storage; transient hash/read/serialization/copy buffers, temporary files, journals/reports/render copies, physical allocation and total RAM/VRAM remain outside that cap. No throughput or process-wide resource improvement follows from call counts alone.

Sources/dependencies/cache must remain stable; fingerprints are not globally atomic filesystem snapshots. Pretrained CUDA repeatability/parity/tolerances, transfer/memory/performance effects, held-out/reference visual quality, Windows directory flush/power-loss and Resolve SDK/host validation remain NOT VERIFIED. No real-time/professional-quality/macOS GPU claim.

## Next action and full remaining scope

Integrate the shared recoverable engine into an **isolated persistent worker** with a versioned request/progress/result/error protocol, bounded render queue, cancellation and supervised process crash/restart. Provide a working CLI client before desktop UI; reuse session/model/cache contracts and explicit unsupported-backend behavior. Then build desktop previews/editor/queue.

Direct range seeking and Stage2/writer checkpoints remain open alongside broader evaluation/resource/alternating stress benchmarks, high-bit video/managed colour, desktop brushes/previews/settings/cache/resources, packaging and OFX/worker/host integration. Actual supported CUDA hardware, model assets, permitted representative/held-out footage, target Windows validation and Resolve SDK/host are still required. Full product goal remains **in progress**; `.ralph/plan.json` and `.ralph/checkpoint.json` retain product_complete false.
