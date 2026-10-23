# Run checkpoint — milestone 17, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, confirmed by milestone16 source/checkpoint and fresh328-test baseline. Milestone16 archived in `checkpoints/milestone16.md` and `.ralph/checkpoints/milestone16.json`. No competing Ralph supervisor or second autonomous loop started. Owned test/CLI worker processes were supervised and stopped. Work stays local/uncommitted; no push/publish/upload, user footage/weight downloads, dependency installs or user `.venv` changes.

## Completed cycles

1. Established actual full baseline (328 tests,145 files, full gate pass). Added bounded strict version1 local JSON framing, complete shared-job snapshots, absolute/canonical path handling, recovery/control/event schemas, duplicate/nonfinite/version rejection and generation/sequence validation. Fixed test symlink spelling assumptions and serializer/type contracts after classification. Full332 tests/no-new-finding gate passed; `benchmark-results/m17-cycle1-*`.
2. Added one session-owner thread, bounded waiting deque/history, concurrent status/cancel/shutdown, pending cancellation without execution, active cooperative cancellation, sequential failure continuation and draining. Default engine uses the existing session and recovery controller; normal jobs retain a model and recovery requests preserve fresh batch context. Added boundary traceback evidence and explicit callback binding after classification. Full336 tests/no-new-finding gate passed; `m17-cycle2-*`.
3. Added stdio child transport, private protocol descriptor with Python/native fd1 log redirection, bounded/coalesced events, partial-write handling and supervised client. Actual child/native EXR fixture verifies session reuse, queue/control cancellation, actual hard exit78/new-generation restart and no restart of a known live child. Corrected Popen keyword typing after classification. Full340 tests/no-new-finding gate passed; `m17-cycle3-*`.
4. Added working `python -m corridorkey.worker run/serve` and batch/recovery CLI client, atomic reports, bounded log tail, preserved recovery report references and actual unsupported-CUDA production failure (no fixture fallback). Child recovery fixture renders/restores then verified-existing resume. Callable client-factory contract fixed after classification. Full343 tests/no-new-finding gate passed; `m17-cycle4-*`.
5. Fault injection reproduced nested settings ambiguity/KeyError, timeout cancellation of a live worker, blocking caller writes/broken-pipe cleanup and late progress. Fixed strict nested parsing, same-child polling, bounded dedicated outbound writer, terminal progress lifetime, mode preflight and timeout/handshake bounds. Actual ignoring-descendant force-close, fragmented Unicode/partial writes, coalescing/critical backpressure and SIGINT tests added. Sandbox post-kill inspection denial is retained as unclean/uncertain rather than raising over original work; force cleanup targets the whole live group atomically after a cooperative deadline. Fixed descendant readiness fixture and clean-stopped EOF handling after actual SIGINT evidence. Generated concrete CLI/controller reports and completed final verification.

## Final actual verification

- **355 tests PASS**, no skips; `.ralph/runs/manual-check/check-0.txt`.
- All4 verifier commands return0: tests, compilation, full code-health gate and whitespace; `.ralph/runs/manual-check/verification.json`.
- **156 Python files /0 syntax errors**; selected runtime imports, declared requirements/VCS and installed dependency consistency PASS.
- Ruff **520 findings**, Pyright **803 errors /17 warnings** remain. **0 new diagnostic fingerprints** against unchanged reviewed baseline. Repository is **not static-clean**. No ignore directives, checker weakening or baseline refresh.
- All5 new worker modules and6 fixture/test files: focused Ruff/Pyright **0 findings** (`benchmark-results/m17-focused-*`). Pylance editor NOT VERIFIED. JS/TS/native diagnostics not applicable to current owned source scope.
- Actual `python3 -S -m corridorkey.worker run --help` passes without site/model packages; `benchmark-results/m17-cli-help-stdlib.txt`.
- Same isolated Python3.12.15/NumPy1.26.4/OpenCV4.11/Torch2.9.1/torchvision0.24.1 and Ruff0.17.1/Pyright1.1.414. No installs. Temporary environment may need recreation. CUDA/model assets/FFmpeg remain unavailable; production child actually reports CUDA requirement.
- Run-start evidence `benchmark-results/m17-start-baseline/checks/`; per-cycle health/tests/fault probes retained; final command `m17-final-verification.txt`.

## Concrete generated results

`benchmark-results/m17-worker-generated-summary.json` identifies the actual local artifact directory and reports:

- Two native EXR fixture jobs complete through a real child/shared session, fixture load counts1/1 and model_reused false/true.
- Recovery fixture modes `render`, `verified_cache_restore`, followed by2 `verified_existing` jobs on resume.
- All3 fixture children exit0 with verified clean stops.
- The real production CLI exits1, writes status failed with `CorridorKey v2 runtime requires a CUDA GPU.`, and closes its worker cleanly; no synthetic backend is selected.

These are **CPU process/controller/export fixture counts**, not pretrained weight-load timings, GPU FPS, numerical/visual parity or professional-quality measurements. Native EXR files are tiny generated fixtures kept local. Process tests also observe actual hard exit78, new-generation restart, SIGINT cancellation/exit130, nonreading peer responsiveness, one-PID observation timeout continuation, and a stopped heartbeat from a SIGTERM-ignoring descendant after force-close. The leader is reaped; complete postmortem process-tree inspection may be denied by the sandbox and is not claimed.

## Problems and root fixes

`.ralph/issues.json` records resolved046 normalization assumptions,047 serializer/error types,048 callback/fault diagnostics,049 subprocess keyword typing,050 callable factory,051 strict nested JSON,052 timeout observation,053 blocking input/pipe cleanup,054 live owned-group force-stop,055 mode preflight order,056 finite bounds/handshake,057 progress lifetime,058 exact readiness fixture and060 manual SIGINT/EOF lifecycle. Each has pre-fix evidence/root/action/checks. ENV059 remains **documented**: post-kill group inspection may be denied, code reports uncertainty/unclean shutdown and never turns that into a clean success. Existing broad static backlog005 remains open.

## Operational contract and limits

See `docs/worker.md` for concrete commands, API/wire schema and exact bounds. Normal jobs use the existing lazy persistent session on one owner thread. Recovery requests close normal resident state and use the existing fresh batch-session identity policy. No alternate inference engine/backend, precision change or default inference option change. Existing CLI/interactive/batch workflows remain available.

Local pipes carry bounded4 MiB finite JSON frames; Python/native logs are isolated on stderr. Concurrent controls operate during model calls. Default4 waiting jobs (allowed1..64), one active; client outbound/server framing queues8 each, critical events128/32 MiB, coalesced progress, bounded256 KiB log tail and4096 admission IDs. These limits exclude engine/model/serialization buffers, source/media, RAM/VRAM, OS pipes and total process-tree resources. The CLI submits sequentially and stops on first failure; the library exposes the bounded queue for desktop use.

A ready event is protocol readiness, not validated CUDA/assets/encoder readiness. Accepted jobs produce terminal outcomes or explicit lost uncertainty. Cancellation is cooperative; native calls may require force-stop after the deadline. Observation timeout polls the same handle. Explicit restart requires a confirmed terminal leader/new generation and never resubmits requests. Incomplete outputs are not relabelled complete; recovery resume still verifies its journal/output inventory. Large result transport errors explicitly say outputs may already be complete. Late callbacks cannot publish progress after a terminal record.

Force shutdown uses the owned live process group atomically after cooperative timeout; inspection denial/failure or surviving transport threads yields unclean status. Windows hard-crash descendant containment/Job Objects and target behavior remain **NOT VERIFIED**; live-tree taskkill fallback is not full host acceptance. Sources/manifests/assets must remain stable. Recovery requests reference a manifest read in the child at execution, not a globally atomic filesystem snapshot.

## Next action and remaining full brief

The locally verified worker/queue foundation is complete; **desktop GUI is now the next priority** in `.ralph/plan.json`. Choose/justify a maintainable framework from actual environment/packaging constraints and build a usable shell around `WorkerClient`: source selection/preview, actual queue/progress/cancel/status and saved settings. Continue alpha/foreground/composite/range/A-B/corrections/cache/resources and developer benchmarking backed by the real runner. Keep unavailable CUDA/assets explicit; do not simulate successful model previews.

Recovery direct seek/Stage2/writer checkpoints, broader resource/alternating stress benchmarks/held-out evaluation, high-bit video/managed colour, remaining desktop/editor controls, packaging and native OFX/host work remain open. Actual CUDA hardware, local weights, permitted representative/held-out footage, Windows containment/packaging and Resolve SDK/host acceptance are required. Full goal remains **in progress**, with product_complete false in `.ralph/plan.json` and `.ralph/checkpoint.json`.
