# Run checkpoint — milestone 24, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, verified against milestone23 source/checkpoint and a fresh451-test/206-file baseline. Read the active objective, AGENTS, engineering contracts, development brief and ledgers. Milestone23 is archived in `checkpoints/milestone23.md` and `.ralph/checkpoints/milestone23.json`. The full goal remains active and incomplete; no competing supervisor was launched.

## Completed cycles

1. Reproduced HEALTH-LIFETIME-135 with real mouse input: parent close accepted/hid a dirty draft after the editor worker stopped. Guard parent close before **any** owner shutdown; preserve the workspace/draft and require explicit Commit/Clear. Accepted editor close disables new strokes/commands, including after late events.453 tests pass,0 new findings.
2. Added core stopped-queue sealing: exclusive ownership, bounded queue-ID-bound marker, complete fsynced bytes published without replacing an existing file, writer archive check after lock acquisition and readonly inspection. Database/old request statuses remain.457 tests pass,0 new findings.
3. Added Archive stopped history, readonly controls and truthful current-owner stopped/error labels. A live owner refuses archival without being stopped by that action.458 tests pass,0 new findings.
4. Added Archive + fresh queue and Open last archive. Old paths/data stay in place; fresh activation uses existing clean processing barriers. A committed seal remains reported if fresh creation fails; no rollback or resubmission is invented.460 tests pass,0 new findings.
5. Added process-exit/publication-race/recovery-snapshot evidence, symmetric selected-asset/queue separation before owner changes and actual generated workflow. Final full gate passes464 tests; CPU correctness3 repetitions completes.

Contracts: [queue-archives.md](queue-archives.md), [correction-editor.md](correction-editor.md), [desktop.md](desktop.md). No inference/output/precision/approximation defaults changed. No dependency install, user `.venv` change, large download, upload or publication.

## Verification

- Fresh baseline: `benchmark-results/m24-start-verification.txt`, archived `m24-start-baseline/`.
- Final: `benchmark-results/m24-final-verification.txt`; `.ralph/runs/manual-check/verification.json` success true, all4 checks return0. **464 tests PASS/no skips;207 Python files,0 AST errors.** Imports, declared requirements/VCS identity, installed dependency consistency, compilation and whitespace pass.
- **Ruff520 findings; Pyright800 errors/17 warnings;0 new fingerprints** against the unchanged baseline. Repository is **not static-clean**. Existing compatible `/private/tmp/corridorkey-compatible-venv`, Python3.12.15/NumPy1.26.4, unchanged runtime/tool stack and optional Qt6.10.2. No checker weakening, baseline refresh or global suppression.
- Existing dataclass instance/class and heterogeneous-set serialization contracts inspected/covered. Dependency-free desktop help and queue sealing/inspection pass. Pylance editor diagnostics NOT VERIFIED.
- Actual `python -m corridorkey.benchmark --output benchmark-results/m24-cpu-benchmark.json`: complete/success,3 repetitions, median29.989369667018764s, min29.886918374919333s, max29.992047250038013s. These are correctness-suite elapsed times, **not inference FPS, GPU speedup or output-quality evidence**. No controlled baseline/candidate comparison claimed.

## Actual generated workflow

`benchmark-results/m24-generated-summary.json`, artifacts `benchmark-results/archive-m24-gwftrxcp`:

- Real Qt mouse stroke; dirty parent close refused, editor remained dirty/visible and its workers were not stopped. Separate stopped-editor probe changes accepted/hides to refused/visible (`m24-dirty-close-before.txt` versus `m24-dirty-close-after.txt`).
- Commit/save/render uses the actual editor service, immutable correction documents, worker and native EXR path with an **explicit constant-alpha CPU fixture**. It is not pretrained inference.
- Stop/await owner, archive old history and create/activate a fresh queue. Rotation submitted **zero work**; old database bytes unchanged. Archived settings loading also submitted nothing. A later **explicit** render used a fresh output destination and recorded one request in the fresh queue.
- Old native outputs and immutable correction snapshot hashes remained unchanged. Archived original request retained its actual result observation. Old PID/output semantics remain historical/unverified.
- Dirty editor and archived-history captures inspected locally; owned worker stopped cleanly. GPU FPS null; neural quality NOT VERIFIED.

## Issues and limits

HEALTH-LIFETIME-135 reproduced/classified confirmed before remediation and resolved. HEALTH-ARCHIVE-136 known-owner status presentation and137 symmetric asset/queue separation were classified before fixes and resolved. Existing static backlog and documented environment/false-positive issues remain.

**HEALTH-FORMS-126 native crash mechanism remains open.** Current full gate and3 CPU correctness runs pass, but do not establish its underlying cause or interactive target acceptance. Prior LLDB target launch was unavailable; retain existing faulthandler evidence and explicit native widget cleanup. No GC/tests disabled or skipped.

Sealed history is read-only through managed APIs, not OS-level immutable or an authenticated/portable backup. Existing unresolved rows remain unresolved; no process death/completion/current output validation is inferred. The source directory is not moved, preserving original managed batch references. Media/model/correction assets stay external and are not copied. Selected detail checksum validation remains authoritative for saved bytes.

Every writer checks the archive marker **after** exclusive ownership. Marker admission4096 bytes/depth8; invalid/symlink markers fail closed. An exclusive hard link publishes complete fsynced marker bytes without replacing even a concurrently created file. A process exit immediately after publication leaves readable sealed history and may leave a private stage alias. Unsupported hard-link/SQLite/filesystem behavior fails explicitly; Windows, network filesystems and power-loss durability remain NOT VERIFIED. File fsync is not proof of directory-entry survival after power loss.

Sealing/fresh creation is a two-step operation. An error can occur after publication; UI re-inspects/reports the actual committed seal rather than assuming rollback. Fresh I/O failure may leave a newly owned incomplete directory, with no automatic cleanup of unknown files. No automatic unseal/retry or media/output deletion. Current-format database bytes are preserved; legacy optional-column migration can still happen when opening a legacy writer before sealing.

4096 retained requests per active workspace and128 headers/page remain. Fresh workspaces reset admission, not total disk consumption. Physical budgets, retention deletion, portable backup/export/search remain incomplete. A stopped/failed editor cannot accept new commits; uncommitted pixels are retained in memory until explicitly cleared and are not claimed automatically saved. Accepted normal close now prevents new strokes during cleanup; committed recovery snapshots remain on disk.

## Next action and original remaining scope

Build synchronized source-aligned A/B comparison and usable range controls using actual completed exports/source-selection offsets and shared decode/colour contracts. Continue total resource/disk budgets, explicit retention deletion/portable backups, target packaging/asset compatibility/distribution and advanced corrections. Keep126 native GC/painting investigation open until target instrumentation/interaction proves acceptance.

The complete original brief still includes direct seek/Stage2/writer recovery, high-bit video/managed colour, controlled alternating/stress/held-out evaluation and Resolve SDK/adapter/host integration. Supported GPU/backend, real assets/permitted representative footage, Windows/packaged application and Resolve SDK/host remain missing or unverified prerequisites. Real GPU performance, visually assessed neural quality and Resolve integration are **NOT VERIFIED**. Independent work remains actionable; the goal is not complete or blocked.

Machine checkpoint: `.ralph/checkpoint.json`; task ledger: `.ralph/plan.json`; issue ledger: `.ralph/issues.json`.
