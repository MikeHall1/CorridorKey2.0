# Run checkpoint — milestone 23, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, verified against milestone22 source/checkpoint and a fresh440-test/201-file baseline. Read the active objective, AGENTS, engineering contracts, development brief and ledgers. Milestone22 is archived in `checkpoints/milestone22.md` and `.ralph/checkpoints/milestone22.json`. The full objective remains active and incomplete; no competing Ralph supervisor was started.

## Completed cycles

1. Investigated native lifetime and fixed a confirmed production resource leak: replacing a stopped correction editor left all old native dialogs/canvases under the main window. Actual retained-child counts changed from2/3/4/5 to1/1/1/1. Retire stopped editors via GUI-thread deferred destruction; preserve the dirty-draft guard.441 tests pass, no new static findings.
2. Added bounded read-only queue history paging by SQLite admission sequence. At most129 headers read for a128-row page; no timestamp tie or concurrent newer insertion causes forward-traversal duplicates/skips.443 tests pass, no new findings.
3. Added Newer/Older/Refresh UI navigation and oldest-record inspection without activation/submission. Failed reads preserve current navigation state.444 tests pass, no new findings.
4. Added explicit worker resource snapshots to existing status: process lifetime peak RSS and only already-initialized current-device CUDA allocator counters. No Torch import or CUDA initialization. Partial imports/probe errors remain unavailable. Added containing-filesystem space sampling without tree scan/creation.449 tests pass, no new findings.
5. Added working Resources window, human summaries/exact bounded details, explicit worker requests, off-thread filesystem I/O and close barriers. Actual generated history/resource workflow and final full gate pass451 tests; CPU correctness benchmark completes3 repetitions.

Contracts: [resource-snapshots.md](resource-snapshots.md), [durable-queue.md](durable-queue.md), [correction-editor.md](correction-editor.md). No processing/output/precision/approximation defaults changed. No dependency install, user `.venv` change, large asset download, upload or publication.

## Verification

- Fresh baseline: `benchmark-results/m23-start-verification.txt` and archived `m23-start-baseline/`.
- Final full gate: `benchmark-results/m23-final-verification.txt`; `.ralph/runs/manual-check/verification.json` success true, all4 checks return0. **451 tests PASS/no skips;206 Python files,0 AST errors.** Imports, requirements/VCS identity, installed dependency consistency, compilation and whitespace pass.
- **Ruff520 findings; Pyright800 errors/17 warnings;0 new fingerprints** against the unchanged static baseline. The repository is **not static-clean**. Compatible environment remains `/private/tmp/corridorkey-compatible-venv`, Python3.12.15/NumPy1.26.4, unchanged upstream runtime tools and optional Qt6.10.2. No checker weakening, baseline refresh or global suppressions.
- Existing dataclass instance/class and heterogeneous-set contracts were inspected and remain covered by passing tests. Dependency-free desktop help passes. Pylance editor diagnostics NOT VERIFIED.
- Actual editor probe: `m23-editor-lifetime-before.txt` versus `m23-editor-lifetime-after.txt`. Additional lifetime, paging, partial/native GPU probe failures and asynchronous disk/close assertions remain in the complete suite.
- Actual `python -m corridorkey.benchmark --output benchmark-results/m23-cpu-benchmark.json`: complete/success,3 repetitions, median29.60450670798309s, min29.36759145895485s, max29.649469750002027s. This is CPU correctness-suite elapsed time (scope grew); **not inference FPS, speedup or GPU/quality evidence**. No controlled baseline/candidate comparison claimed.

## Actual generated desktop observations

`benchmark-results/m23-generated-summary.json`, artifact root `benchmark-results/desktop-m23-5jz3gx47`:

- Saved270 request observations traversed in pages128/128/14; oldest saved0 inspected. No journal activation, database remained byte-identical and render jobs submitted0.
- Actual production child status reported no active/pending job, actual PID/generation/time and peak RSS23625728 bytes. CUDA allocator fields null/NOT VERIFIED; no context initialized. This is a process lifetime peak, not current/total RAM.
- Actual containing-filesystem space reported at the nearest existing ancestor; requested output path remained absent. No job fit, physical cache total, quota or reserved space claim.
- Owned worker stopped cleanly. History and worker resource captures were inspected locally. GPU FPS null and neural quality NOT VERIFIED; no model rendering was performed in this demo.

## Issues and exact limits

Classified before remediation: HEALTH-LIFETIME/PAGING/RESOURCES128–135.128/129/131–134 resolved;130 documents a Ruff SQLite Row false positive and safe explicit dict conversion, without suppression.135 remains open. Existing static backlog and documented environment/false-positive issues remain.

**HEALTH-FORMS-126 native crash mechanism remains open.** Fresh/final gates and3 CPU correctness repetitions pass. Production retained-editor leakage is fixed separately; it does not prove the old SIGSEGV cause. LLDB2103 is installed, but its target launch exited-1/no such process; no native backtrace obtained (`m23-lldb-audit.txt`). Keep prior failed faulthandler evidence and explicit test-owned native destruction. Interactive target-platform/native Qt acceptance is NOT VERIFIED; do not disable GC or tests.

**HEALTH-LIFETIME-135 is next.** Source inspection shows parent close checks editor worker completion after calling child.close, without independently honoring the canvas dirty guard. A stopped worker and dirty draft can coexist (covered by the replacement guard test). Reproduce parent close/asynchronous-close painting, then preserve dirty state and recovery snapshots independently of worker completion. This is a potential defect; parent data-loss behavior has not yet been reproduced.

History remains capped at4096 retained admissions,128 displayed headers/page; paging does not archive/delete records or impose a physical DB budget. Existing selected request/event/batch byte/checksum limits remain. Resource details64KiB and short failure labels4096 characters are display bounds, not global memory budgets. Worker peak RSS only supports explicit macOS/Linux units; Windows remains unverified. GPU counters exclude other processes/driver allocations and are unsynchronized. Filesystem free space is a momentary volume observation; native I/O can block and parent shutdown waits for it. No automatic resource polling, memory enforcement or implicit backend selected.

## Next action and full remaining scope

First reproduce/fix HEALTH-LIFETIME-135 and regress parent/editor dirty-state shutdown, including painting during asynchronous close. Continue native GC/painting investigation126 when target instrumentation works. Then archive/retention and total resource/disk budgets, synchronized range/A-B playback and target packaging/asset compatibility/distribution. Paging and explicit snapshots are implemented; archive/global budgets are not.

The original full brief still requires direct seek/Stage2/writer recovery, high-bit video/managed colour, advanced corrections, controlled alternating/stress/held-out evaluation and Resolve SDK/adapter/host integration. Supported GPU/backend, real model assets, permitted representative footage, Windows/packaged application and Resolve SDK/host acceptance remain missing or unverified. Real GPU performance, visually assessed neural output quality and Resolve integration are **NOT VERIFIED**. Independent work remains actionable; the goal is not complete or blocked.

Machine checkpoint: `.ralph/checkpoint.json`; task ledger: `.ralph/plan.json`; issue ledger: `.ralph/issues.json`.
