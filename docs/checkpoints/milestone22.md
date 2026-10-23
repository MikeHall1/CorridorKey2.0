# Run checkpoint — milestone 22, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, verified against milestone21 source/checkpoint and a fresh422-test/191-file baseline. Read AGENTS, the current objective, development brief, engineering contracts and task/issue ledgers. Milestone21 is archived in `checkpoints/milestone21.md` and `.ralph/checkpoints/milestone21.json`. The full objective remains active and incomplete; no competing supervisor was launched.

## Completed cycles

1. Common desktop engine/writer form with exact numeric text, complete shared-codec validation, advanced-field preservation and stale-edit rejection. Apply submits nothing.426 tests pass, no new static findings.
2. Current-workspace one-job recovery recipe preparation, off-thread filesystem work, unique future output, preserved source/hints/corrections and explicit Run/resume. Preparation submits nothing; UI I/O races are rejected.429 tests pass, no new findings.
3. Separate bundled implementation and local asset roots through RuntimePaths/session/worker/client, retaining runtime re-export compatibility. Recovery now honors worker default weights when a batch omits them; explicit batch weights retain precedence.433 tests pass; new test import diagnostics were classified/fixed and corrected code health passed.
4. Deliberate local asset root/runtime override selection, read-only metadata checks and isolated actual backend probe. Applying paths requires clean stopped worker ownership and never loads a model or submits work.438 tests pass, no new findings.
5. Close benchmark-cleanup activation gaps, admit exact saved recipe bytes before directory creation, remove optional-Torch coupling from metadata tests and prioritize unsupported backend status. Exercise actual generated workflow. Investigate native Qt full-suite crashes and explicitly destroy affected test-owned native widgets after owner shutdown. Final full gate passes440 tests, with native mechanism investigation still open.

Features/contracts: [desktop-settings.md](desktop-settings.md), [desktop.md](desktop.md), [durable-queue.md](durable-queue.md). Core inference defaults/approximations/precision remain unchanged. No new runtime/dev dependencies, user `.venv` changes, large downloads, uploads or publication.

## Verification evidence

- Fresh baseline: `benchmark-results/m22-start-verification.txt`, archived `benchmark-results/m22-start-baseline/`.
- Final: `benchmark-results/m22-final-verification.txt`, `.ralph/runs/manual-check/verification.json` success true; all4 checks return0.
- **440 tests PASS, no skips;201 Python files,0 AST errors.** Imports, declared requirements/VCS origin, installed dependency consistency, compile and whitespace pass in the existing `/private/tmp/corridorkey-compatible-venv` (Python3.12.15, NumPy1.26.4, unchanged upstream runtime stack and optional Qt6.10.2).
- **Ruff520 findings; Pyright800 errors/17 warnings;0 new fingerprints** against the unchanged static baseline. The repository is **not static-clean**. No suppression/checker weakening/baseline refresh. Pylance editor diagnostics NOT VERIFIED.
- `benchmark-results/m22-all-tests-lifetime-fixed.txt`: separate full440-test pass after lifetime corrections. `m22-qt-lifetime-fixed.txt`:43 desktop tests pass. Complete final gate also ran with Python faulthandler enabled.
- Dependency-free desktop help and local metadata inspection pass. Dataclass instance/class narrowing and heterogeneous-set serialization contracts were inspected; existing regression coverage remains passing.

## Actual generated workflow

`benchmark-results/m22-generated-summary.json` points to local artifacts under `benchmark-results/desktop-m22-xqq6g90t`.

- Common form preserved exact sequence FPS23.976023976023978 and deliberately selected float EXR/alpha output. Recipe preparation submitted0 jobs.
- Actual worker/session/native EXR/cache/controller, with an **explicit constant-alpha CPU test fixture**, produced `[render, verified_existing, verified_cache_restore]` across Run, Verify/resume and corruption recovery. Saved recipe bytes and the intentionally corrupt original were preserved; restored output used a fresh destination.
- Actual local metadata/backend inspection reported CUDA unavailable. Separate **production** worker selection/request failed with the CUDA requirement and explicit no CPU/MPS fallback, without deserializing dummy metadata fixtures as models. Selected owners stopped cleanly.
- UI captures were inspected. `assets-current.png` prioritizes actual unavailable backend above inventory; its read-only refresh submitted no jobs.

Fixture timings are not neural inference FPS, output-quality or benchmark-speedup evidence. GPU FPS is null; pretrained neural quality **NOT VERIFIED**. Nonempty metadata fixture files are explicitly not real model assets. No representative footage/weights were downloaded.

## Problems and remaining uncertainty

HEALTH-FORMS/ASSETS/ENV115–127 record classifications before remediation, evidence, root causes and checks. All except HEALTH-FORMS-126 are resolved. Existing static backlog and documented environment/false-positive issues remain.

**HEALTH-FORMS-126 remains open.** Two full runs and a focused fault audit reproduced SIGSEGV at varying native Qt sites; the second full trace shows garbage collection during correction painting. New-only test cleanup was insufficient. Affected existing/new desktop/canvas fixtures now explicitly defer native destruction on the GUI thread after owned work stops and assert the destroyed signal. A focused UI run and two subsequent full440-test runs pass. This corrects fixture lifetime and mitigates the reproduction; it does not prove the underlying Qt/native crash mechanism or interactive-platform safety. Keep all failed evidence in `m22-final-first-failure/`, `m22-final-second-failure/` and `m22-qt-fault-audit.txt`. No GC/tests were disabled or skipped.

The process inventory attempt (`pgrep`) was unavailable because sysmond/process listing is unsupported in this environment. Generated workflow owners and verification tool handles were observed terminal; a global orphan-process inventory is **NOT VERIFIED**. Do not infer process death from old PID files.

Local asset checks cover presence/type/nonempty size and backend availability only. Checkpoint tensor/format compatibility, complete directory content and real model readiness require actual loading. Asset selection is app-session scoped, not saved in projects; recipes save a runtime override, while foundation root remains worker scoped. Benchmarks use their own manifests. Download management, packaged distribution and global RAM/disk/VRAM limits remain incomplete.

## Next action and full remaining scope

Continue the native widget/painting/GC lifetime investigation with target-platform instrumentation and interactive acceptance. Preserve explicit cleanup, all tests and no implicit retry. Then continue queue paging/archive, total resource/disk feedback, synchronized range/A-B playback, broader controls and target packaging/dependency/asset distribution. Common forms, one-job recipes and local metadata/backend asset selection are now implemented; deeper compatibility/download UX remains.

The complete brief still includes direct seek/Stage2/writer recovery, high-bit video/managed colour, advanced corrections, alternating/stress/held-out evaluation and Resolve SDK/adapter/host integration. Supported GPU/backend, real model assets, permitted representative footage, Windows/packaged application and Resolve SDK/host remain missing or unverified acceptance prerequisites. GPU performance, visually assessed neural quality, native target behavior and Resolve integration are **NOT VERIFIED**. Independent work remains actionable; the goal is not complete or blocked.

Machine checkpoint: `.ralph/checkpoint.json`; task ledger: `.ralph/plan.json`; issue ledger: `.ralph/issues.json`.
