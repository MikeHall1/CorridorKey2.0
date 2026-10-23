# Run checkpoint — milestone 9, 2026-10-22

Five implementation/verification cycles completed; stop at the requested boundary. No second supervisor or detached runner was started. Milestone 8 is archived in `checkpoints/milestone8.md` and `.ralph/checkpoints/milestone8.json`. All local implementation remains uncommitted; no push, publication, deployment, media upload or model/footage download occurred.

## Completed cycles

1. Established actual baseline: original 164 tests pass, syntax100/0, actual Ruff524 findings and compatible Pyright830 errors/17 warnings. Provisioned isolated Python3.12.15 with unchanged requirements (NumPy1.26.4), upstream Torch2.9.1/torchvision0.24.1 and pinned Ruff0.17.1/Pyright1.1.414. All declared runtime imports and dependency consistency checks passed. Existing user `.venv` was preserved; full environment freeze is `docs/environments/cpu-macos-py312.txt`.
2. Audited serializer callers and fixed the actual asdict type warning by narrowing dataclass instances. Typed finite JSON recursion preserves dict/list/tuple/set/frozenset/Path, deterministic natural/canonical set ordering, explicit unsupported class/object rejection and atomic failure preservation. Key collisions/nonfinite inputs are rejected. Eight focused regression tests pass; full suite172/172 after this change. No production caller supplying a dataclass class was found; it is not represented as a confirmed production-instance defect.
3. Added `scripts/code_health.py`, pinned dev requirements and reviewed `.ralph/static-baseline.json`. All findings remain in reports; no automatic baseline refresh, global ignores or diagnostic weakening. The gate detects new message/rule/path/severity fingerprints including multiplicities, handles moved lines, reports missing checkers NOT VERIFIED, checks syntax/imports/requirements/VCS/installed dependencies and is invoked by Ralph verification. Four gate regressions pass. Known closure warnings were investigated and documented as false positives without modifying working inference behavior.
4. Implemented normalized alpha MAE/SAD, raw SAD, explicit boundary and forward-gradient errors with masks/populations, finite/range validation and unavailable metrics for missing references/empty populations. Analytical tests validate exact kernels/counts and spatial-only boundary behavior. Interim full suite181/181; gate no new findings.
5. Added local `.npy` candidate/baseline comparison CLI with identical reference/masks, content hashes, shape/dtype, safe non-pickle single-array reading, changed-input detection, atomic JSON and input overwrite protection. Eight metric/CLI tests pass. Documented limitations, updated persistent ledgers and ran integrated verification.

## Final verification actually run

- **184/184 tests PASS**, no skips, compatible environment; `.ralph/runs/manual-check/check-0.txt`.
- Compilation and whitespace PASS; `.ralph/runs/manual-check/verification.json`.
- Full repository AST **105 files / 0 errors**, selected runtime imports PASS, declared requirements/VCS identity/installed dependency consistency PASS.
- Ruff **523 findings**; Pyright **826 errors / 17 warnings**. **No new findings** against the reviewed compatible pre-change baseline. Full repository is **not static-clean**. Raw reports and comparison: `.ralph/runs/manual-check/code-health/`.
- Focused diagnostics/evaluation/code-health modules and their three new test files: Ruff and Pyright **zero findings**, actually executed with pinned versions.
- Pylance editor diagnostics **NOT VERIFIED**; CLI Pyright verified. JS/TS/native checks currently not applicable to owned source.
- Actual generated two-pixel CLI fixture: candidate MAE0.25, baselineMAE1, delta−0.75; `benchmark-results/alpha-m9-generated-comparison.json`. This is an arithmetic fixture, not a real-footage quality measurement. No GPU/model throughput or visual improvement was measured.

## Problems and remaining work

Resolved HEALTH-TYPE-001, HEALTH-SERIAL-002, HEALTH-TOOLS-003, HEALTH-ENV-004 and new metric-test typing issue HEALTH-EVAL-006, with root causes/checks in `.ralph/issues.json`. HEALTH-STATIC-005 remains open for individual triage; F821/B023 closure lifetime findings have separate documented false-positive reasoning (007/008) and remain visible. No blanket lint cleanup/refactor was made.

The task ledger `.ralph/plan.json` now marks code-health baseline and serializer tasks done. **Evaluation remains in progress**: current CLI handles bounded in-memory normalized arrays only. Next implement bounded streaming image-sequence/inference-export evaluation with source/reference hashes, authorship/permitted-use metadata and baseline/candidate integration. Then add valid-foreground colour/composite and flow/occlusion-aware temporal metrics. Missing references must remain unavailable, and equal-population/units rules must remain explicit.

Higher-bit video/managed source colour, durable recovery, desktop/corrections editor/queue, packaging and official SDK integration remain incomplete. This Mac has no validated neural inference backend; CUDA hardware, model/foundation weights, permitted reference/held-out footage, relevant training data and Resolve SDK/Windows/NVIDIA host validation are still missing. CPU tests and static checks do not complete those acceptance criteria. The complete product goal is not achieved.

Resume by reading the objective, AGENTS, current ledgers and `docs/code-health.md`, and establish a fresh actual baseline before the next bounded run. The compatible environment is temporary and may need recreation. Continue the evaluation task, while retaining the five-cycle limit.
