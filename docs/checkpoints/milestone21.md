# Run checkpoint — milestone 21, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, confirmed by milestone20 source/checkpoint and a fresh401-test/181-file baseline. Repository instructions, the active objective, task/issue ledgers and engineering contracts were read. Milestone20 is archived in `checkpoints/milestone20.md` and `.ralph/checkpoints/milestone20.json`. No second Ralph supervisor was launched. The full development brief remains incomplete.

## Completed work

1. Added an opt-in durable desktop queue journal with an independent exclusive SQLite owner lock, checksummed bounded request/event records, session metadata and explicit dispatch phases. Intent is committed before dispatch. Saved observations do not establish current process or output validity. Tests:404 passing.
2. Added read-only queue history and explicit activation for future work. Reopening history never submits jobs. Settings can be loaded for deliberate submission; old PIDs remain historical. Corrected ownership-switch and shutdown handling. Tests:408 passing.
3. Added explicit desktop batch recovery/cache controls using the existing worker/controller, immutable managed batch snapshots and actual report inspection. Source manifests and previous outputs are preserved. Tests:413 passing.
4. Hardened shared bounded JSON admission and batch schema validation; rejected duplicate keys, nonfinite values and boolean schema versions. Normalized snapshot defaults through the shared job codec so path aliases cannot change recovery identity. Tests:418 passing.
5. Exercised the generated native EXR/cache/recovery workflow and production missing-assets failure. Distinguished persistence failure before and after dispatch: a possibly executed request cannot be fabricated as rejected. Improved history/report presentation. Final tests:422 passing.

Issues HEALTH-QUEUE-099 through HEALTH-QUEUE-114 were classified before fixes and resolved with evidence in `.ralph/issues.json`. Root causes included SQL binding mismatch, insufficient type narrowing, incomplete SQLite cleanup, terminal-phase overwrite risk, bridge ownership races, inconsistent batch path normalization and incorrect post-send failure semantics.

## Verification

Final full gate: `benchmark-results/m21-final-verification.txt` and `.ralph/runs/manual-check/verification.json`.

- **422 tests PASS, no skips;191 Python files,0 AST errors.** Imports, declared requirements, VCS dependency identity, installed dependency consistency, compilation and whitespace checks passed in `/private/tmp/corridorkey-compatible-venv`.
- Ruff520 findings; Pyright800 errors and17 warnings. **Zero new fingerprints** against the unchanged static baseline. The repository is **not static-clean**. Focused queue/recovery modules/tests and touched bridge/window/report helper checks passed.
- Dependency-free desktop `--help` passed. No dependency installation or user `.venv` modification was required.
- Actual generated workflow: first batch `[render, verified_cache_restore]`; explicit resume `[verified_existing, verified_existing]`; after deliberate corruption `[verified_cache_restore, verified_existing]`. Recovery used a fresh destination and preserved the corrupt original and source manifest.
- Read-only reopening submitted **zero jobs**. A separate production worker durably reported missing configured assets and stopped cleanly. No processing fallback or asset download occurred.
- Artifact summary: `benchmark-results/m21-generated-summary.json`; images/reports under the artifact root recorded there. UI captures were inspected; a later read-only render refreshed column widths and report ordering.

The generated demo used an explicit test-only constant-alpha engine. Its timings are CPU fixture timings, not neural inference performance or output-quality evidence. Production GPU FPS and neural quality remain **NOT VERIFIED**.

## Contracts and limits

The journal admits4096 retained requests; history displays the latest128 headers. Requests/events and batch JSON are bounded to4MiB; JSON nesting to128. Report reads are bounded to32MiB and displayed text to2MiB. Progress persistence is throttled to0.25 seconds. These are admission/display bounds, not a global RAM or disk quota. Owner locking and process-crash behavior were tested locally; power-loss durability, Windows storage/containment and native OS dialogs remain unverified. No saved PID is used as proof of a live owner. Resume is explicit and actual output/cache validation stays in the existing controller.

## Remaining work and next action

Next implement common validated desktop job/settings forms and recoverable single-job construction, avoiding a requirement to hand-author batch JSON. Add deliberate local model/foundation asset selection and validation with explicit unsupported-backend status. Continue queue paging/archive and resource/disk feedback, synchronized range/A-B tools and native packaging. Preserve immutable data, explicit resume and no automatic retry.

Direct-seek/Stage2 writer recovery, high-bit-depth/colour acceptance and representative alternating-workload stress remain in scope. Supported GPU hardware/backend, all required model assets, permitted representative footage and Resolve SDK/host access remain missing or unvalidated prerequisites. Real GPU performance, visually assessed output quality, Windows behavior and Resolve integration are **NOT VERIFIED**. Independent implementation remains actionable; the full objective is active, not complete or blocked.

Machine-readable checkpoint: `.ralph/checkpoint.json`; persistent task ledger: `.ralph/plan.json`.
