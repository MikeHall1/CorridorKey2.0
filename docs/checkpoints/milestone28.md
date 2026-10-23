# Checkpoint — completed-cache desktop workflow, milestone28

Full product scope remains **in progress**. Five cycles is now a checkpoint interval; continue automatically in the same loop. Previous m27 is preserved in `checkpoints/milestone27.md` and `.ralph/checkpoints/milestone27.json`. Existing uncommitted work and the user environment were preserved.

User-visible result: select one highlighted completed queue render → **Prepare OFX cache…** → choose an existing folder → prepare/cancel/copy the real cache path. Actual completed geometry/FPS/source range and draft/test-only labels are shown. Unsent form edits cannot change the selected record. Invalid replacements disable old preparation. Hash/decode/seal I/O runs on an owned background thread; close waits for actual terminal delivery. Exact foreground3/alpha1 channels reject silent conversions; committed cache publication remains truthful on late cancellation/stage cleanup errors. No inference defaults/cache format changed.

The native ABI test now enumerates the actual module in an owned child which exits before explicit file cleanup. Actual Windows execution remains unverified. The repository runner no longer enforces an obsolete five-cycle stop or repeats full suites every pass; STOP, explicit invocation budgets and per-pass execution timeout remain. No competing supervisor was launched.

## Evidence and review

- `benchmark-results/m28-verification-summary.json` records commands, durations, environment/config/file hashes and distinct **self-review**, not independent review. Reviewed batch SHA256: `1b350e21820c21555d8a952ad3eec4300f4de9f1fd665a4445d5bdc7401a439e`.
- Final complete gate: `benchmark-results/m28-final-verification.txt`, archived `m28-final/`: **504 tests PASS/no skips,40.386s;226 Python files, zero AST errors**. Compile/import/requirements/VCS/dependency/whitespace checks pass. Ruff520/Pyright696 errors17 warnings retain **zero new fingerprints**; repository is not static-clean. Runner focused check9 tests/0.299s; workflow focused24 tests/9.193s. Existing runner diagnostics remain, rather than being misreported as a clean focused pass.
- Earlier m28 full gates489/492/497 were run under the previous verification policy. Final504 covers changed source; unchanged historical/prerequisite evidence is reused. No repeated CPU benchmark was run under the revised policy; m27 correctness timings remain historical and are not inference FPS.
- `m28-generated-summary.json`: actual Qt/worker/native EXR/source range[1,3) → prepared cache → compiled native reader; expected pixels(.25,.25,.25,.5), one render request, original exports/queued settings unchanged, cancelled second preparation preserves first, close waits, worker cleanup error null. Explicit CPU neural substitutes; screenshot inspected. GPU FPS null, quality/Resolve NOT VERIFIED.
- HEALTH159–168 fixes reviewed and locally verified; native Qt mechanism126 and broad static005 remain open. Source/content/time/colour binding remains an unmet requirement, not a claim covered by this batch. No installs, dependency/interpreter changes, downloads, uploads, publication or baseline/checker weakening.

## Acceptance and next action

`acceptance.md` is the requirement matrix with stable IDs, scenarios, thresholds, environments, evidence and gaps. Provisional engineering targets are declared before future measurements. It distinguishes core/desktop fixtures from actual GPU/quality/package/host acceptance, and tracks all20 performance and20 quality investigations.

Next: inspect the current native cache/source/time/colour interfaces and define a locally testable producer/host binding and invalidation contract (CK-OFX-02). Preserve explicit refusal of unsupported retimes/proxies/scales. Then continue independent alternating benchmarks, resource budgets/retention/portable backups, advanced corrections, high-bit/managed colour, direct-seek/Stage2/writer recovery and remaining packaging work.

Unchanged external prerequisites from m27: CUDA unavailable, five model/foundation paths absent, CMake/FFmpeg absent; Clang available. Actual Windows x64/MSVC/Resolve/Fusion, real model assets, permitted reference/held-out footage and clean-machine packaged app acceptance remain unverified. Do not repeatedly retry unchanged unavailable prerequisites.

No active managed subprocess remains at this checkpoint. Machine checkpoint `.ralph/checkpoint.json`; tasks `.ralph/plan.json`; issues `.ralph/issues.json`. Continue the same chat loop. Do not start another supervisor, reset the cycle count because the goal changed, or mark product complete.
