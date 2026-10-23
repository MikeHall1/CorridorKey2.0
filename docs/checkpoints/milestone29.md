# Checkpoint — producer-bound caches and controlled benchmark pairs, milestone29

Full product acceptance remains **in progress**. The active objective is the then-active user objective (now superseded by [the review objective](../review-objective.md)). Five cycles is a checkpoint interval, not a stop. This coherent parallel work package follows the five m28 cycles (six total); no competing supervisor was launched. Previous checkpoints are archived under `checkpoints/` and `.ralph/checkpoints/`.

## User-visible implementation

- Default-off `--source_witness` / Common job settings records native scene-linear RGB pixels actually consumed by a new render. Strict3-channel frame-directory/EXR input/output, max16384 selected frames; video/RGBA/premul/implicit transfers refuse. Full/draft/low-VRAM/shot reads share one observer; inconsistent rereads fail. Allselected digests finalize after owned cleanup into a bounded verified sidecar. Existing defaults remain unchanged.
- Preparation from a witnessed completed render emits version2 native cache. The effect verifies every selected mapped opaque floatRGBA Source frame before Output lease/write. Past/current/future mismatches fail; exact bits, no hidden conversion. Per-instance serialization and RenderInstanceSafe permit nested upstream evaluation; hash staging is48KiB. Legacy version1 stays manual. See [OFX cache](ofx-cache.md).
- Headless `benchmark_pair` and the existing developer panel run declared manual baseline/candidate configurations in AB/BA order (≥3pairs), freeze requests/identities, check actual effective settings and repetition stability, retain raw quality and refuse partial/mismatched speedups. Warmup is excluded from inner measured timing and each warm sample owns a separate warmed session. Actual paired reports inspect/export through background ownership; malformed reports/unexpected export exits deliver terminal errors. See [paired measurements](benchmark-pairs.md).

## Verification and review

Final `benchmark-results/m29-final-after-review-verification.txt`, archived `m29-final/`: **526 tests PASS/no skips in42.649s;232 Python files,0 AST errors**. Compile/import/requirements/VCS/installed dependency/whitespace pass. Ruff520/Pyright696errors17warnings retain **0 new fingerprints**. Existing backlog remains; not static-clean.

`m29-verification-summary.json` contains exact commands, focused outcomes/durations, source/config/environment hashes, exclusions and reused evidence. Reviewed batch SHA256 `5f37821baf29711743f23ce8b7e47b4e6aefe9358fd69c6b564b7d5c3a541c26`. Initial combined524tests passed; its static gate exposed one new annotation finding, fixed without baseline/checker weakening. Initial failed evidence retained in `m29-before-review-fixes/`.

`m29-bound-summary.json`: actual shared runtime with explicit CPU neural substitutes → producer witness → v2 sealer → strict compiled official-suite host under ASAN/UBSAN. Source range[1,4), max absolute output error0, original source/exports unchanged,154 image leases; no sanitizer errors. Independent HDR/source/chunk-boundary/nested-instance tests pass. These are correctness results, not real-time/pretrained/Resolve evidence. No extra CPU benchmark repetitions were run.

Parallel owners: `/root/ofx_binding_design` native/test implementation; `/root/paired_benchmarks` paired runner/desktop/report integration; `/root/review_m28` independent stable-source reviews. Main integrated/reviewed and alone updated ledgers. Independent reviewer reproduced171 and173, then accepted targeted fixes.170 global-lock source-fetch deadlock was fixed with per-instance ownership;172 explicit dispatcher route and174 honest metadata annotation fixed.169–174 resolved with evidence; no unresolved introduced critical/high/medium finding. Existing native Qt126 and broad static005 remain open. Reviews/identities: `m29-review-summary.json`.

## Limits and next executable action

CK-OFX-02 remains incomplete: exact full-selected-source verification costs O(count×pixels) per render; bit-equivalent decoder differences may refuse; managed primaries, expected companion settings/project semantics and actual host invalidation of outputs served without Render remain unverified. No trusted notification-only epoch is introduced. CK-BEN-03 remains incomplete for real NVIDIA/representative quality measurements; CPU protocol fixtures do not establish speedup.

Unchanged m27 prerequisites: no CUDA, five configured weights/foundation paths absent, CMake/FFmpeg absent; Clang available. Do not repeat unavailable probes without a change. Actual Windows x64/MSVC/Resolve/Fusion, clean-machine package, pretrained performance and permitted reference/held-out footage acceptance remain external. No installs/environment changes, large downloads, uploads, publishing or pushes authorized. Existing work/licences/notices preserved.

**Next:** inspect `recovery_store.BundleStore` ownership/budget and `desktop_recovery` interfaces; deliver bounded read-only managed-cache inventory plus explicit safe retention/trim workflow, preserving active leases, source exports and journals. Constructor/context currently creates/prunes staging and applies budget, so cannot be reused for a read-only inventory. Continue total resource budgets, retention/portable backup, advanced corrections, high-bit/managed colour, direct-seek/Stage2/writer recovery, evaluation/packaging and required target host acceptance against [acceptance](acceptance.md).

## Context handoff

No managed subprocess or active subagent task remains. All delegated work is integrated. Full CPU/static evidence covers current source; subsequent documentation/ledger edits require only lightweight checks. Do not rerun unchanged suites or restart milestones after compaction. No supported native compaction action is exposed; request `/compact` once at this safe boundary, then continue this same goal. No goal pause/completion/block status has been requested or claimed.

Authoritative files: `.ralph/checkpoint.json`, `.ralph/plan.json`, `.ralph/issues.json`, [acceptance](acceptance.md). User `.venv` remains untouched; reuse existing `/private/tmp/corridorkey-compatible-venv/bin/python` (Python3.12.15/NumPy1.26.4/Torch2.9.1/Qt6.10.2/Ruff0.17.1/Pyright1.1.414) while applicable. Preserve large pre-existing uncommitted tree; source hashes identify this reviewed batch.
