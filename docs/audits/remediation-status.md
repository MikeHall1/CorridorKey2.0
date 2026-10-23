# Audit remediation status — 2026-10-24

This is the implementation follow-up to the historical read-only audit. Original findings remain an evidence record; this document records subsequent fixes. No pretrained/GPU/Windows/Resolve acceptance is implied.

## Implemented changes

| Finding | Mechanism / regression proof |
|---|---|
| ARCH-001 | Batch-wide union of source, hint, correction/document and selected model paths is admitted before direct/worker/recovery mutations. Existing output-layout and alias checks reused. `test_audit_safety` covers cross-job hints, hardlinks, no-child-before-refusal and selected nested model protection. |
| ARCH-002 | `prepare_report_destination` retains admitted protected identities; normal publication rechecks only destination/path ancestry. New multiply-linked destinations trigger full inode refresh, so newly generated export hardlinks remain protected. Inventory counts each protected child once. `test_audit_safety` verifies no rescans, late symlink refusal and new-export hardlink protection. |
| SEC-001 | Runtime checkpoint, stats and both conversion loaders use `weights_only=True`. Real harmless pickle payload cannot execute at any of the four boundaries. Valid tensor/dict legacy conversion remains tested. |
| INF-001 | MoGe requires complete state for its declared architecture rather than silently leaving random encoder weights. Wrong/missing/extra/shape-incompatible state fails. Actual approved checkpoint compatibility remains NOT VERIFIED. |
| INF-002 | Enabled foundation statistics require exact shapes, real finite values and positive stds; whole file validates before buffer mutation. Runtime checkpoints cannot contain `foundations.*` entries that overwrite separate approved model/stats assets. |
| INF-003 | C-RADIO modules and relative imports are scoped to resolved snapshot root; failed owned imports are removed. A/B and same-root tests preserve identity. |
| INF-004 | Official C-RADIO call uses its supported no-keyword forward signature once; existing token→spatial conversion remains. Internal TypeError propagates rather than triggering a hidden retry. |
| UI-002 | Unknown/invalid video FPS refuses instead of becoming24. Explicit sequence FPS behavior remains separate. |
| INF-005 | Current-frame native guidance is reused across alpha and trimap refinement in the existing loop. Five-frame fixture now requests5 reads with0 restarts instead of10 reads/5 restarts. |
| UI-001 | Capture, progressive, correction and background dialogs route rejection/Escape through existing close/dirty/ownership barriers. |
| UI-003 | Progressive focus consumes accepted playback source identity/play state, maps export-local offsets, validates ranges and reports errors visibly. Expected superseded preparation/validation is cleaned without disabling the feature. |
| UI-004 | Cancel-delivery failures stay visible and owned; existing retained-cleanup control explicitly retries the target. Repeated shutdown does not silently resend. |
| SEC-002 | Legacy branch/tag/default resolution happens once to an immutable40-character publisher commit; both CorridorKey files use that commit. Existing explicit revision/repository flags remain. Mock publisher tests perform no downloads. |
| QA-001 | Tracked read-only GitHub CPU matrix: Python3.12, declared CPU Torch/dependencies, full tests, source worker smoke, visible full static-regression gate on Linux/Windows. Official action revisions pinned after read-only tag lookup. Hosted execution not yet verified; no custom artifact upload or telemetry added. |

## Deliberate compatibility changes

- Legacy checkpoint dictionaries containing unsupported executable/custom Python objects now refuse. The tensor/primitive public runtime format and safe conversion path remain supported; do not re-enable unrestricted pickle to bypass an error.
- MoGe incomplete state and invalid statistics now fail rather than producing successful-looking results. Runtime-format checkpoints containing frozen `foundations.*` tensors explicitly refuse; use the separately declared foundation assets. Approved pretrained compatibility needs the actual assets.
- Unknown video FPS now produces an invalid-metadata error, never a guessed playback/render duration.
- Selected model directory links/reparse entries are refused by safety admission rather than traversed ambiguously. This matches existing recursive report protection.
- CI defines a merge-check mechanism but branch protection settings and hosted platform results are not changed or claimed.

## Open findings

**QA-002 — OPEN, classified legacy debt.** No checker suppression, baseline refresh or mass refactor was used. Broad raw diagnostic counts remain visible. A bounded diagnostic review found guarded/unreachable apparent unbound states in `draft.py`, `shot_analysis.py` and sequential video reads, plus missing type narrowing in colour conversion. A real short/unequal-input defect exists in vendored `MoGe/moge/utils/tools.py::strip_common_prefix_suffix`, but there are no repository callers. It is recorded for upstream/dedicated maintenance, not silently changed solely to lower counts. Removing every legacy type/style diagnostic safely is a distinct incremental workstream; this task does not claim it done.

**REL-001 — NOT VERIFIED.** Actual CUDA/model quality/parity/VRAM/throughput, real FFmpeg output, actual optional segmentation, clean Windows bundle/upgrade, MSVC/Resolve viewer/export/retime/proxy/managed-colour acceptance and final distribution approval require unavailable prerequisites. No asset download, environment migration or invented measurements were substituted.

## Evidence and review

Regression-first failures and passing focused runs are in `benchmark-results/audit-fixes/` (local generated evidence). Model11-test and Qt63-test lanes passed. The integrated1566-test run passed before the final selected-root hardlink addition; a subsequent integrated run and final static gate record that additional state. Final stable gate: **1568 tests PASS/no skips in131.430s;427-file static regression gate PASS with0new findings**. Raw diagnostics remain511 Ruff and684 Pyright errors+17warnings. Machine-readable summary: `benchmark-results/audit-fixes/verification-summary.json`; reviewed source identity: `734d804b0b07589f79e4809abc35833c5e1b47302ba87d2801db1f67388dd135`.

The code reviewer approved18-file scoped contracts after finding/fixing post-admission foundation overrides. Architecture review independently found selected-root nested hardlink and linked-child gaps; both use existing model inventory/alias ownership rather than disabling guards. Final verdict and source identity are recorded with the verification summary.

Official references used for CI: [workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow), [setup-python](https://github.com/actions/setup-python), [checkout](https://github.com/actions/checkout). Installed huggingface-hub0.36.2 `HfApi.model_info` signature/docstring was inspected before implementing single-revision resolution. No network model request executed.
