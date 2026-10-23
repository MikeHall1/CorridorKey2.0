# Independent adversarial audit review

Reviewed 2026-10-24 against the dirty checkout based on `7d808e61d8a42d77e30792ded6911aabca96a08f`. This is a bounded independent evidence/severity review, not another implementation pass. No application, test, configuration or authoritative ledger was edited.

## Code Review Summary

**Files Reviewed:** 13 production source boundaries, eight audit reports (including the final findings register) and the architecture reproducers; supporting specification and raw verification records. Large source files were inspected at the relevant call/validation/write boundaries, not exhaustively reread.
**Total Issues:** 16 register entries retained: 13 correctness/security/reproducibility findings plus 3 process/debt/acceptance gaps. No additional runtime defect identified by this review. QA-001, QA-002 and REL-001 are not 3 more reproduced runtime failures.

### By Severity

- CRITICAL / P0: 1 — ARCH-001.
- HIGH / P1: 5 — INF-001, INF-002, INF-004, UI-002, SEC-001.
- MEDIUM / P2: 10 — ARCH-002, INF-003, INF-005, UI-001, UI-003, UI-004, SEC-002, QA-001, QA-002, REL-001.
- LOW / P3: 0.

### Recommendation

**REQUEST CHANGES for release/production readiness; COMMENT on the read-only audit deliverable.** The audit may conclude with its defects and unavailable acceptance explicitly recorded. This is not an instruction to repair the application during the audit. The architect's **BLOCK** remains supported.

## Severity challenges resolved

### ARCH-001: retain P0 under the user's explicit data-loss taxonomy

`corridorkey/batch.py:71–79,108–121` validates distinct output roots but composes only job-local preservation checks (`corridorkey/session.py:215–221`). The actual writer replaces an existing matching EXR (`corridorkey/video_io.py:954–962`). The supplied reproduction runs the real batch/session/runtime/writer with a CPU neural substitute: a pre-existing second-job hint is changed from approximately 0.9 to 0.4 and then consumed by that job while the batch reports success.

This is confirmed destruction of an admitted input, not merely hypothetical collision risk. The user defined P0 to include **data loss**, so P0 is justified even though the architect initially used P1. Keep the prerequisite prominent: one job's configured output must overlap another job's source/hint/correction asset. Do not imply every batch loses data or that arbitrary paths are overwritten without this relationship.

The proposed root-cause repair is appropriately batch-wide and pre-write; do not weaken it to warning after publication or assume the user intended chained jobs. Shared-layer ownership/design belongs to the architect/coordinator. Preserve existing noncolliding rerender behavior and alias protection.

### SEC-001: retain conditional P1, not universal P0 or remote exploitation

`corridorkey/runtime.py:165–179` explicitly disables restricted checkpoint loading; format validation occurs later at `:186–187`. `corridorkey/runtime_paths.py:31–35` automatically selects a legacy file when no safetensors alternative exists. The recorded sentinel proves a checkpoint-selected callable runs before validation. A harmless callable is a safe demonstration of that capability; it does not make the mechanism harmless.

However, exploitation requires loading an attacker-controlled/untrusted local asset. No network service, compromised publisher, downloaded malicious artifact or real compromise was demonstrated. Trusted legacy compatibility is deliberate, but the inspected loader and user option documentation provide no explicit trust acknowledgement that separates executable legacy assets from ordinary tensor data. P1 is defensible; P0/remote-RCE wording would overstate the evidence. Deliberate execution of C-RADIO snapshot Python is a distinct, documented integration boundary, not another independently proven vulnerability.

Recommend restricted loading or an explicit trusted migration/admission path while retaining legitimate compatibility. Do not promise that changing the runtime loader alone makes arbitrary local model-code snapshots safe.

## Evidence and root-cause reconciliation

| Findings | Independent assessment |
|---|---|
| INF-001 / INF-002 | Confirmed missing/invalid foundation data can be accepted. The outer trainable-state check excludes foundations, so it does not establish foundation completeness. Preserve the narrow distinction between required inference tensors and legitimately unused prediction heads. Stats validation must be atomic and apply to enabled foundations; identity defaults must not silently replace required runtime data. |
| INF-003 | Confirmed process-global module identity reuses A's implementation with B's requested root. Separate sessions within one process make this reachable; fresh-process desktop paths reduce exposure but do not disprove it. |
| INF-004 | Confirmed arbitrary internal `TypeError` is swallowed and inference retried. This fails the root-cause guard even though frequency on the actual published C-RADIO model is unknown. Retain supported API compatibility only through a documented narrow boundary. |
| INF-005 | Two same-frame guidance reads and the forward-reader restart establish the defect. The 29,160 decoded instances are a cursor-semantics fixture result, not measured real-video throughput. Reuse within the existing frame loop; no new general cache is justified. |
| UI-001 / UI-003 / UI-004 | Supplied actual Qt reproductions and source support rejection bypass, stale/unmapped progressive focus and suppressed cancellation delivery. Retained ownership is positive, but does not mean cancellation was delivered. Do not call these proven process leaks or permanent draft loss. |
| UI-002 | `video_io.py:211` substitutes 24 before the positive/finite FPS check. Invalid timing becomes apparently native timing. Explicit refusal is the minimal repair; another guessed rate would mask the same failure. |
| ARCH-002 | Repeated inventory counts both source and generated children, causing accepted output growth to exhaust the shared bound. Source corroborates progress-publication propagation into worker cleanup. Empty-file proof establishes inventory rejection, not real render timing. Any optimization must retain hard-link and changed-destination protection; deleting scans without replacing that contract is insufficient. |
| SEC-002 | Unpinned default CorridorKey revision and separate paired downloads are supported by `scripts/download_weights.py:49–52,90,97`. Keep as P2 reproducibility/compatibility risk; no actual incompatible published pair or supply-chain compromise was observed. |

No confirmed primary finding needs removal. Do not promote QA-001/002 diagnostic/process debt into thousands of proven bugs, and do not present REL-001 unavailable hardware/host acceptance as a reproduced runtime defect.

## Validation performed by this reviewer

- Read source boundaries in `batch.py`, `output_paths.py`, `session.py`, `runtime.py`, `runtime_paths.py`, `foundations.py`, `spatial_draft.py`, `video_io.py`, `desktop_progressive.py`, `desktop_ofx_capture.py`, `worker_client.py`, `third_party/MoGe/moge/model/v2.py` and `scripts/download_weights.py`.
- Independently executed the first Python reproduction block in `inference-performance-review.md` using `/private/tmp/corridorkey-compatible-venv/bin/python`. Observed: wrong-shaped stats skipped and negative std retained; A then B loads A's implementation twice; empty MoGe encoder state accepted; internal C-RADIO TypeError followed by successful fallback call. No real weights or GPU used.
- Read architecture reproducers and traced their public admission/session/write paths. Did not run a third copy of the already independently/coordinator-observed destructive-input fixture or recreate 51,000 empty files merely to repeat evidence.
- Compared every original existing file in `initial-state.json` against current SHA256: **no original file content changed** at review time.
- Read final static gate JSON: 422 Python sources, no syntax errors, regression gate success, **512 Ruff findings and 701 Pyright diagnostics (684 errors +17 warnings)**; both raw checker exits are 1, zero new fingerprints. These are diagnostics, not a static-clean result.
- Accepted the documented full-suite plus narrow escalated socket rerun as combined coverage, not as a single wholly green full-suite invocation. No full-suite rerun performed.

## Audit-document corrections reported to coordinator

1. Verification interpretation originally said packaging preflight exit 1; raw JSON and table say **2**. Correct the prose, not the retained evidence.
2. WP05 originally proposed removing unspecified discarded review-stack work. Align it to the demonstrated same-frame native-guidance duplication, without inventing an additional optimization finding.
3. UI-003's initial-focus source citation should include `desktop_progressive.py:188`, not `:194` (which starts `_engine_available`). The main `:327` and `:387–393` citations are correct.

The coordinator corrected items 1 and 2 before final register review and acknowledged item 3 for both citations. These are bounded audit corrections, not reasons to discard the underlying findings or to modify runtime code during this task. The final register's 16 entries and 1/5/10 severity totals agree with this assessment.

## Coverage limits and remaining risk

The evidence supports a connected shared-engine implementation with important composition and admission defects, not production readiness. Broad areas appropriately remain unverified: pretrained model semantics/visual quality, CUDA numerical and memory behavior, real video timestamps/FFmpeg exports, optional provider execution, actual Windows packaging and Resolve behavior. Passing CPU fixtures do not establish those contracts. The security scan is a small local-rule review, not a dependency vulnerability audit; large-file recovery/hostile-filesystem and every retained static diagnostic were not exhaustively evaluated.

Not checked: no new full suite, third-party vulnerability lookup, real GPU/media/Windows/Resolve run or exhaustive rereview of all 422 Python files.
Risk: confirmed cross-job input loss and implicit untrusted-checkpoint execution remain unfixed; controlled tests should use trusted assets and disposable media copies.
