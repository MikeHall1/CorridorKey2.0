# Findings register

Authoritative consolidated register: **16 findings — 1 P0, 5 P1, 10 P2, 0 P3**. Confirmed code behavior is separated from risks and verification gaps. Severity follows the user’s definitions; ARCH-001 is P0 because an admitted input is actually overwritten, though it requires a cross-job collision. Initial architect lane assigned P1; final independent review agrees with P0 under this taxonomy. No code fixes were made.

Effort: S hours–1 day; M roughly1–3 days; L multi-day/target-dependent. Estimates exclude unavailable hardware procurement and are not commitments. Source paths below are relative to `corridorkey/` unless explicitly rooted. Exact reproduction snippets live in the linked lane report or evidence file; use the compatible interpreter described in verification-evidence.md.

| ID | Priority | Category | Finding |
|---|---|---|---|
| ARCH-001 | P0 | Correctness/data preservation | A batch job overwrites another job’s hint |
| SEC-001 | P1 | Security | Legacy asset loading executes checkpoint-controlled Python |
| INF-001 | P1 | ML correctness | MoGe accepts missing inference encoder weights |
| INF-002 | P1 | ML correctness | Invalid normalization statistics silently pass |
| INF-004 | P1 | ML failure recovery | Broad TypeError retry masks foundation failures |
| UI-002 | P1 | Video timing | Missing FPS is silently invented as24fps |
| ARCH-002 | P2 | Reliability/performance | Expected export growth invalidates report admission |
| INF-005 | P2 | Performance | Combined native refinements repeatedly restart video decoder |
| UI-001 | P2 | Desktop lifecycle | Escape bypasses owned operation shutdown |
| UI-003 | P2 | Desktop workflow | Progressive focus is not the displayed source playhead |
| UI-004 | P2 | Cancellation | Progressive cancel delivery failure is discarded |
| INF-003 | P2 | Model lifecycle | C-RADIO code namespace stays bound to first asset root |
| SEC-002 | P2 | Supply-chain reproducibility | Legacy default download is not revision-pinned |
| QA-001 | P2 | Testing/release automation | No repository CI gate is present |
| QA-002 | P2 | Maintainability | Large static backlog remains behind a no-new gate |
| REL-001 | P2 | Release evidence gap | Target product acceptance remains unverified |

## ARCH-001 — P0: A batch job overwrites another job’s hint
- **Category / locations:** Correctness/data preservation; `batch.py:71–79,108–121; session.py:215–221; output_paths.py:208–214; video_io.py:954–962`.
- **Description and evidence / reproduction:** Actual CPU shared-runtime/EXR proof: two jobs complete; Job B’s hint changes from0.9 to0.4000000059604645. Exact script and coordinator log retained.
- **Impact:** Loss of an admitted original input and silently changed downstream render. Requires a planned output to overlap another job’s dependency; not every batch loses data.
- **Root-cause hypothesis:** Protection is scoped to the current job, not the entire batch.
- **Recommended fix:** Before any write/model load, compare all planned writes with the union of batch sources/hints/corrections/assets, including alias identities and direct/worker entrypoints.
- **Estimated effort:** M.
- **Verification approach:** Cross-job source/hint/correction/link collisions refuse before mutation; noncolliding rerenders unchanged.
- **Confidence:** High — independently reproduced.
- **Details:** [architecture-review.md](architecture-review.md).

## SEC-001 — P1: Legacy asset loading executes checkpoint-controlled Python
- **Category / locations:** Security; `runtime.py:165–179; runtime_paths.py:28–32,44–47; foundations.py:193–215; scripts/convert_to_safetensors.py:23,39`.
- **Description and evidence / reproduction:** Real loader executed harmless checkpoint-selected print before format validation; Semgrep confirms four unrestricted Torch callsites. See evidence/legacy-load-proof.json/.log.
- **Impact:** Code execution with user privileges if a selected legacy asset is untrusted. No remote exploit, compromised asset or actual compromise claimed.
- **Root-cause hypothesis:** Automatic legacy compatibility selects unrestricted pickle without an explicit runtime trust boundary.
- **Recommended fix:** Use restricted tensor loading where compatible; otherwise explicit trusted isolated conversion/admission, preserving documented legacy workflow and safetensors behavior.
- **Estimated effort:** M.
- **Verification approach:** Sentinel cannot execute via normal load; valid legacy migration and safetensors regressions; actionable unsupported errors.
- **Confidence:** High — real loader proof.
- **Details:** [security-test-review.md](security-test-review.md).

## INF-001 — P1: MoGe accepts missing inference encoder weights
- **Category / locations:** ML correctness; `third_party/MoGe/moge/model/v2.py:99–105; foundations.py:29–38,270–272`.
- **Description and evidence / reproduction:** Tiny subclass through real pretrained loader accepts empty model state, retaining randomly initialized encoder parameter. Exact executable snippet in lane report.
- **Impact:** A malformed serialized asset can yield finite meaningless features and successful-looking output.
- **Root-cause hypothesis:** Unchecked strict=False state load; outer trainable-state check excludes foundations.
- **Recommended fix:** Require inference-used encoder state completeness; allow only documented unused-head compatibility, not arbitrary missing tensors.
- **Estimated effort:** S.
- **Verification approach:** Empty/missing/extra/shape-invalid encoder cases fail; approved real checkpoint loads.
- **Confidence:** High — loader branch reproduced.
- **Details:** [inference-performance-review.md](inference-performance-review.md).

## INF-002 — P1: Invalid normalization statistics silently pass
- **Category / locations:** ML correctness; `foundations.py:170–171,193–215,261–278; runtime_paths.py:54–61`.
- **Description and evidence / reproduction:** Real safetensors fixture: wrong mean shape skipped, missing keys defaulted, negative std accepted; logs say loaded.
- **Impact:** Wrong normalization can silently alter all inferred features; negative std clamps to1e-6.
- **Root-cause hypothesis:** Required learned statistics are treated as optional per-key overrides.
- **Recommended fix:** Validate regular file, all enabled-model keys/shapes/finiteness/positive std before mutating buffers; keep optional training mode distinct.
- **Estimated effort:** S.
- **Verification approach:** Missing/directory/shape/NaN/Inf/nonpositive fixtures fail atomically; valid normalization bitwise preserved.
- **Confidence:** High — malformed file reproduced.
- **Details:** [inference-performance-review.md](inference-performance-review.md).

## INF-004 — P1: Broad TypeError retry masks foundation failures
- **Category / locations:** ML failure recovery; `foundations.py:243–246`.
- **Description and evidence / reproduction:** CPU substitute raises internal TypeError with keyword; forward retries without keyword and succeeds. Two calls observed.
- **Impact:** A real failure can be hidden and expensive/stateful work repeated. Frequency with shipped snapshot unknown.
- **Root-cause hypothesis:** Exception class is mistaken for proof of unsupported keyword API.
- **Recommended fix:** Select explicit supported call contract before forward; propagate internal TypeError once, without general retry.
- **Estimated effort:** S.
- **Verification approach:** Internal error one call and visible failure; explicitly supported old/new signatures preserve layout.
- **Confidence:** High — suppression reproduced.
- **Details:** [inference-performance-review.md](inference-performance-review.md).

## UI-002 — P1: Missing FPS is silently invented as24fps
- **Category / locations:** Video timing; `video_io.py:211,217,252,818`.
- **Description and evidence / reproduction:** Real metadata probe with decoder fixture FPS0 succeeds with native_fps24.0.
- **Impact:** Wrong duration and hint/output timing if decoder cannot identify actual rate. Not a proven failure on all video.
- **Root-cause hypothesis:** Fallback replaces invalid timing before positive/finite validation.
- **Recommended fix:** Reject unavailable/nonfinite/nonpositive video FPS; keep explicit sequence FPS separate.
- **Estimated effort:** S.
- **Verification approach:** 0/NaN/unavailable refuse before writer/model; valid fractional FPS unchanged.
- **Confidence:** High — exact branch reproduced.
- **Details:** [desktop-video-review.md](desktop-video-review.md).

## ARCH-002 — P2: Expected export growth invalidates report admission
- **Category / locations:** Reliability/performance; `output_paths.py:17,106–140,154–174; worker_client.py:396–411,449–495`.
- **Description and evidence / reproduction:** 17000 input entries accepted; adding34000 expected alpha/fg outputs causes100000 inventory-limit failure.5000-file scan makes10025 info calls,0.168–0.176s locally.
- **Impact:** Late report failure can close active worker; repeated full scans delay event draining. No GPU slowdown measurement.
- **Root-cause hypothesis:** Expanding inventories are recounted twice and rebuilt on progress hot path.
- **Recommended fix:** Retain protected dependency identities, protect planned subtrees directly, cheaply revalidate destination; explicit safe refresh outside hot path.
- **Estimated effort:** M.
- **Verification approach:** 17k-frame export growth accepted safely; no full unchanged scans per tick; alias/collision tests preserved.
- **Confidence:** High — boundary/count proof; production slowdown unmeasured.
- **Details:** [architecture-review.md](architecture-review.md).

## INF-005 — P2: Combined native refinements repeatedly restart video decoder
- **Category / locations:** Performance; `runtime.py:1491–1500; spatial_draft.py:55–69; video_io.py:505–515,534–546`.
- **Description and evidence / reproduction:** Real five-frame PNG/CPU runtime requests10 reads and5 restarts.240-frame sequential decoder fixture counts29160 decoded instances.
- **Impact:** Duplicate sequence reads; quadratic video decoding for spatial draft+alpha refinement+trimap combination. Real-video elapsed cost unmeasured.
- **Root-cause hypothesis:** Two same-frame consumers independently call a forward-only native reader after its cursor advances.
- **Recommended fix:** Reuse current-frame native guidance in the existing per-frame loop; preserve refinement order, no general cache needed.
- **Estimated effort:** S.
- **Verification approach:** Combined mode one native read/frame, linear decoder counts and identical pixels; real video follow-up.
- **Confidence:** High for duplicate/restart; real-video wall time NOT VERIFIED.
- **Details:** [inference-performance-review.md](inference-performance-review.md).

## UI-001 — P2: Escape bypasses owned operation shutdown
- **Category / locations:** Desktop lifecycle; `desktop_ofx_capture.py:429–436; desktop_progressive.py:444–447; desktop_corrections.py:872–885; desktop_background.py:89–92`.
- **Description and evidence / reproduction:** Actual offscreen Qt Escape hides capture with rendering=True/closing=False and no cancel; progressive remains enabled with active timer.
- **Impact:** Work can continue invisibly; correction close guard bypassed but no proven lost draft.
- **Root-cause hypothesis:** QDialog.reject bypasses closeEvent-only guards.
- **Recommended fix:** Route reject through existing close path, as readiness already does; preserve ownership/dirty/latecommit barriers.
- **Estimated effort:** S.
- **Verification approach:** Escape/titlebar/parent close equivalence for active/idle/cancel/dirty states.
- **Confidence:** High — native Qt proof.
- **Details:** [desktop-video-review.md](desktop-video-review.md).

## UI-003 — P2: Progressive focus is not the displayed source playhead
- **Category / locations:** Desktop workflow; `desktop_progressive.py:188,323–329,387–393; desktop_playback.py:363–364; desktop_window.py:1570–1605`.
- **Description and evidence / reproduction:** Actual Qt playback shows source3 while scheduler stays1 and playing=False. Range[2,4) plus workspace0 raises uncaught ValueError during poll.
- **Impact:** Upcoming-frame refinement is disconnected; trimmed ranges can schedule wrong frame or fail callback.
- **Root-cause hypothesis:** Manual/export-local index is treated as absolute source playhead; playback state not wired.
- **Recommended fix:** Map validated displayed source coordinate and play/pause into scheduler; validate initial focus within visible failure boundary.
- **Estimated effort:** M.
- **Verification approach:** Playback/seek/pause/latest-wins integration; export0 maps source offset; invalid focus actionable no unhandled callback.
- **Confidence:** High — both paths reproduced.
- **Details:** [desktop-video-review.md](desktop-video-review.md).

## UI-004 — P2: Progressive cancel delivery failure is discarded
- **Category / locations:** Cancellation; `desktop_progressive.py:220–226,380–381`.
- **Description and evidence / reproduction:** queue.Full on bridge send, disable, five polls: one cancel attempt, cancel_sent=False, retained active worker.
- **Impact:** Job keeps resources while UI says cleanup pending; ownership retained, so not a proven process leak.
- **Root-cause hypothesis:** Broad except discards failure; disabled recipe returns before retry.
- **Recommended fix:** Show error and retain retryable cancel target; bounded or explicit retry, do not silently kill worker.
- **Estimated effort:** S.
- **Verification approach:** First send failure then success; ownership persists to terminal; permanent failure visible.
- **Confidence:** High — queue-full path reproduced.
- **Details:** [desktop-video-review.md](desktop-video-review.md).

## INF-003 — P2: C-RADIO code namespace stays bound to first asset root
- **Category / locations:** Model lifecycle; `foundations.py:56–82; session.py:105–129`.
- **Description and evidence / reproduction:** Two real imported temporary hf_model snapshots A/B return A code with B path on second load.
- **Impact:** Separate sessions may mix implementation and weights. Desktop fresh-worker transitions avoid this sequence.
- **Root-cause hypothesis:** Constant sys.modules namespace reused without origin check.
- **Recommended fix:** Snapshot-specific namespace/children or explicit different-root restart requirement; clean owned partial import failure.
- **Estimated effort:** S–M.
- **Verification approach:** A→B/B→A/same-root/relative imports/failure recovery; requested origin enforced.
- **Confidence:** High — real import machinery proof.
- **Details:** [inference-performance-review.md](inference-performance-review.md).

## SEC-002 — P2: Legacy default download is not revision-pinned
- **Category / locations:** Supply-chain reproducibility; `scripts/download_weights.py:52–55,90,97; README.md:90–97`.
- **Description and evidence / reproduction:** Default revision=None is passed separately for both CorridorKey files; observed from source only, no network download performed.
- **Impact:** Different installs can obtain different paired assets; mutable publisher state can move between fetches. No upstream compromise observed.
- **Root-cause hypothesis:** Legacy convenience path bypasses newer explicit immutable download-plan workflow.
- **Recommended fix:** Use/recommend existing pinned asset plan; preserve CLI compatibility or require/document explicit revision and paired identity.
- **Estimated effort:** S–M.
- **Verification approach:** Default paired revision immutable; interruption no false complete; recorded provenance and compatibility.
- **Confidence:** High for unpinned default, medium for hypothetical mismatch.
- **Details:** [security-test-review.md](security-test-review.md).

## QA-001 — P2: No repository CI gate is present
- **Category / locations:** Testing/release automation; `Repository tracked-file inventory; scripts/code_health.py:100–160; tests/`.
- **Description and evidence / reproduction:** Initial inventory has no .github workflows/GitLab/Jenkins config. Local1533-case coverage is manually invoked. Remote CI not inspected.
- **Impact:** Regressions/platform drift can merge without these checks. External CI may exist; this audit cannot confirm it.
- **Root-cause hypothesis:** Verification tooling exists without a tracked automation contract.
- **Recommended fix:** Add bounded CPU/static and Windows smoke jobs with optional external GPU/host gates.
- **Estimated effort:** M.
- **Verification approach:** Fresh clean checkout runs compatible deps/tests/checkers; artifacts expose skips/known debt, loopback fixture permission.
- **Confidence:** High for local absence; external automation unknown.
- **Details:** [security-test-review.md](security-test-review.md).

## QA-002 — P2: Large static backlog remains behind a no-new gate
- **Category / locations:** Maintainability; `scripts/code_health.py:100–160; .ralph/static-baseline.json; evidence/static-final/`.
- **Description and evidence / reproduction:** 422 files:512 Ruff findings,684 Pyright errors+17warnings. Gate0, checker1; zero new fingerprints.
- **Impact:** Reduced signal and difficult type-contract review; counts are not confirmed runtime bugs.
- **Root-cause hypothesis:** Historical classified baseline prevents new regressions but does not retire old debt.
- **Recommended fix:** Triage runtime-relevant diagnostics in owned batches; preserve checker scope and no global suppression.
- **Estimated effort:** L, incremental.
- **Verification approach:** Each batch focused tests plus decreasing classified backlog and no new findings; no mass baseline refresh.
- **Confidence:** High for counts; individual defect significance needs triage.
- **Details:** [security-test-review.md](security-test-review.md).

## REL-001 — P2: Target product acceptance remains unverified
- **Category / locations:** Release evidence gap; `desktop_bundle.py:118–120; docs/acceptance.md; benchmarks/suites/; native/ofx/`.
- **Description and evidence / reproduction:** No CUDA/weights/FFmpeg here; preflight refuses wrong platform/missing builder. CPU/Qt/native harnesses do not execute Windows/Resolve/trained pipeline.
- **Impact:** Cannot approve beta/production quality, speed, Windows packaging or host compatibility. This is an evidence gate, not an observed target failure.
- **Root-cause hypothesis:** Required hardware/assets/host validation has not been supplied for this audited state.
- **Recommended fix:** Run staged target CUDA/reference/FFmpeg/Windows/Resolve acceptance with exact build/settings/provenance; review final distribution inventory.
- **Estimated effort:** L, external prerequisites.
- **Verification approach:** Cold/warm/quality/temporal pairs, clean Windows render/cancel/recover, actual Resolve viewer/export and managed colour cases.
- **Confidence:** High for current limits; target functionality unknown.
- **Details:** [security-test-review.md](security-test-review.md).

## Hypotheses and non-findings

No measured GPU speedup or visual-quality improvement was found. Discarded review-stack allocations, unbounded legacy journal reads and broad module size warrant targeted follow-up only; they are not additional confirmed ranked defects here. Intentional local C-RADIO code loading is a trust requirement, not automatically a vulnerability. Existing resolved ledger issues were not relisted as current defects. No dead-code deletion or dependency removal is proposed without reachability evidence.
