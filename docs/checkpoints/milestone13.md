# Run checkpoint — milestone 13, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, confirmed by milestone12 source/checkpoint and fresh249-test baseline. Milestone12 archived in `checkpoints/milestone12.md` and `.ralph/checkpoints/milestone12.json`. No competing supervisor/extra loop was started. All work stays local/uncommitted; no push, publish, deploy, media upload, dependency upgrade, weights/footage download or user `.venv` modification occurred.

## Completed cycles

1. Fresh baseline passed249 tests/gate. Added immutable canonical whole-job EXR replay identities: source/hints/corrections, all validated settings/writer/profiling fields, model/foundation/stat files, implementation and explicit backend environment. Destination is excluded; path/config/model/source changes invalidate conservatively. Unsupported automatic scheduling/video and incomplete context fail explicitly. Five identity tests; full254 tests/gate passed.
2. Added bounded immutable file bundles, checked payload inventory, owned private staging, atomic complete publication, fresh restore destinations, SQLite exclusive ownership and file/POSIX directory sync. Reproduced/fixed partial owner-record crash cleanup while preserving unknown/symlink/user content. Actual child exit73 proves owner release and recognized staging cleanup. Six store tests; full260 tests/gate passed.
3. Added checksummed per-request attempt journals, running/interrupted/failed/cancelled/complete transitions and verified completed-output checks. Partial files alone never establish success. Output capture requires complete native EXR indices. Reproduced/fixed new test typing/lint. Four journal tests; full264 tests/gate passed.
4. Added working `python -m corridorkey.recovery run/status` on shared session/batch requests. Run/resume distinguish new render, verified byte replay and verified existing outputs; preserve old/partial/user output data and choose fresh siblings. Cache-corrupt data is recomputed only under owned-file rules. Identity changes during render prevent publication. Zero budget keeps journals without cache admission. Status is read-only metadata, explicitly NOT VERIFIED for payload/backend. Fixed callback binding/factory typing. Seven real shared-session/native-EXR fixture tests; full271 tests/gate passed.
5. Reproduced/classified/fixed valid-JSON completion metadata tampering, file/parent/case path conflicts, strict schema types and declared-size copy checks. Added actual hard child exit74 and cancellation/disk-full tests. Fixed a new test import error. Audit also found model identity could differ from independently resolved/retained session assets: now actual asset paths and batch model/backend/implementation context are pinned, with explicit fresh-session resume on changes. Added commands/docs and ran generated CPU replay/status evidence, then final verification.

## Final actual verification

- **280/280 tests PASS**, no skips; `.ralph/runs/manual-check/check-0.txt`.
- Compilation/whitespace PASS; `.ralph/runs/manual-check/verification.json`.
- **131 Python files /0 syntax errors**, selected runtime imports PASS, declared requirements/VCS/installed dependency checks PASS.
- Ruff **523 findings**, Pyright **826 errors /17 warnings** remain. **Zero new diagnostic fingerprints** against reviewed compatible baseline. Full repository is **not static-clean**; no baseline refresh, broad ignores or checker weakening.
- All four new recovery modules and four test files: focused Ruff/Pyright **zero findings**;31 recovery tests pass. Pylance editor **NOT VERIFIED**. JS/TS/native diagnostics not applicable to current owned source scope.
- Same isolated Python3.12.15/NumPy1.26.4/OpenCV4.11/Torch2.9.1 and pinned Ruff0.17.1/Pyright1.1.414, unchanged runtime requirements. No installs this run. Temporary environment may need recreation.
- Fresh baseline archived at `benchmark-results/m13-baseline-n0qmfn0f/checks/`; cycle1–4 health/tests in `m13-cycle*-health/`/`m13-cycle*-tests.txt`; final command `m13-final-verification.txt`.
- Generated CPU shared-session/EXR fixture observed **1 session load /1 export job** for2 identical requests (second restored), followed by2 verified-existing jobs on resume; restored bytes match. `benchmark-results/recovery-m13-generated-summary.json` records reports/context/status CLI. These are actual storage/controller work counts, **not** neural/GPU throughput or visual-quality measurements. Actual process crash/cancel/disk-failure tests are local CPU evidence.

## Problems and root fixes

`.ralph/issues.json` records resolved HEALTH-RECOVERY-023 (partial-stage ownership),024 (journal test contracts),025 (synchronous callback false-positive evidence and explicit partial binding),026 (callable session-factory contract),027 (inventory/metadata/size/version integrity),028 (new test import),029 (actual loaded model asset lifetime binding). Each has pre-fix evidence/root action/checks. Existing HEALTH-STATIC-005 remains open for risk-based triage; documented pre-existing false positives remain visible. No inference default/option, precision, backend fallback or neural algorithm changed.

## Operational scope and limits

`docs/recovery.md` gives concrete run/resume/status commands. Production run requires local model assets and CUDA/driver context, EXR and explicit scheduling (`auto_vram=false`). No source/model downloads or CPU/MPS inference fallback occurs. Actual session-loaded asset paths must match pinned identities; asset/backend/implementation changes within a batch stop it until resume creates a new session. Hashes before/after detect observed changes; inputs/dependencies must remain stable, without a globally atomic snapshot guarantee.

Cache entries are immutable derived artifacts with payload/complete-record checksums and strict names. Eviction/cleanup removes recognized owned files only; unknown content stops deletion. SQLite releases ownership on actual process exit. Journals commit independently and require explicit resume. Verified completed outputs can be retained, otherwise bundles restore or jobs restart whole into fresh directories. Existing renders/partial/user data are never overwritten by this controller. Replay origin records/timings are historical, not new inference measurements.

The byte budget covers logical sealed bundle files/records; filesystem allocation/atomic-record temporaries, journals/reports, render copies, source/weights and RAM/VRAM are outside it. It is not a total disk/process cap. Hashing can be expensive; no speedup is claimed. POSIX file/directory sync is requested; Windows directory flush, real power-loss and target-platform behavior remain NOT VERIFIED. Process-crash tests do not complete those requirements.

## Next action and full remaining scope

Recovery remains **in progress**. Next integrate durable per-frame IID C-RADIO/MoGe reuse at `FoundationWindowCache`, with stage-specific identities and exact shape/channel/dtype/checksum/budget guards. Test stale, corrupt, cancellation and disk-failure behavior. Preserve chronological RVM and original defaults. Recurrent state checkpoints/mid-shot continuation are unfinished; do not expose resume controls implying them. Then integrate recoverable queue/cache behavior into desktop workflows.

Evaluation still needs multi-report-per-render support, broader stress/photographic/independent-flow suites, alternating timing automation and held-out/blinded validation. Higher-bit video/managed source colour, desktop/editor/queue, packaging and official SDK integration remain incomplete. CUDA/model assets/permitted representative/held-out footage/training data and Resolve SDK/Windows/NVIDIA host acceptance remain external/unvalidated here. CPU/static/generated/neural-substitute evidence does not satisfy those criteria. The full product goal remains active and incomplete.

Resume by reading the objective, AGENTS, current ledgers/code-health policy, establishing another actual baseline and taking the next bounded recovery increment. Do not start a competing loop to evade the five-cycle cap.
