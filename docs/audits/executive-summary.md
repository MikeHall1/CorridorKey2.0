# Executive summary

**Subsequent implementation:** see [remediation status](remediation-status.md) for fixes after this historical read-only audit.

**Recommendation: suitable for controlled testing only. Not beta-ready or production-ready.** Use trusted model assets and disposable media copies; do not run batches whose outputs overlap another job’s inputs. The audit made no application changes.

## Top10 priorities

| Rank | Finding | Decision |
|---:|---|---|
|1| **ARCH-001 P0 — batch overwrites another job’s hint** | Fix batch-wide input preservation first. Actual shared-runtime EXR proof changed alpha0.9→0.4 while reporting success. |
|2| **SEC-001 P1 — unrestricted legacy checkpoint execution** | Establish explicit trusted-asset/migration boundary. Harmless real-loader proof executed checkpoint-controlled Python. |
|3| **INF-001 P1 — missing MoGe encoder weights accepted** | Reject incomplete inference state before rendering. |
|4| **INF-002 P1 — invalid/missing foundation statistics accepted** | Validate all required statistics atomically; no silent defaults. |
|5| **INF-004 P1 — broad TypeError retry masks model failure** | Choose supported API contract before inference; propagate actual failures. |
|6| **UI-002 P1 — unavailable FPS becomes24fps** | Reject unknown video timing rather than inventing native metadata. |
|7| **ARCH-002 P2 — report scans reject growing valid exports** | Keep collision protection without rescanning full expanding inventories on progress. |
|8| **INF-005 P2 — duplicate native guidance restarts decoding** | Reuse same-frame data; preserve refinement order and parity. |
|9| **UI-001 P2 — Escape bypasses owned shutdown** | Route dialog rejection through existing cancellation/dirty guards. |
|10| **UI-003 P2 — progressive focus misses displayed source time** | Connect playback and partial-range source coordinates to scheduler. |

Six further findings cover lost cancel-delivery errors, C-RADIO snapshot code reuse, unpinned legacy downloads, absent repository CI, static debt and external release evidence. See the [16-entry findings register](findings-register.md) for exact locations, evidence, effort, fixes and tests. P0 follows the requested data-loss definition; it is conditional on the reproduced cross-job collision, not a claim every render is destructive.

## Health and maintainability

The shared Python engine, optional Qt UI, bounded worker protocol, durable queue/recovery and prepared-cache OFX boundaries are coherent. Strong fixture/fault-path testing exists. The key weaknesses are composition-boundary safety, permissive model admission, duplicated hot-path work and UI lifecycle integration. Fix these in existing responsible layers rather than rewriting or adding abstractions.

## Evidence

- Fresh full discovery: **1533 tests**,15 sandbox-denied localhost bind errors. Targeted escalated rerun: **21 tests pass**, covering all failures. No assertion failure remains in combined evidence; not a single green full unsandboxed run.
- **422 Python files**: no-new static gate passes; **512 Ruff diagnostics,684 Pyright errors+17warnings** remain. Not static-clean.
- Source worker handshake/status/clean shutdown passes; offscreen Qt and native OFX harness checks run. Windows packaging preflight correctly refuses this host.
- Decoder experiment: **9/9 exact-parity cases**, generated64×64 frames, three measured pairs. Ratios vary; no production speedup claimed.
- Local Semgrep:4 targeted rules,203 Python files,5 matches manually interpreted. Not a full CVE/supply-chain audit.
- Separate reviewer reproduced four inference findings and retained all16 findings. Architecture **BLOCK**, code reviews **REQUEST CHANGES** for readiness.

## Performance and security

Measured local report-admission scans and deterministic repeated-decoder counts justify targeted changes. GPU/VRAM/real-time and real-footage quality remain unmeasured. Legacy pickle execution is a confirmed local trust risk, not a remotely demonstrated exploit. C-RADIO intentionally executes snapshot code; use of safetensors alone does not make its Python implementation trusted. No actual compromise was observed.

## Feature/release assessment

Source workflows, corrections, recovery, evaluation and EXR contracts have substantial local evidence. Neural extraction/native-resolution quality lack pretrained validation. Tracking, click-provider guidance, progressive playback, broader video timing, portability and host-assisted preparation remain partial or target-unverified. The existing OFX path is prepared-cache/capture integration, not live neural inference inside Resolve.

CUDA, weights, FFmpeg, actual segmentation provider, Windows bundle/clean-machine operation and Resolve viewer/export acceptance were unavailable. macOS CPU/Qt checks are not substitutes. Final distribution/licence review remains a release gate.

## Next implementation lanes

1. **Safety:** cross-job preservation, report admission and trusted asset boundary.
2. **ML:** strict state/statistics/API contracts, snapshot isolation, native guidance reuse.
3. **Qt:** rejection/cancellation, playback focus and partial-range integration.
4. **QA/release:** CPU/Windows CI, visible static-debt triage and target acceptance.

Run disjoint ownership with at most two implementation workers under repository policy, plus a coordinator and bounded reviewer as capacity permits. The [roadmap](improvement-roadmap.md) defines11 packages and release gates. No fixes, installs or releases were performed in this audit.
