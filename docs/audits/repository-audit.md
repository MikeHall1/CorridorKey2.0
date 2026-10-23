# Comprehensive CorridorKey repository audit

## Verdict

**Suitable for controlled testing, with safeguards; not beta-ready, release-candidate or production-ready.** The repository contains a substantial, connected implementation with strong CPU/fault-path testing and a sound shared-engine direction. It also contains reproduced destructive cross-job behavior, silent model-admission faults, timing/cancellation defects and unfulfilled target-platform acceptance. Use trusted assets and disposable copies; avoid colliding batch inputs/outputs until ARCH-001 is fixed.

This is a review only. No application code, test code, original files or development-ledger statuses were changed. Existing dirty work was preserved. Audit outputs are under `docs/audits/`; the existing decoder runner created only its fresh generated benchmark directory. No commit, push, release, download, dependency change or supervisor loop was performed.

## Discovery and review identity

- Repository root: `/Users/mike.hall/Documents/vsCode/corridorKey2_fork`.
- Branch `review/corridorkey-audit-20261023`; HEAD `7d808e61d8a42d77e30792ded6911aabca96a08f`. Recent history includes bounded export indexes, photo backgrounds, foreign collected projects, calibration and tracking diagnostics.
- Audit includes existing dirty runtime, desktop, native OFX, tests and docs. Snapshot hashes/status in [initial-state.json](evidence/initial-state.json); final preservation check in evidence. HEAD alone is not the audited source identity.
- Applicable AGENTS contract and development/product briefs, architecture/options/status/code-health, checkpoints and task ledger were consulted across lanes. Latest user instruction explicitly overrides implementation continuation. Ledger has397 issues;7 are open/documented, and ten extension tasks remain in progress. Prior1533-test claims were treated as historical, not reused as current proof.
- Origin is the MikeHall1 fork; upstream nikopueringer/CorridorKey2.0. Local history inspected, remote issue/CI state not fetched. No GitHub write operation.

## Repository map and ownership

Initial nonignored inventory:706 files;156 `corridorkey` Python modules;219 Python test/support files;116 docs;52 vendored third-party files;7 native OFX files;15 benchmark manifests. Configuration/orchestration artifacts are not production engine code.

| Area | Role / execution path | Review depth |
|---|---|---|
| `infer.py`, `run_interactive.py`, drag/drop shell/batch launchers | CLI/interactive settings → JobRequest/InferenceSession → shared runtime | Entry/admission/output contracts traced; launch/platform declarations inspected |
| `corridorkey/config.py`, `session.py`, `runtime.py`, model/stages/foundations | Validated config, persistent model lifecycle, precompute/temporal/spatial inference, exports | Deep high-risk paths, tiny model proofs; no real pretrained inference |
| `worker*`, `app_process.py`, queue/recovery/durable modules | Single-owner process, bounded framed JSON, explicit durable replay | Source boundaries and broad fault/integration tests; not all race schedules |
| `desktop*`, correction editor/canvas, playback/progressive | Optional Qt Widgets, background ownership and preview/correction workflows |25 primary files deeply reviewed; actual offscreen proofs; broad tests cover adjacent modules |
| guidance/tracking/clean plate/calibration/portability | Optional connected services, provenance and bounded work | Interface/call-path and feature-map review, existing CPU tests; real provider/quality absent |
| `video_io.py`, colour/refinement/output paths | OpenCV decode, FFmpeg video, native EXR, linear/alpha contracts | Deep timing/source-protection/refinement path review; real EXR and generated decoder proof |
| `native/ofx`, `third_party/OpenFX` | Prepared-cache playback and upstream capture companion; official headers/provenance | Native source, strict Clang/harness/sanitizer tests; not Resolve/MSVC |
| `third_party/MoGe`, `vfms`, `weights` | Vendored model code vs external asset locations | Integration/loader and licence boundaries; not every vendored numerical implementation |
| `tests`, `benchmarks`, scripts | Unit/integration/Qt/native fixtures, evaluation and command harnesses | Full discovered suite plus focused proofs, actual decoder experiment, static gate |
| `.ralph`, `.omx`, `.codex`, `.serena` | Historical task state/local tooling | Metadata orientation only; no orchestration repair, activation or ledger mutation |
| `benchmark-results`, caches, temp products | Generated historical evidence, not shipped runtime | Relevant evidence inspected; no exhaustive artifact audit or cleanup |

### Dependency and platform map

Runtime requirements: NumPy<2, OpenCV, Transformers<5, timm, safetensors, huggingface-hub, einops, SciPy, Pillow, tqdm and utils3d pinned to a Git commit. Torch/torchvision installed separately per README (tested2.9.1/0.24.1). Optional Qt Essentials6.10.2; PyInstaller6.22.3 Windows builder; Ruff/Pyright pinned developer tools. Native C++ OFX uses vendored headers. FFmpeg is an external executable, not present here.

Full inference is CUDA/NVIDIA oriented. CPU supports correctness/adapter/model-substitute checks, not an implicit production fallback. macOS ARM local checks do not establish Apple inference or Windows operation. Linux source launcher and Windows drag/drop/bundle recipe exist; actual Windows x64 packaging and Resolve remain target-gated. No web frontend, HTTP application service or database backend is implied; SQLite is local ownership/queue/cache state.

## Findings and evidence

The [register](findings-register.md) is authoritative: **16 findings,1P0/5P1/10P2**. Priorities emphasize source preservation and honest model/timing contracts, not style or speculative speed. Specialist details are intentionally not duplicated here:

- [Architecture](architecture-review.md): batch-wide safety composition and report-admission scaling; status **BLOCK**.
- [Inference/performance](inference-performance-review.md): five confirmed loading/normalization/lifecycle/refinement issues; **REQUEST CHANGES**.
- [Desktop/video](desktop-video-review.md): four timing/lifecycle/playhead/cancel issues; **REQUEST CHANGES**.
- [Security/testing](security-test-review.md): conditional unsafe legacy loading, download reproducibility, manual CI/static debt and release evidence.
- [Feature assessment](feature-completeness.md): implemented does not mean pretrained or host verified.

Actual CPU EXR batch proof overwrote a later job’s declared hint from0.9 to0.4 while reporting completion. That meets the requested P0 data-loss definition despite requiring an input/output collision. Tiny checkpoint/snapshot proofs establish model-loader permissiveness without claiming measured visual damage. Qt proofs use real widgets; mocked decoder metadata proves a specific FPS branch, not failure of every video file.

## Verification result

Fresh suite:1533 tests,15 sandbox localhost-bind errors; targeted escalated21-test rerun passes every affected case. Static422-file no-new gate passes while512 Ruff and684 Pyright errors/17warnings remain. Real source worker handshake/shutdown passes. Decoder benchmark9/9 exact parity on generated64×64 data; varied ratios and concurrent host activity cannot establish production speedup. Semgrep4 local rules/203 Python targets finds5 trust-boundary callsites. Package preflight correctly refuses this host.

See [verification-evidence.md](verification-evidence.md) for exact commands, exit codes, durations, original failed logs and limitations. No single wholly green full unsandboxed suite is claimed. No CUDA/weights/FFmpeg/Windows/Resolve/optional provider execution was available.

## Roadmap and implementation handoff

[Improvement roadmap](improvement-roadmap.md) provides11 dependency-aware packages, acceptance criteria and delegation boundaries. First fix batch input preservation, trusted legacy admission, model completeness and Qt ownership. Then remove duplicate native reads and hot report inventories; integrate source-time progressive focus. Add CPU/Windows CI while retaining visible static debt. Only then pursue measured GPU optimization and target release acceptance.

Avoid a rewrite: one shared engine, explicit quality modes, bounded work ownership and immutable provenance are strengths. No unsubstantiated dead-code removal, dependency replacement or generic service layer is recommended. Advanced models/backends are optional work after correctness and reproducible evidence, not a shortcut to product acceptance.

## Independent review and audit limits

The final reviewer independently re-executed foundation proofs, checked key source paths and original-file hashes, calibrated severity, and corrected report-only inaccuracies (packaging exit2 and a Qt line reference). See [independent review](evidence/independent-review.md). No automatic implementation followed the findings.

This is repository-wide coverage by subsystem, not exhaustive proof of every line or vendored operator. No live dependency vulnerability database, remote GitHub checks, complete coverage metric, legal distribution eligibility, clean-machine installation, actual model performance or host acceptance was established. Residual unknowns remain explicit rather than being counted as passed or fabricated defects.
