# Run checkpoint — milestone26, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, revalidated milestone25 source/ranges and established a fresh476-test/210-file baseline before runtime edits. Milestone25 is archived at `checkpoints/milestone25.md` and `.ralph/checkpoints/milestone25.json`. The complete development brief remains the goal; native bundle/GPU/quality/Resolve acceptance is unproven. No competing supervisor was launched.

## Completed cycles

1. Identified/classified frozen child startup incompatibility: sys.executable points to the bundled program, so interpreter -m/-c arguments do not work as assumed. Added narrow application dispatcher and shared source/frozen command factory for worker, backend probe, benchmark and cold batch. Source unittest suites explicitly refuse in frozen bundles. Actual subprocess module/help/backend checks and479-test gate pass.
2. Added Windows x86_64/Python3.12/PyInstaller6.22.3 one-directory console bundle recipe, private output directories, existing source gate, notices/text/code enumeration, dependency metadata, external model/media policy and retained build/failure reports. Recipe generation compiles; failure injection preserves user files.482 tests/full gate pass. Primary OpenFX documentation investigated/cited in resolve.md; Resolve-specific host capabilities remain unverified.
3. Added explicit trusted bundle-executable smoke command and development --source mode. Actual backend launch, worker ready/status and clean shutdown submit zero jobs; failures retain reports without replacing existing files.484 tests/full gate pass. Source smoke is not native bundle evidence.
4. Reproduced backend failure after the parent imports the source package then changes cwd: child cannot resolve the module. Fixed backend child cwd to actual code root while preserving caller cwd. Before/after JSON and real regression prove backend changes from NOT VERIFIED to observed cuda_available=false.485 tests/full gate pass.
5. Added CUDA-enabled Torch-build admission for Windows/NVIDIA and observed commit/dirty-working-tree provenance in build reports. CPU-only/unverified Torch builds reject; device availability/inference is separate. Isolated failure fixture's provenance mock after its global subprocess mock incorrectly consumed git calls. Final486 tests/full no-new-findings gate pass. CPU repeated benchmark recorded below.

## Verification and evidence

- Fresh baseline `benchmark-results/m26-start-verification.txt`, archived `m26-start-baseline/`.
- Cycle archives `m26-cycle1/` through `m26-cycle4/`; final `m26-final/`, log `m26-final-verification.txt`. `.ralph/runs/manual-check/verification.json` success true, all4 commands return0.
- **486 tests PASS/no skips;218 Python files,0 syntax errors.** Runtime imports, declared requirements/VCS identity, installed dependency consistency, compilation and whitespace pass.
- Ruff520 findings; Pyright696 errors/17 warnings (713 diagnostics), **0 new fingerprints** against unchanged baseline. Repository is **not static-clean**. New application/bundle/smoke modules/tests/entry have focused Ruff/Pyright0 findings. Existing runtime/benchmark static findings remain visible.
- Existing dataclass instance/class and heterogeneous-set serializer contracts inspected; existing tests retained. Pylance editor diagnostics NOT VERIFIED. No tools/checkers were weakened, baseline refreshed, packages installed or user environment changed. requirements-build.txt is an optional separate builder pin, not a runtime upgrade.
- `m26-build-preflight.json`: actual check exits2/ready=false for wrong target OS and missing PyInstaller. This is expected refusal, **not a built/validated application**.
- `m26-source-smoke.txt` references actual `desktop-smoke-s9pb0cyn/report.json`: passed backend/worker source launch, active=null/pending=[], clean_shutdown=true, cleanup_error=null. CUDA=false/cuda_build=null. No model job submitted, no pretrained inference. `m26-backend-cwd-before.json` and `after.json` preserve the regression evidence.
- `m26-assets.json`: all5 configured model/foundation asset paths absent. No weights/media downloaded or uploaded. Build instructions and smoke commands: [packaging.md](packaging.md).
- Actual `python -m corridorkey.benchmark --output benchmark-results/m26-cpu-benchmark.json` completed successfully:3 repetitions, median32.13611712504644s, min31.914010790991597s, max32.722464042017236s. All486-test repetitions pass. These are CPU correctness-suite elapsed times, not inference FPS, GPU speedup or quality evidence; no controlled baseline/candidate comparison is claimed.

## SDK investigation

[resolve.md](resolve.md) cites official OpenFX render/concurrency/sequence/window and temporal-image/stride/component contracts. This supports design requirements, not a Resolve/Fusion compatibility claim. The fetched Blackmagic support page supplied dynamic placeholders rather than host-specific SDK capability docs. No C++ adapter, SDK native build, Resolve host render, GPU sharing or arbitrary output-clip capability has been validated. No SDK/media upload or distribution occurred.

## Issues and limits

HEALTH-PACKAGE/TYPE/SMOKE/TEST146–151 were classified/evidenced before fixes; failed focused logs are retained. Existing static backlog and **HEALTH-FORMS-126 native Qt crash mechanism remain open**. Passing desktop suites do not establish the underlying native crash cause or target interactive acceptance.

Build recipe behavior and smoke failure/lifetime transport are locally verified; actual PyInstaller execution is **NOT VERIFIED** because this Mac is not the target Windows platform and PyInstaller is absent. Native DLL/Qt plugins/dynamic C-RADIO/MoGe imports, source/asset compatibility, model rendering, packaged benchmark/cancel/recovery, Windows process containment and clean-machine application behavior remain to test. Built-unvalidated status is deliberately distinct from acceptance. Console streams are currently required; a polished launcher/installer/signing remain unfinished.

The recipe retains existing notices/source data and package metadata; this is not proof of complete dependency licence distribution obligations. Model/FFmpeg assets remain external. Deliberate download/deeper compatibility UX remains incomplete. Observed source identity is not an atomic snapshot or a bit-reproducible build. The source gate is the existing verify command, not another autonomous supervisor.

## Next action and remaining full scope

Inspect official target SDK/header contracts and implement a thin cached-frame C++ OFX path connected to actual shared-engine companion outputs. Validate bounds/signed strides/alpha/time mapping/instance isolation and then Windows Resolve behavior; do not invent host background access or GPU sharing. Execute the packaging recipe/full acceptance on a prepared Windows CUDA environment when available.

Continue total resource/disk budgets, retention deletion/portable backups, advanced corrections, high-bit video/managed colour, direct-seek/Stage2/writer recovery, controlled alternating/stress/held-out evaluation and actual GPU/quality/Windows/host acceptance. Supported GPU/assets/permitted representative footage, target packaging and Resolve SDK/host remain missing or unverified prerequisites. Independent work remains actionable; the goal is neither complete nor blocked.

Machine checkpoint `.ralph/checkpoint.json`; task ledger `.ralph/plan.json`; issue ledger `.ralph/issues.json`.
