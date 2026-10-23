# Security, tests and release readiness

Scope: current dirty checkout, not merely HEAD. Coordinator lane D; application and historical ledgers unchanged. See [verification evidence](verification-evidence.md) for exact commands and [findings register](findings-register.md) for authoritative severity and remediation.

## Trust boundaries and confirmed concerns

- **SEC-001:** Runtime accepts legacy pickle checkpoints automatically. `runtime.py:165–179`, `runtime_paths.py:28–32,44–47`, `foundations.py:193–215` and conversion utility permit unrestricted deserialization. A temporary, locally generated checkpoint invoked a harmless `print` callable before model format validation. This is confirmed code execution when a user selects an untrusted legacy asset, not a remotely reachable service exploit. Prefer restricted weights loading; explicitly isolate/acknowledge trusted legacy conversion when compatibility requires it. Existing nonexecuting asset audits must not be mistaken for runtime trust approval.
- C-RADIO executes local snapshot Python (`foundations.py:46–81`) or trusted remote-code cache (`:97–99`). That is an intentional integration requirement, not independently a vulnerability. Publisher provenance and user trust remain necessary even for safetensors. Hashing arbitrary user code identifies it but does not establish its safety.
- **SEC-002:** README's legacy download entrypoint uses an unpinned default CorridorKey revision (`scripts/download_weights.py:52–55,90,97`) and independently downloads the two paired assets. It does not provide the stronger immutable plan and publication guarantees of `asset_download.py`. This is a reproducibility and compatibility risk; no compromised upstream artifact was observed.

Four local Semgrep rules reviewed explicit unsafe Torch loads, shell subprocesses, pickle calls and trusted model-code execution. Five matches: four unrestricted Torch callsites and one model-code callsite. No `shell=True` match. This is a bounded pattern scan, **not** a comprehensive vulnerability assessment or dependency CVE audit. Scan required narrow macOS trust-anchor escalation; metrics and version checks were disabled, rules were local, no uploads or external rules used.

## Positive controls inspected

Worker messages use bounded framed JSON rather than pickle (`worker_protocol.py`, `local_json.py`). Subprocess commands use argument lists and owned lifecycle/timeout handling. Output-report admission protects source, manifest, model and export identities (`output_paths.py`); it does not resolve ARCH-001's cross-job output write gap. Download plans enforce approved origins, immutable publisher identities, bounded transfer and explicit collection. Portable projects/recovery use schema, digest and path validation rather than extracting arbitrary archive trees. Broad regression coverage exercises aliases, interruption, disk errors, late publication, subprocess exit and stale UI results. Native OFX tests compile real C++ harnesses, including sanitizer tests; these are not actual Resolve acceptance.

## Testing assessment

The repository has 219 Python test/support files in the initial nonignored inventory, with 1,533 discovered test cases. Coverage spans scheduling, recovery, caches, JSON contracts, hints/corrections, real native EXR I/O, Qt offscreen widgets, worker subprocesses, benchmark admission and native ABI harnesses. Many runtime tests deliberately replace neural models with deterministic CPU doubles. That is useful integration proof but cannot establish trained-model quality or CUDA numerical behavior. Test counts are not branch coverage; coverage instrumentation was not run.

**QA-001:** No tracked CI configuration exists in `.github/`, GitLab or Jenkins paths. Tests and health checks are manual commands. Local success does not create a merge gate or Windows execution evidence. Add small Linux/Windows CPU/static jobs first, preserving optional external GPU/Resolve gates.

**QA-002:** The static regression gate passes against its historical baseline, but the checkout is not static-clean: 512 Ruff diagnostics and 684 Pyright errors plus 17 warnings across 422 Python files. Do not report these as 1,213 confirmed bugs. Prioritize runtime-relevant undefined names/contracts; classify false positives; retire debt without broad suppressions or baseline laundering. Raw diagnostics are retained.

## Release, installation and licences

**REL-001:** CUDA/model/representative quality, real FFmpeg encode/decode, clean Windows bundle and Resolve host acceptance are NOT VERIFIED. Source worker smoke passes; package preflight correctly refuses macOS ARM and missing PyInstaller. Requirements separate optional desktop/build tools and pin developer tools; runtime requirements retain broad lower bounds and an exact utils3d VCS commit. Torch is installed separately per README. The preserved user `.venv` is not the declared compatible environment; checks used existing isolated Python 3.12/NumPy 1.26 instead.

LICENSE and THIRD_PARTY_NOTICES are present, including upstream and OpenFX provenance. The top-level licence is custom, not a generic permissive licence; actual release distribution needs its own licence review and final bundle inventory. This audit makes no legal eligibility determination. No release, redistribution, installer execution, dependency update or download was performed.

## Gaps and recommendation

No live vulnerability database query, remote GitHub CI/issues inspection, clean-machine install/upgrade, signed distribution, GPU test, reference-footage quality verdict or native Windows/Resolve run was performed. These are explicit limits, not passing checks. Use controlled testing with trusted assets and disposable copies until correctness findings and release gates are addressed.
