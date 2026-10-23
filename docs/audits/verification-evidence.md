# Verification evidence

## Scope and environment

Snapshot: [initial-state.json](evidence/initial-state.json) contains HEAD, original dirty status and SHA256 identities for 706 nonignored files. Root `/Users/mike.hall/Documents/vsCode/corridorKey2_fork`; branch `review/corridorkey-audit-20261023`; HEAD `7d808e6` (full identity in snapshot). Audit date 2026-10-24 local / 2026-10-23 UTC. Existing code/ledger changes were not authored by this audit.

macOS 27.0.1 ARM64; existing isolated Python 3.12.15; Torch 2.9.1, torchvision 0.24.1, NumPy 1.26.4, OpenCV 4.11.0.86, Qt Essentials 6.10.2. CUDA unavailable, no CUDA Torch build; configured weights/foundation directories empty; FFmpeg absent from PATH. Existing user `.venv` preserved. No packages or model assets installed/downloaded.

## Executed commands

Commands run from repository root. Audit-only `evidence/run_check.txt` records stdout/stderr, UTC time, elapsed wall time and exit code in matching JSON. Shell invocation was `python3 docs/audits/evidence/run_check.txt NAME COMMAND...`. Initial recorder had `.py` suffix; it was renamed to `.txt` to keep audit tooling outside the application's Python static scope.

| Check | Exact child command | Exit | Wall time | Evidence |
|---|---|---:|---:|---|
| unit-tests | `/private/tmp/corridorkey-compatible-venv/bin/python -m unittest discover -s tests -v` | 1 | 146.123s | [log](evidence/unit-tests.log) |
| socket-tests | `/private/tmp/corridorkey-compatible-venv/bin/python -m unittest discover -s tests -p *asset_download*.py -v` | 0 | 1.089s | [log](evidence/socket-tests.log) |
| code-health | `/private/tmp/corridorkey-compatible-venv/bin/python scripts/code_health.py --output-dir docs/audits/evidence/static` | 1 | 20.944s | [log](evidence/code-health.log) |
| code-health-final | `/private/tmp/corridorkey-compatible-venv/bin/python scripts/code_health.py --output-dir docs/audits/evidence/static-final` | 0 | 19.932s | [log](evidence/code-health-final.log) |
| environment | `/private/tmp/corridorkey-compatible-venv/bin/python -c import sys,platform,json,torch,importlib.metadata as m,shutil; print(json.dumps({"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"cuda_available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda,"packages":{x:m.version(x) for x in ["torch","torchvision","numpy","opencv-python","PySide6-Essentials","ruff","pyright"]},"ffmpeg":shutil.which("ffmpeg")},indent=2))` | 0 | 0.793s | [log](evidence/environment.log) |
| desktop-smoke | `/private/tmp/corridorkey-compatible-venv/bin/python -m corridorkey.desktop_smoke --source --output-parent docs/audits/evidence` | 0 | 1.145s | [log](evidence/desktop-smoke.log) |
| package-preflight | `/private/tmp/corridorkey-compatible-venv/bin/python -m corridorkey.desktop_bundle check` | 2 | 0.097s | [log](evidence/package-preflight.log) |
| decode-benchmark | `/private/tmp/corridorkey-compatible-venv/bin/python -m corridorkey.benchmark --manifest benchmarks/suites/decode-reuse.json --output docs/audits/evidence/decode-benchmark-report.json` | 0 | 4.427s | [log](evidence/decode-benchmark.log) |
| legacy-load-proof | `/private/tmp/corridorkey-compatible-venv/bin/python -c import tempfile,pathlib,torch; from corridorkey.runtime import load_runtime_checkpoint
class Payload:
 def __reduce__(self): return (print,("AUDIT_PROOF: checkpoint-controlled callable executed",))
with tempfile.TemporaryDirectory() as root:
 p=pathlib.Path(root)/"local-proof.pt"; torch.save({"payload":Payload()},p); print(load_runtime_checkpoint(p))` | 0 | 1.068s | [log](evidence/legacy-load-proof.log) |

## Interpretation

- Full suite: **1,533 tests in 144.673s; 15 errors**, all localhost HTTP fixture binds denied by sandbox (`PermissionError: [Errno 1] Operation not permitted`). No assertion failures or skipped cases reported. The narrow `*asset_download*.py` rerun was explicitly escalated: **21 tests in 0.991s, OK**. This covers every failing test plus related cases. Combined evidence covers all 1,533 discovered tests, but there was no single wholly green unsandboxed full-suite run. Expected injected exceptions and Qt offscreen warnings appear in logs and are not test failures.
- Initial health run counted the newly created audit recorder as an application Python file, producing three Ruff findings in that recorder. It is retained verbatim in `static/`; not a repository regression. After renaming only that audit artifact, final original-source scope is **422 files**, syntax/imports/dependency consistency PASS, **zero new diagnostic fingerprints** against the existing unchanged baseline. Raw checkers exit 1: **512 Ruff diagnostics; 684 Pyright errors +17 warnings**. Gate exit 0 does **not** mean static-clean. Final raw reports: [health](evidence/static-final/health.json), [Ruff](evidence/static-final/ruff.json), [Pyright](evidence/static-final/pyright.json).
- Source desktop smoke: backend subprocess, real worker handshake/status and clean shutdown PASS. No inference submitted. Offscreen Qt widget tests execute within the suite; neither is visual desktop acceptance.
- Packaging preflight: exit 2 expected; requires Windows x86_64 and PyInstaller. Correct refusal, not a successful bundle build.
- Decoder benchmark: **9/9 results exact-equal**, three measured pairs after warmup per schedule/case, 12 frames at64×64, cache262,144 bytes. Generated uint16 PNG, signed HDR EXR and premultiplied BT.2020 EXR. Actual report has settings/provenance/timing/counters. Candidate/reference median ratios vary approximately0.749–1.464; this tiny concurrent-host run does not establish a production speedup. Only Quick-Key current/anchor schedules record useful decode reuse; overlapping window baseline already reuses frames. Neural inference remains unmeasured. Artifacts were created by the existing runner in its configured `benchmark-results/decode-reuse-*` fresh directory, not user media.
- Default CPU benchmark was intentionally not run: it repeats the entire unit suite three times and adds no neural throughput measurement. Existing decode-reuse benchmark is the meaningful CPU-compatible workload used here.
- Harmless legacy-load proof prints a checkpoint-selected sentinel through the real loader before format validation. No shell, network or durable side effects.
- `git diff --check` exit0; [log](evidence/diff-check.log) empty.

## Local security scan

Exact command: `semgrep scan --metrics=off --disable-version-check --config docs/audits/evidence/security-rules.yml --json --output docs/audits/evidence/semgrep.json corridorkey scripts third_party infer.py run_interactive.py bundle_entry.py`.

Exit0; 4 local rules,203 Python targets,5 matches; [raw result](evidence/semgrep.json). Scanner summary reports git-filtered scope; this is not a whole dependency/native-code vulnerability scan. Initial sandbox tool startup failed with `ca-certs: empty trust anchors`; narrow escalation succeeded without disabling certificate validation. Duration not recorded reliably. Results manually traced; no claim of zero vulnerabilities.

## Tool use and exclusions

OMX0.21.33 available; three native OMX-role review lanes plus coordinator QA lane fit four concurrency slots. Native lanes avoid tmux/team state churn and satisfy isolated review workstreams; no additional supervisor launched. Separate final reviewer assesses the combined findings. Ponytail audit/simplicity and Superpowers verification-before-completion practices used under OMX authority. Serena/Context7 tools are exposed; exact-source navigation sufficed and no uncertain external API implementation was proposed, so neither was invoked merely for tool coverage. No browser frontend: Playwright and Chrome DevTools not applicable. `gh` unavailable on PATH; remote issues/CI state not inspected. Local history/remotes recorded; no Git network mutation. Durable audit checkpoint and raw evidence stay under this directory, not historical development ledgers.

## NOT VERIFIED

Pretrained CUDA inference, numerical parity, full-resolution VRAM/RSS peaks, professional matte/foreground quality, real-video FFmpeg encode/decode, VFR end-to-end behavior, optional segmentation model execution, Windows/MSVC/bundle/clean-machine/upgrade behavior, physical storage/power-loss stress, actual Resolve viewer/export/retime/proxy/colour behavior, vulnerability database freshness and final distribution licence approval. Required hardware/assets/host/tool prerequisites are not replaced by fixtures. No measured production performance improvement is asserted.
