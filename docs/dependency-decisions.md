# Dependency and obsolete-code decisions — 2026-10-24

**Decision: retain the current dependency set; no dependencies or dead modules
were removed.** This records the completed bounded PERF-DEPS/PERF-CLEAN inventory,
not a new broad audit, upgrade proposal or product-acceptance claim. Preserve
full-quality defaults, public/persisted interfaces and upstream notices.

## Evidence and environment

Declarations: [core](../requirements.txt), [desktop](../requirements-desktop.txt),
[build](../requirements-build.txt), [development](../requirements-dev.txt), and
the separate Torch/torchvision installation in [README](../README.md) and
[CPU CI](../.github/workflows/cpu-checks.yml) (install step, lines 40–45 at this
record). CI configuration is not proof of a successful hosted run.

Versions below were observed on 2026-10-24 by read-only
`importlib.metadata.distribution(...).version`, `requires` (`Requires-Dist`) and
`read_text('direct_url.json')` in the existing
`/private/tmp/corridorkey-compatible-venv` (Python 3.12.15). They are **not latest
version recommendations**. The [recorded CPU freeze](environments/cpu-macos-py312.txt)
and [environment limits](code-health.md#reproducible-local-environment) provide
the existing baseline; that freeze predates the separately observed Qt install.
No packages/environments were changed. The user `.venv` remains untouched:
its observed Python 3.13/NumPy 2 stack does not satisfy `numpy>=1.26,<2`.

## Inventory and decisions

| Scope / candidates | Declaration / observed version | Consumers and decision |
|---|---|---|
| Core: Torch / torchvision | Separately selected `2.9.1` / `0.24.1`; both observed | **Retain.** Shared model/runtime tensor execution; installed torchvision requires Torch `==2.9.1`. `corridorkey/model.py`, `runtime.py`; timm also declares both. Do not replace the shared engine or silently select a new backend. |
| Core: NumPy / OpenCV / safetensors | `>=1.26,<2` / `>=4.11` / `>=0.6`; observed `1.26.4` / `4.11.0.86` / `0.8.0` | **Retain.** Actual colour, decode/preprocessing, correction, checkpoint and durable-cache consumers: `video_io.py`, `color_utils.py`, `runtime.py`, `durable_stage2.py`. |
| Dynamic foundation closure: transformers / timm / einops | `>=4.57,<5` / `>=1.0` / `>=0.8`; observed `4.57.6` / `1.0.30` / `0.8.2` | **Retain; defer optionalization/removal.** `foundations.py:46–110` loads selected C-RADIO snapshot code dynamically or lazily imports AutoModel. Absence of direct project timm/einops imports does not disprove snapshot use. Local `vfms/cradio` contained only its placeholder, not executable snapshot code/weights; the closure remains unverified. |
| MoGe: SciPy / utils3d | `>=1.16` / Git commit `83bfa5bda008803f6521ce9547f94023b8afeddd`; observed `1.17.1` / `1.7` | **Retain.** `foundations.py:29–31` imports MoGe v2; `third_party/MoGe/moge/model/v2.py:14–19` imports utils3d and geometry_torch; `utils/geometry_torch.py:10–13` imports geometry_numpy; `geometry_numpy.py:7–9` imports `scipy.signal` and utils3d. Encoder-only use does not bypass these imports. Installed utils3d `direct_url.json` matched the declared repository and full commit. |
| Core/support: huggingface-hub / Pillow / tqdm | `>=0.36` / `>=11` / `>=4.67`; observed `0.36.2` / `13.0.0` / `4.70.1` | **Retain.** Hub is imported by MoGe and download tooling; installed transformers/timm require hub, torchvision requires Pillow, and transformers/hub require tqdm. Vendored MoGe also has image/progress consumers. Removing a direct requirement would not remove the transitive dependency. |
| Desktop: PySide6-Essentials / shiboken6 | Desktop pin `6.10.2`; both observed | **Optional; retain separate manifest.** Qt desktop/canvas consumers; installed Essentials requires shiboken6 `==6.10.2`. Do not make headless model execution require Qt. |
| Build: PyInstaller; CMake/C++17/OpenFX | Build pin `6.22.3` (not installed in inspected compatible environment); CMake `>=3.20`, C++17 | **Optional; retain.** `desktop_bundle.py:60–75` emits the PyInstaller recipe; `native/ofx/CMakeLists.txt` uses vendored OpenFX headers. Builder/SDK availability is not native Windows/Resolve acceptance. |
| Development: Ruff / Pyright | Pins/observations `0.17.1` / `1.1.414` | **Development-only; retain.** `scripts/code_health.py` and CPU CI invoke the existing regression gate. Keep diagnostics and known debt visible; do not suppress checkers to justify cleanup. |
| Optional providers: SAM2 / Hydra; xformers | No new core declaration or installation | **Optional/lazy; retain capability boundaries.** `subject_guidance.py:375–428` dynamically imports the deliberately selected local provider after asset admission. Vendored DINOv2 `layers/attention.py:22–33` guards xformers imports; block/SwiGLU layers also have optional paths. No requirement to install them follows from this inventory. See [provider limitations](subject-guidance-provider.md). |
| Transitively installed support packages | Existing CPU freeze; installed `Requires-Dist` edges | **Retain/defer; no individual uninstall.** transformers: tokenizers/packaging/YAML/regex/requests/filelock/tqdm/safetensors/hub; hub: fsspec/hf-xet/typing-extensions and shared support; Torch: sympy/networkx/jinja2 and shared support; Pyright: nodeenv/typing-extensions. These are environment-specific edges, not a complete target-platform lock. |
| utils3d graphics transitives: moderngl / glcontext | Observed `5.13.0` / `3.1.0`; utils3d declares moderngl, which declares `glcontext>=3.0.0` | **Retain/defer.** Installed utils3d uses lazy backend imports, and its `numpy/rasterization.py` imports moderngl. Laziness does not erase declared installation requirements. No upstream metadata rewrite or manual transitive removal is justified. |

## Public, dynamic, persisted and packaged cleanup boundaries

- **Retain the public `CorridorKeyV2` alias:** `corridorkey/__init__.py:1–15`
  exposes it lazily. A different implementation class name is not obsolescence.
- **Retain MoGe v1:** `third_party/MoGe/moge/model/__init__.py:9–18` explicitly
  dispatches both versions by dynamic import. No accepted removal/migration is
  established by the application's current v2 loader.
- **Retain `scripts/ralph_loop.py`:** `desktop_bundle.py:86–88` still invokes its
  `verify` action. This does not authorize launching another supervisor.
- **Preserve saved settings:** `session.py:32–65`,
  `worker_protocol.py:60–95`, and `desktop_state.py` carry complete JobRequest
  configuration through worker/project codecs. Queue/portable/recovery records
  are additional consumers, not permission to rename or discard fields.
- **Preserve packaged/durable identity:** `desktop_bundle.py:49–72` enumerates
  notices and MoGe source, collects CorridorKey/MoGe submodules and package
  metadata; `recovery_identity.py:41–49` hashes implementation Python trees.
  Packaging inclusion alone does not prove useful execution, but these contracts
  must be checked before deletion. No module met the required removal proof.
- **Only accepted production work removal:** the unused review-window stack is
  skipped by the production reader factory (`runtime.py:1031–1032`), while
  concrete reader constructors keep `include_review=True` by default
  (`video_io.py:463–501`, `552–579`). Public triples and injected reader factories
  remain compatible. [Reader efficiency](reader-efficiency.md) records focused
  parity/allocation evidence and **inconclusive timing**. Per-frame review
  preprocessing/cache work remains; no dead module or dependency was removed.

## Uncertainty and stop conditions

No current upstream maintenance, advisory, vulnerability, licence-compliance or
distribution certification was performed or inferred. Preserve LICENSE,
THIRD_PARTY_NOTICES.md and vendored licences. Existing package metadata is
consumer evidence, not a security review.

Revisit a removal/optionalization only after exact selected C-RADIO code closure,
dynamic/public/persisted/packaged/platform consumers and a compatible replacement
are demonstrated, followed by focused regression and target validation. Actual
foundation loading/inference, GPU/footage quality, clean Windows packaging,
optional provider installation/execution, and Resolve SDK/host acceptance remain
**NOT VERIFIED**. Local imports or declarations do not pass those criteria.

This persistence change is documentation-only: no application, test, manifest,
environment or ledger changes. The new-file diff was inspected and a read-only
Python check verified trailing whitespace, final newline and local Markdown link
targets. No tests or performance experiments were rerun for the document.
