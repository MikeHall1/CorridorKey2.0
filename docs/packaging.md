# Windows desktop bundle recipe — not a validated release

The implemented recipe targets Windows x86_64 with Python3.12 and a deliberately selected CUDA-capable Torch2.9.1/torchvision0.24.1 environment matching the project requirements. No build/download/install is performed automatically. Current Mac source validation is not Windows binary or GPU validation. [PyInstaller requires building on the target OS](https://pyinstaller.org/en/latest/).

In a separate prepared Windows build environment, install unchanged runtime, desktop and development requirements plus `requirements-build.txt` (PyInstaller6.22.3). Select the appropriate CUDA Torch wheels deliberately using the official PyTorch installation instructions; this recipe does not choose a CUDA/driver combination. Then:

```sh
python -m corridorkey.desktop_bundle check
python -m corridorkey.desktop_bundle build --output-parent C:/local-builds
```

The output parent must already exist. Preflight checks platform, Python, required package presence, builder/Qt pins and CUDA-enabled Torch build metadata on Windows. CPU-only/unverified Torch builds reject. GPU availability is recorded separately; an available CUDA build is not proof of usable hardware or model correctness. It is not a dependency solver or inference verifier. Build runs the existing complete source gate first, then generates a spec in a new private `desktop-build-*` directory, invokes PyInstaller and retains `build-report.json` with actual versions, command, source data hashes, existing commit/dirty-working-tree identity and success/failure. This records observed identity, not an atomic source snapshot or a bit-reproducible build. Existing build directories are not replaced. The source gate records runtime requirements/VCS consistency and static regressions as usual. No source tests are removed to make a build pass.

The generated one-directory console-capable bundle includes the shared engine, dynamic application modules, vendored MoGe code/configuration, package metadata and existing notices. [PyInstaller spec files define Analysis/PYZ/EXE/COLLECT and bundled data](https://pyinstaller.org/en/stable/spec-files.html). Console mode is deliberate for worker protocol/log streams; a polished no-console launcher remains future work. No weights, source footage, FFmpeg executable or user outputs are selected by the recipe's data enumeration. Local assets remain explicitly selected using the existing desktop assets panel; no hidden download or CPU/MPS inference fallback is added.

The generated Analysis selects source-only collection for `corridorkey` and `moge` (`module_collection_mode` uses `py`), rather than keeping a separate bytecode copy in PYZ. Recovery implementation identity hashes Python files under `corridorkey/`, `third_party/` and the collected root-level `moge/`, so either possible MoGe import location participates. This uses the documented [PyInstaller 6.22.3 source collection mode](https://pyinstaller.org/en/v6.22.3/hooks.html#hook-global-variables); it does not establish that the Windows bundle has run successfully.

`bundle_entry.py` calls freeze_support then the narrow application dispatcher. Source children retain `python -m`; frozen children call the application with an explicit supported module selector. Worker, backend probe, benchmark and cold batch launches share this contract. [A frozen sys.executable identifies the application bootloader](https://pyinstaller.org/en/v6.22.3/runtime-information.html), so treating it as a generic interpreter is incorrect. Source unittest benchmarks explicitly report unavailable in frozen applications; inference benchmarks retain the shared engine path.

Actual native build, required dynamic model imports, Qt plugins/DLLs, packaged benchmark/render/cancel/recovery, Windows cleanup, CUDA inference, signing, installer and clean-machine behavior remain NOT VERIFIED. A successful builder result is labelled built-unvalidated. Preserve and review PyInstaller warnings and dependency licence files before distribution; source notices/metadata alone do not prove all distribution obligations are met. This run makes no distribution authorisation or legal-compliance claim and publishes nothing.

A trusted built executable can be checked without submitting model work:

```sh
python -m corridorkey.desktop_smoke --executable C:/local-builds/desktop-build-ID/dist/CorridorKey/CorridorKey.exe --output-parent C:/local-results
```

The smoke tool creates a new private report directory, launches backend availability and an actual worker ready/status/shutdown exchange, and retains failures. `--source` instead checks the source dispatcher and must never be presented as native-bundle evidence. CUDA availability in this report is not pretrained inference or quality approval; manual packaged GUI and the full target acceptance suite remain required.
