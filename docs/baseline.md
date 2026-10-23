# Starting baseline — 2026-10-21

The supplied workspace had an unborn `main` branch, no files, no remotes, and no user edits. Fetched `https://github.com/MikeHall1/CorridorKey2.0` into a temporary checkout, then imported its HEAD into local branch `implementation/baseline-reliability`.

Actual starting commit: `485f5e9ab759ead6138cc990776184b59d6ea971` (README contact update). It matches the review revision in the brief. Upstream reference is `https://github.com/nikopueringer/CorridorKey2.0`; no claim that upstream HEAD was compared. No prior AGENTS.md or ancestor instructions were present. No changes have been pushed.

Host: Darwin arm64, Python 3.13.14. CUDA/NVIDIA GPU, `nvidia-smi`, FFmpeg, Torch, NumPy, OpenCV, weights, reference footage and a prior benchmark baseline were absent. Sandboxed sysctl hardware details may be unavailable; the report records this explicitly. Python 3.13 is incompatible with the current NumPy <2 runtime constraint; use a Python 3.11/3.12 CUDA environment for full-runtime validation. Dependency declarations are unchanged.

Existing tests: only `scripts/smoke_test_runtime.py`, requiring assets and PyTorch; optional model loading requires CUDA. No CPU regression suite existed. New tests exercise dependency-free core behavior and optional CPU I/O adapters; they are not model inference.

For adapter validation, created an isolated temporary venv at `/private/tmp/corridorkey-cpu-venv` with Python 3.13, CPU PyTorch 2.9.1, NumPy 2.2.6, OpenCV 4.11.0.86 and einops 0.8.1. NumPy 2 is a **test-only deviation** from the runtime NumPy <2 constraint, needed for Python 3.13 wheels. Requirements were not changed; repeat on the declared runtime stack before release. No GPU backend or model assets were installed.

Inspected `LICENSE` and `THIRD_PARTY_NOTICES.md`: custom Corridor Key Licence, CC BY-NC-SA plus additional terms, attribution and integration/distribution conditions. C-RADIO uses NVIDIA Open Model License; MoGe MIT with DINOv2 Apache notices; RVM code Apache 2 and weights CC BY 4. Preserve these files. Packaging/integration needs a separate review of the exact applicable terms before distribution; this milestone does not package or distribute.

Confirmed defects against source: both writer workers abandon bounded queues on exceptions; shared basename-derived temp paths; lexical/case-sensitive discovery; model_scale accepted but output renderer rejects it; first cleanup failure skips remaining resources and replaces the processing failure. Also found RGBA hints converted to colour luma, partial-video hint directories ignoring source offset, frame-directory start/linear adjustments silently ignored, and high-bit PNG decoding reduced to eight bits.

Initial GPU throughput, latency, memory and visual quality: **unmeasured**. The inherited VRAM table lacks hardware provenance here and is a heuristic, not a newly validated guarantee.
