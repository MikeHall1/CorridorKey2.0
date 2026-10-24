# Batch-local despill reuse

The renderer reuses an ordinary configured despill candidate only when the same
current render batch needs it in the same working domain. Neural EXR cutout and
checker share scene-linear despill. Neural and source-draft video outputs share a
linear or display-coded candidate when their settings select the same input.
A linear cutout plus legacy checker still needs two different candidates. Neural
legacy cutout no longer computes an unused scene-linear candidate. There is no
cross-frame cache, service, change to model work, colour transform or precision.

Global despill strength zero skips composite-region retrieval, packing and transfer
at the runtime batch seam. Correction documents and their assets still undergo
normal preflight; malformed assets do not become accepted when despill is off.
An explicitly supplied renderer mask still undergoes native shape/dtype/device/
finite/range validation even when despill is off. Existing metadata float parsing
(including alpha/raw-FG/empty-output paths) remains in place.

Removing the unused legacy-video linear candidate also removes a confirmed false
rejection: finite decoded HDR (including inverse brightness adjustment) can overflow
the *unused* linear channel sum while the chosen clamped display input is valid.
The legacy output now matches the independent chosen-domain oracle; explicitly
requested linear processing still rejects that overflowing used channel sum. No
model/source/native-mask guard is removed. This boundary has no old output to
compare and is tested separately, not used as a performance/parity case.

Candidate reuse is within the function call only. A candidate can remain alive
while an exported array owns its storage; it is not retained in a persistent cache.
Call counts and tensor `nbytes`/weak-reference lifetime observations describe work
and storage references, **not** RSS, CUDA allocator peaks or a memory cap.

## Validation and fixed experiment policy

```sh
QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 \
python -m unittest \
  tests.test_render_despill_reuse tests.test_composite_region_rendering -v
```

The new call-count tests first reproduce redundant work on the previous runtime.
Combined outputs must equal individually rendered outputs byte-for-byte across
legacy/linear/mixed domains, alpha association, regions and global disablement.
Existing global fixture tests cover actual decode/EXR writes, low-memory/spatial/
Quick Key/neural fallback/tracked regions, source offsets, hints and model work.

A separate ignored, fixed CPU component experiment is staged under
`benchmark-results/product-next/render-despill-optimization/`. Its policy was
recorded **before implementation or measurements**, SHA-256
`606edf9ee2b2f7c82f490edade4e00a05d85acca5e309f59ffa529289a7778dd`.
It requires a separate coordinator measurement release after source/review freeze.
Inputs are synthetic post-model float32 CPU tensors. Neural cases consume batch
2 × 128 × 192; source-draft helpers consume one preallocated 128 × 192 frame per
call. Per-case actual shapes, helper batch/frame counts and calls are recorded; a
separate pre-measurement metadata amendment preserves the original policy and
workloads/repetitions/thresholds. Sixteen
predeclared cases include affected paths, mixed domains and unchanged controls.
Only the isolated measurement process requests one Torch intra-op thread; actual
intra-/inter-op counts are recorded without changing the user's environment.

Each case uses two warmups/version then five measured pairs, alternating AB/BA,
seven calls/sample, `perf_counter_ns`, with inputs allocated outside the clock.
Output creation and retirement remain inside the timed renderer loop. A median
paired after/before ratio ≤0.95, at least four of five ratios below one, and ratio
MAD/median ≤0.10 permits an **observed CPU component speedup** description.
The symmetric ≥1.05/four-above-one/noise gate identifies an observed regression.
Everything else is inconclusive; every case is reported without policy tuning,
selective reruns or aggregating away controls/regressions. Numerical acceptance is
exact output bytes/dtype/shape, not a tolerance. Untimed numerical and call/storage/
lifetime instrumentation runs are separate from the timing experiment.

The source maps, baseline runtime, executed harness SHA, raw times, output hashes,
versions/settings and outcomes remain in local ignored evidence. Warm imported
synthetic postprocessing does not measure cold startup, model inference, end-to-end
FPS, representative footage, CUDA performance or VRAM. Actual pretrained/GPU,
FFmpeg, Windows and Resolve acceptance remains **NOT VERIFIED**. This guide
contains no measured speed claim until the separately released experiment runs.
