# Explicit sample calibration

Open **Calibrate performance…** beside Resources. Opening the dialog performs no
filesystem inventory, model loading or processing. Choose a local image sequence,
source In and 1–120 frames, separate existing scratch/report parents, and explicit
profile JSON. Capture metadata first; inspect the workload, missing assets and
required read bytes. Set a deliberate verification read budget (the dialog starts
at zero), then Verify. Start is a separate action after verification.

Calibration temporarily retires the editor's shared worker and progressive
preview operations. It never runs a second model owner alongside them. Cancel
waits for worker retirement; an unverified cleanup blocks another processing
owner and application close. Use New worker explicitly afterward. Render settings,
queue entries and previous exports are unchanged. Profile edits invalidate the
verified plan. No recommendations apply themselves to jobs.

Each profile has exactly `id`, `purpose` (`preview` or `final`) and `overrides`.
Supported overrides cover processing scale, temporal window/stride, Stage 2 batch,
explicit low-memory chunk/profile settings, foundation cache and Quick Key policy.
For example:

```json
[{"id":"baseline","purpose":"final","overrides":{}},
 {"id":"draft","purpose":"preview","overrides":{"model_scale":0.5,"hann_chunk":4,"hann_stride":2,"stage2_batch":2,"low_vram":true,"foundation_cache_mb":0,"quick_key":false}}]
```

The original requested backend, precision, colour, writer and output kinds remain
in effect. Final-eligible profiles require scale 1 and Quick Key off; that label
does not establish visual quality. Automatic VRAM configuration and synchronized
stage profiling must be explicitly disabled. Video, hints and correction documents
are refused until their complete timing/conditioning identities are supported;
they are never silently removed. Sequence timing uses the declared constant FPS.

Up to eight profiles and 3–7 repetitions are supported. A profile/repetition
performs one cold measured job, one unmeasured full-sample warmup and one warm
measured job, alternating profile order between repetitions. The declared job and
frame budgets include warmups. Cold means a fresh worker process, including its
startup and verified shutdown in the trial wall time. It does not mean a cold OS
filesystem cache. Each warm trial uses a fresh worker, one successful warmup and
one same-generation measured job, ending its timer before verified shutdown.
Temporal state restarts for the sample. All jobs use the existing shared engine.

Source sequence, model assets and CorridorKey package Python files are hashed
within the explicit read budget; membership and filesystem pins are rechecked
around trials. Python/platform and installed package versions are recorded.
Actual records must match requested/effective settings, source In/count/FPS/native
size, writer, output kinds and owned paths. CUDA recommendation eligibility also
requires actual device/neural-work observations; an explicit fixture flag vetoes
it. Sparse Quick Key work counters describe evaluated anchors rather than all
written frames. Missing or contradictory observations cannot become measurements.
A common worker startup or model-load prerequisite failure stops the matrix after
verified retirement rather than repeatedly retrying unchanged missing hardware.

Reports retain every raw trial record, failures and skipped-configuration reasons.
Throughput and latency scopes are explicit. Missing allocator/resource observations
remain null/unverified; lifetime peaks are not per-job or whole-machine memory-fit
proof. A recommendation requires every declared repetition of comparable eligible
warm CUDA measurements. Numeric render-duration ranges use original available
frame count and observed warm processing FPS, excluding startup/model load and
preflight. The sample's observed range is not a confidence interval. Scene changes,
cuts, context, thermals and memory growth can invalidate it. There is no real-time,
quality or universal hardware guarantee.

All outputs and scratch are retained under fresh owned roots for local inspection;
calibration does not delete them automatically. The sampled output byte limit is
a soft stop, not a hard disk quota. Limits also bound jobs, frames, metadata and
wall time. Artifact failures and uncertain process containment are reported
separately, with retained paths. A final report is flushed to an exclusive pending
file before rename; failed publication is not a successful report commit.

Inspect report reads and formats the local report off the GUI thread. Display
projections are bounded and do not traverse full trial arrays on timer callbacks.
Reports expire after seven days or a backward clock; a caller-supplied changed
binding also invalidates them. Reopening does **not** reverify current hardware,
models or environment and says so. GPU name/index/CUDA descriptors are not a
physical GPU UUID or verified driver identity. Installed-package versions are
captured, not continuously revalidated. Do not reuse an old recommendation after
relevant conditions change without fresh calibration.

Focused validation in the compatible CPU environment:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen /private/tmp/corridorkey-compatible-venv/bin/python -W error -m unittest test_calibration test_calibration_task test_desktop_calibration test_desktop_window_calibration -v
```

These tests use explicit CPU subprocess and UI fixtures plus synthetic metadata
arithmetic; they do not load pretrained weights or establish actual CUDA speed,
quality, memory fit, Windows packaging or Resolve acceptance. To evaluate those,
provide actual hardware, local weights and representative local footage, verify
and run the dialog explicitly, retain reports, compare requested/effective settings
and visually evaluate each approximation independently.
