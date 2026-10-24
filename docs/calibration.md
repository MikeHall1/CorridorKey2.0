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

## Optional reference-agreement constraints (PERF-H)

The dialog's **Optional quality policy** field selects a local JSON policy. Empty
means the legacy **timing-only; quality NOT VERIFIED** objective. A malformed,
missing or unsupported selected policy never silently becomes timing-only.
Changing/clearing the field invalidates the captured/verified plan. Capture and
Verify run in the existing owned background task; selecting a path itself does
not parse it or run a model. Set a sufficient explicit read budget before capture
when selecting quality metadata.

Example policy (paths relative to this policy):

```json
{
  "schema_version": 1,
  "kind": "calibration_quality_policy",
  "objective": "fastest_warm_reference_agreement",
  "manifest": "references/alpha-sequence.json",
  "limits": [{"metric": "alpha.mae", "units": "alpha/pixel", "maximum": 0.01}],
  "accepted_tradeoffs": ["Declared references are agreement targets, not an independent visual acceptance verdict."]
}
```

Use an existing `alpha_sequence_evaluation`, `colour_sequence_evaluation` or
`temporal_sequence_evaluation` manifest, with exact source identity, declared FPS,
reference provenance and explicit reference/mask/background/flow paths. Frame IDs
are **sample output-relative**, not original source coordinates, and must lie
inside the captured sample. Source In remains in the measured job binding. Native
EXR alpha output must already be explicitly selected in every profile; colour
objectives also require their corresponding existing RGB output and association.
No output, precision or colour setting is changed to satisfy the policy.

Supported constraints use the existing dossier metric names/units: alpha MAE,
boundary MAE and forward-gradient MAE (`alpha/pixel`, `alpha/pair` for gradient),
foreground/composite MAE (`linear RGB/channel`), temporal residual MAE
(`alpha/pixel pair`). Each `maximum` is an absolute nonnegative reference error,
not a speed/quality combined score. All required metrics must be measured and at
or below their thresholds in **every declared measured repetition** before a
profile can be recommended. Equality passes. Missing, undefined, unsupported or
failed requested metrics exclude the trial with a reason; they do not become0.

The first supported objective chooses the fastest comparable warm profile for
preview/final purposes under those tolerances. Recommendations still require the
existing complete actual production-CUDA execution observations. CPU fixtures,
provider test doubles and passing reference metrics alone cannot generate real
recommendations. A `final` profile label remains a settings eligibility label,
not absolute/visual ground-truth quality. Reference agreement, direct numerical
parity and human visual review are distinct. Observed-memory optimization is
**not supported** in this increment: unprofiled/missing allocation observations
cannot become a trustworthy memory-fit objective or hard cap.

Policies are limited to64KiB, templates to1MiB,1..12 distinct constraints, up to16
bounded tradeoff statements and at most the captured1..120 sample-relative frames.
Policy/template/reference assets are single-link regular files, pinned alongside
source/models/code and protected from scratch/report destinations. Verification
accounts for metadata reads, hashing and the shared template preparation rereads
within the existing explicit read budget. The immutable plan stores policy bytes
semantics, template/source/reference hashes, definitions/provenance and all pins.
References, template/policy bytes and implementation membership are revalidated
before worker work and around trial evaluation; stale evidence fails closed.
Policy capture proves the plan's declaration precedes its own trials, not an
independent historical approval or authenticity assertion.

`benchmark_quality.prepare_quality_templates` and `evaluate_job_quality` bind
actual completed native EXR artifacts to each measured job/source/range/output.
Raw per-trial evaluation reports are retained as separately hashed artifacts;
metric values, populations, thresholds, pass/fail/unavailable status and exclusion
reasons remain in the calibration report. Cold and warm inference timers end
**before** posthoc evaluation; its cost is separately recorded and still consumes
the overall wall/output budgets. No inference speedup is inferred from these
portable tests. Cancellation is checked before/after the shared native evaluator;
individual decoder/evaluator calls cannot be forcibly interrupted. Cancellation,
partial trials or missing evidence never produce a quality-constrained winner.

Focused validation:

```sh
/private/tmp/corridorkey-compatible-venv/bin/python -m unittest tests.test_calibration tests.test_calibration_task tests.test_calibration_quality tests.test_desktop_calibration -v
```

Analytic PNG/NumPy/EXR fixtures exercise actual export evaluation, threshold
boundaries, missing/malformed policy, immutable identities/stale references,
read/output protection, evaluation timing/cancellation, repetition exclusions,
legacy compatibility and offscreen policy selection/capture/verification. These
are portable correctness evidence, not CUDA performance, learned-model quality,
long-clip resource, Windows or Resolve acceptance.
