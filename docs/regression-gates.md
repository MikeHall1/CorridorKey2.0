# Report-level regression screens (PERF-J)

`python -m corridorkey.regression_gate` reads an **existing** completed version1
`inference_pair` report. It does not run inference, alter a benchmark, choose a
winner, or claim machine performance from portable tests. Legacy runtime defaults
and existing benchmark commands are unchanged.

## Declare before evaluation

Write a version1 policy, commit/review it **before collecting candidate results**,
and pin its exact byte SHA256 in the invoking automation. Supply that pin rather
than calculating it from an untrusted policy at evaluation time:

```sh
python -m corridorkey.regression_gate \
  --report benchmark-results/paired.json \
  --policy benchmarks/suites/my-regression-policy.json \
  --policy-sha256 <previously-recorded-64-hex-byte-digest> \
  --output benchmark-results/new-regression-screen.json
```

The required policy shape is:

```json
{
  "schema_version": 1,
  "kind": "regression_policy",
  "context_sha256": "<canonical expected context SHA256>",
  "repetitions": 3,
  "performance": {"max_elapsed_regression": 0.05, "max_relative_spread": 0.1},
  "quality": [{"job": 0, "metric": "alpha.mae", "units": "alpha/pixel", "max_increase": 0.0001}],
  "required_evidence": ["numerical_parity", "observed_memory"],
  "visual_review_required": true,
  "accepted_tradeoffs": ["Explicit description of deliberately accepted changes; does not waive a failed gate."]
}
```

This is a shape illustration, **not a runnable or accepted policy**. The context
is `regression_gate.context(report)` serialized by `regression_gate.digest`.
Establish its expected value from an independently reviewed workload specification
or prior matched calibration report, not by blessing the candidate being gated.
It binds actual repository/assets/weights/corrections, hardware/dependencies,
requested and effective per-job settings, actual source/hints/writer and model-reuse records, actual model lifecycle/full warmup/cache metadata, frozen quality templates and result-free quality/reference/frame/definition contracts,
reported timing/warmup/filesystem-cache conditions and exact repetition count.
Ephemeral output destinations alone are excluded. Same-engine configuration
comparisons only: cross-revision pairs are not supported by the pair runner.
Uncontrolled filesystem cache is recorded, not silently called cold or warm.

A policy pin proves unchanged bytes relative to the supplied pin, **not chronology,
authorship or authenticity**. Every output explicitly marks chronological
predeclaration NOT VERIFIED. External CI/review records must establish it. Likewise
report hashes identify the raw bytes inspected; they do not authenticate measurements.

## Decisions and evidence

- The existing strict report reader validates terminal pair structure and alternating
  order. The gate additionally rematches frozen jobs and actual requested/effective
  settings and checks identity stability across **every** sample. It requires
  3..128 paired repetitions. Missing context, partial/cancelled source runs,
  mismatched policies, unsupported schemas and invalid data are `not_verified`.
- Performance uses raw measured elapsed time, never the report's cached speedup.
  Candidate/baseline elapsed ratios must stay within `1 + max_elapsed_regression`;
  equality passes. Any ratio envelope crossing the limit is `inconclusive`.
  Spread is `(max-min)/median`, checked on each side and paired ratios. Exceeding
  `max_relative_spread` is `inconclusive`, even for apparent improvements or regressions.
  This is an observed envelope, **not a confidence interval or statistical power claim**.
- Quality limits are absolute **increases in error** for explicit job/metric/unit
  selections. Supported metrics are the existing dossier's alpha MAE, boundary
  MAE, forward-gradient MAE, foreground/composite linear RGB MAE and temporal
  residual MAE. Existing quality schema, reference/source/range/binding,
  metric/population/geometry and repeated measurement contracts are revalidated.
  Equality passes; any observed increase above tolerance fails. Missing requested
  metrics stay `not_verified`; no error is invented from timing.
- `numerical_parity` (direct output-to-output numerical comparison), `observed_memory`
  and human visual review have no independently bound acceptance evidence in this
  pair schema. Requiring them yields `not_verified`. Reference-agreement MAE is
  **not** direct pixel parity or absolute visual quality. Required visual review
  cannot be waived by arbitrary approval JSON. Tradeoffs remain explicit recorded
  declarations, not overrides. Add real supported evidence before enabling product
  acceptance; do not turn off requirements merely to obtain green output.
- Aggregate precedence: `fail`, `not_verified`, `inconclusive`, then `pass`.
  `pass` means only the selected supported screen passed. With empty quality and
  evidence requirements it is deliberately just a timing screen, not quality or
  product acceptance. Cancellation during evaluation gives `cancelled`.

Bounded regular-file reads reject duplicate keys, nonfinite JSON and oversized or
excessively nested inputs. Output is exclusively created at a new destination,
with shared report-path alias protection; sources and policies are never overwritten.
Outputs preserve raw input byte hashes/sizes, exact policy, context digest, raw
elapsed ratios and per-repetition quality changes. Exit0 means screen pass;
exit1 includes fail/inconclusive/not-verified; cancellation exits130.

## Portable validation

```sh
/private/tmp/corridorkey-compatible-venv/bin/python -m unittest discover -s tests -p test_regression_gate.py -v
/private/tmp/corridorkey-compatible-venv/bin/ruff check corridorkey/regression_gate.py tests/test_regression_gate.py
/private/tmp/corridorkey-compatible-venv/bin/pyright --pythonpath /private/tmp/corridorkey-compatible-venv/bin/python corridorkey/regression_gate.py tests/test_regression_gate.py
```

Analytic fixtures deliberately regress raw timings and actual generated EXR alpha
reference-agreement errors, exercise exact thresholds, noise, missing/mismatched
context, policy tampering, cancellation and source/output aliases. These are
portable correctness checks, not GPU throughput, learned-model quality, memory,
Windows or Resolve acceptance.

## Version2: captured native numerical agreement (PERF-J-004 increment)

Version1 and its timing-only API/defaults remain unchanged. Version2 adds exactly
one `numerical` field and requires `numerical_parity` in `required_evidence`:

```json
"numerical": {
  "constraints": [
    {"job": 0, "metric": "alpha.mae", "units": "alpha/pixel", "maximum": 0.0001},
    {"job": 0, "metric": "colour.composite_mae", "units": "linear RGB/channel", "maximum": 0.001}
  ],
  "budgets": {"max_files": 4096, "max_bytes": 268435456, "max_pixels": 4194304}
}
```

This is an additive schema illustration; retain every other required policy field,
use `schema_version:2`, declare workload/repetitions/tolerances **before candidate
collection**, and independently pin the policy byte SHA as above. The policy does
**not** contain a post-run evidence digest: no acceptance-policy rewrite is needed
to attach newly collected results. A post-run capture receipt links back to the
unchanged policy SHA and exact paired-report/export bytes:

```sh
python -m corridorkey.regression_gate --report paired.json --policy policy-v2.json \
  --policy-sha256 <predeclared-policy-sha> --capture-numerical --output numerical.json
# Record the numerical.json byte SHA in the invoking run's evidence receipt.
python -m corridorkey.regression_gate --report paired.json --policy policy-v2.json \
  --policy-sha256 <same-policy-sha> --numerical-evidence numerical.json \
  --numerical-evidence-sha256 <captured-evidence-byte-sha> --output screened.json
```

The Python equivalents are `capture_numerical(...)` and the optional
`numerical_evidence_path` / `numerical_evidence_sha256` arguments to `evaluate`.
Evidence capture is local and explicit, never a benchmark runner or model load.
Chronological predeclaration/authorship/authenticity remain NOT VERIFIED: hashes
bind bytes, not the truth of arbitrary measurement JSON. External run/review
records must establish measurement provenance. The usable workflow permits those
records to predate candidate collection without pretending to prove chronology.

### Exact numerical scope

- Paths are derived exclusively from each admitted completed pair's measured job
  records/output directories, not a caller-supplied list of images. The evidence
  pins report/context/policy, actual job IDs/output/source/range/FPS/native geometry,
  source bytes, execution/colour/channel/dtype contracts, every selected output
  frame and full file identities. Requested/effective/model/cache/hardware/code
  report identity remains bound through the existing pair context. Source bytes
  must match the actual report's asset identity. Unknown/incompatible bindings are
  unavailable, not silently normalized.
- This first adapter supports native-size, unprofiled, homogeneous HALF/FLOAT EXR
  alpha and optional **straight foreground** RGB from completed outputs. Every
  submitted frame is included, at most120 frames/job and256 pair-job selections.
  Pixel count is explicitly bounded by the unchanged4Mi-pixel default admission
  ceiling; there is no implicit UHD/precision/colour conversion. Native/output
  geometry, declared source range/FPS, channels and actual encoded dtype must
  agree. Image-sequence FPS must match the requested `frame_dir_fps`; video/file
  inputs instead use their finite actual native FPS, matched across both sides. Unsupported layouts/associations/colour bases remain NOT VERIFIED.
- Baseline exports become the references in the existing
  `benchmark_quality.evaluate_job_quality` and native alpha/colour sequence
  evaluators. No new metric implementation or caller-declared result is trusted.
  `alpha.mae` is mean absolute coverage difference. `colour.composite_mae` is
  **linear black-composite reference agreement**, using straight foreground and
  its alpha on each side. Signed/HDR RGB is preserved without clipping; alpha0
  contributes no invisible RGB and near-zero alpha is naturally weighted.
  This is **not raw foreground pixel parity**, invisible-colour parity, bitwise
  identity, absolute ground-truth quality or human visual approval. Declare both
  metrics when alpha agreement and visible foreground agreement are required.
- Every declared metric/job must be measured for every pair. Absolute thresholds
  are inclusive (`value <= maximum` passes). Honest capture of deliberately wrong
  exports still produces numerical FAIL: integrity alone is not metric acceptance.
  Missing/corrupt/stale evidence or required metrics are NOT VERIFIED; timing noise
  remains INCONCLUSIVE independently of numerical outcomes. Existing aggregate
  precedence is unchanged. Posthoc evaluation is never folded into timing samples.

Metadata is limited to4MiB; policy constraints to32. There are at most4096 unique
files and a conservative explicit read/evaluation budget up to1GiB, charged before
hashing/decoding using `sum(file_bytes)*(8+16*pair_job_count)`. This deliberately
bounds source/evaluator rereads and final fences rather than claiming precise I/O
telemetry. Native per-file encoded/pixel admission precedes existing decoding.
Cancellation is forwarded through `evaluate_job_quality(..., check=...)` to
bounded export scans, source/reference hashing and existing per-frame sequence
evaluators/readers. A request observed after one decoded frame stops before later
frames are decoded; an individual native decoder call cannot be forcibly interrupted.

Source, policy, evidence and actual export files/directories are protected at
publication admission and final fences, including single-link/lexical checks,
hardlink aliases, late parent redirects and fresh exclusive output creation.
Evidence is reconstructed from actual records and revalidated after evaluation.
These are cooperating-process guards, not a hostile-filesystem transaction or
cryptographic measurement attestation. Raw reports/exports are never overwritten.

Version2 alone leaves observed-memory and visual-review contracts NOT VERIFIED.
Version3 below implements the separate memory phase; the following original boundary
notes explain why raw profiled timing cannot substitute for throughput. Next visual boundaries: a separate **profiled resource** receipt must bind the same
workload/settings/hardware/source but never replace unprofiled throughput samples,
state allocator/process scope and sample coverage, and use explicit thresholds;
a human review receipt must bind actual review exports/frame IDs, policy/tradeoffs,
reviewer identity and decision without treating a declaration as authenticated
approval. These original boundary notes are addressed by version3 resource evidence and the limited version4 declaration contract below; authenticated human approval is still not implemented or implied by numerical PASS.

Portable tests include actual analytic CPU-owned EXR outputs, unchanged
predeclared tolerances, real alpha/foreground changes **before integrity capture**,
HDR/zero/near-zero alpha, mismatched source/range/colour/dtype/execution/coverage,
missing/tampered/corrupt/overbudget/cancelled evidence, CLI capture/evaluate,
no-overwrite and late hardlink/parent-alias fences. These checks make no GPU
performance or visual quality claim.


Numerical execution matching preserves each side's complete captured execution
record but compares explicit invariant backend/device/GPU/CUDA/precision, colour,
alpha, temporal and nonfinite semantics. Allowed requested settings can legitimately
change foundation cache dtype/extraction policy and observed stage2 call/evaluation
counters; those remain separately pinned per side, not mistaken for hardware or
semantic mismatches. Actual device/backend/precision/colour incompatibilities still
refuse admission. Producer-shape regressions cover30fps video with an irrelevant
24fps image default, cache and stage2-batch variants, the sequence-FPS guard, and
per-frame cancellation; these remain analytic CPU fixtures, not CUDA measurements.

Each side's foundation-cache precision is validated against its own captured
effective configuration: float32 when low-VRAM or a positive foundation cache
budget is enabled; None when both are off. This permits the legitimate
cache-off→float32-cache-on comparison but refuses missing/float16/otherwise
inconsistent cache precision without trusting a fresh integrity capture.

## Version3: separate observed-memory phase

Version3 adds required `observed_memory` evidence and a `resources` contract.
Version1 timing-only and version2 numerical policies are unchanged. Version3 may
also include the same optional `numerical` contract when numerical evidence is
required; it never silently upgrades old policies or accepts a historical raw
JobResult as resource provenance. Add this field to the other required policy
fields, set `schema_version:3`, and predeclare/pin the unchanged policy **before**
collecting either phase:

```json
"resources": {
  "constraints": [
    {"job": 0, "metric": "peak_allocated_vram_bytes", "maximum_bytes": 8589934592},
    {"job": 0, "metric": "peak_reserved_vram_bytes", "maximum_bytes": 10737418240}
  ],
  "limits": {
    "max_jobs": 24, "max_frames": 240, "max_wall_seconds": 900,
    "max_output_bytes": 1073741824, "read_budget": 34359738368, "max_files": 100000
  }
}
```

The only supported resource metrics are per-job **CUDA allocator allocated and
reserved peak bytes**, not process-lifetime RSS, host RAM, whole-device usage,
model-load/startup peak, an instantaneous free-memory estimate or a hard cap.
Both peaks must be positive exact integers and reserved>=allocated. Each explicit
job/metric constraint applies to the maximum across **both sides and every measured
repetition** (unmeasured warmup peaks are retained separately). Equality passes;
any actual measured exceedance fails. Missing/cancelled/partial/fixture/unknown
observations are NOT VERIFIED, never zero or a made-up resource improvement.

Run the explicit companion phase, then pass its separately recorded byte receipt
to the existing gate. Do not insert an unknowable post-run SHA into the acceptance
policy:

```sh
python -m corridorkey.regression_resources --report paired.json --policy policy-v3.json \
  --policy-sha256 <predeclared-policy-sha> --output-parent <existing-separate-parent>
# The command prints a NEW owned attempt/resource-evidence.json path.
# Record its exact byte SHA as this run's evidence receipt.
python -m corridorkey.regression_gate --report paired.json --policy policy-v3.json \
  --policy-sha256 <same-policy-sha> --resource-evidence <attempt/resource-evidence.json> \
  --resource-evidence-sha256 <receipt-byte-sha> --output new-screen.json
```

`run_phase(...)` returns an immutable `ResourceAttempt` with report path and any
retained uncertain worker owner. API-injected providers are explicitly marked
`fixture_provider` and **cannot** qualify as production neural-memory evidence.
Raw records reporting CPU/unknown/non-neural execution are also refused. Portable
record-shape simulations test admission/arithmetic only; they are not actual GPU
measurements, model compatibility, memory-fit or target acceptance. Hashes bind
local bytes, not cryptographic measurement authenticity or human approval.

### Connected producer and lifecycle

The phase derives immutable jobs from the admitted unprofiled pair, preserving
requested/effective settings, writer, source range/colour/hints/corrections and
model paths. Only the top-level `profile_stages=True` flag and fresh owned output/
temporary destinations differ. It uses the existing owned `WorkerClient` and
shared inference engine. Each original side/repetition gets a fresh worker;
warm-session samples reproduce one complete unmeasured warmup per job followed
by measured jobs with verified model reuse. Cold samples begin with a fresh model.
Alternating pair/side order and full job/frame/repetition coverage are retained.
Unknown cleanup retires no further work; callers must retain an uncertain owner
and resolve its retirement rather than launching another model owner.

The first phase deliberately supports **image-sequence inputs only**; video is
rejected before model work rather than misusing `frame_dir_fps`. Enabled shot
profiles are also explicitly unsupported because their immutable base binding
includes the unprofiled flag; no plan is silently rekeyed or dropped. Existing
source/hint/model/correction bytes and implementation files are pinned and
verified before work, fenced around jobs and checked afterward. Actual benchmark
repository/environment/dependency/CPU/RAM/driver metadata is reused; missing or
changed driver information is unavailable, not fabricated. Profiled record GPU
name/logical index/CUDA/autocast/source/output/frame/model identities must match
the unprofiled pair. Physical-board serial identity and real multi-GPU behavior
remain unverified where the original producer did not record them. No private
environment data is invented or uploaded.

The runtime already resets allocation peaks for each profiled job after model
loading and reports the peaks after completed processing. Profile-only calls now
bind reset, synchronization, allocated/reserved reads and device-name query to
one captured selected index; additive `device_index`, `reset_device_index` and
`reset_scope` fields establish that scope. No device remapping, inference math,
non-profile defaults or precision policy changed. Resident model allocations and
reservations remaining from a warmup are still included after reset; prior
high-water history and model-loading peaks are excluded. Mocked index1 API tests
do not establish actual GPU1 behavior.

### Bounds, exclusions and proof limits

No time measured by this synchronized phase enters `inference_pair` throughput,
ratios or noise decisions. Its separate elapsed duration is bookkeeping only;
profiling overhead is not claimed without another matched actual experiment.
The original unprofiled report remains byte-identical. Required memory and
numerical/quality checks remain independent of timing-noise status.

Limits cover the complete measured+warmup workload, output/artifact files,
provenance reads and wall time. They allow at most1024 jobs,122880 submitted frames,
100000 files,86400 seconds and1TiB output/read budgets. The declared verification
read charge is conservative: `6*sum(pinned_input_bytes)+3*max_output_bytes`,
covering provenance/evaluator/receipt admission rereads. Existing repository
hashing has optional streamed-byte/cancellation bounds used only by this phase;
legacy callers retain their defaults. Metadata/receipts are bounded to32MiB and
raw job records to4MiB. Actual inference I/O is not mislabelled as verification
I/O. Output/wall checks are cooperative sampled bounds, **not hard disk or kernel
runtime limits**. Native decoder/GPU calls cannot be forcibly interrupted in
process; the owned worker lifecycle supervises retirement.

Reports, models, source/corrections/hints, existing exports and the new resource
attempt are protected at admission and publication. Each attempt is exclusive;
old renders/receipts are never overwritten. Single-link/pinned byte identities,
root identity, file/byte budgets and late alias fences are checked. If no output
budget remains for a terminal receipt, artifacts are retained and the command
reports NOT VERIFIED rather than overrunning the declared budget to publish PASS.
Missing CUDA/local models never trigger a download or alternate measurement backend.

The limited version4 visual-declaration increment is documented below. A user
declaration is not authenticated inspection or calibrated display approval; a
verified-human requirement still remains NOT VERIFIED without a real trust basis.

Resource admission validates original records with a validation-only clone at the
benchmark's **actual measured artifact destination**, not its now-retired frozen
staging destination. The frozen profile template is not mutated; source/settings/
writer/hints and every repetition's source/model/output contract remain checked.

Resource source/hint identities use the already verified ordered frame pins and
the shared legacy sequence-identity serializer, with no second unbounded content
read. Bounded source/hint and model membership scans run before workers and before
completion, so added/removed frames or model members invalidate the receipt. The
legacy identity wire format and natural frame order are unchanged. Correction
asset hash keys use admitted canonical paths: a legitimate relative `../` asset
remains supported, but its original expected SHA cannot be skipped by lexical
path differences. Alias rejection is retained; no correction file or acceptance
policy is rewritten.

Cancelled/failed phases retain raw records/files but skip expensive artifact
content hashing and identity inventory. `artifact_verification` stays NOT VERIFIED
and such receipts cannot satisfy a memory gate. The existing bounded metadata-only
size scan still admits terminal-receipt output against the output budget; it is a
cooperative boundary, not a hard realtime cancellation guarantee. Successful
receipts retain full streamed hashes, descriptor/fingerprint/deadline guards and
final input/artifact fences.

## Version4: bound visual declarations, not authenticated inspection

Version1–3 policy schemas and their existing visual-review behavior are unchanged:
`visual_review_required:true` still remains NOT VERIFIED for those versions.
Version4 requires the existing numerical contract and `numerical_parity` evidence,
`visual_review_required:true`, and one additional field:

```json
"visual": {
  "mode": "bound_declaration_required",
  "selections": [{"pair": 0, "job": 0, "source_frames": [12, 15]}]
}
```

`source_frames` are **absolute source indices**, not output-local indices. Rows
must be ordered and unique by `(pair,job)`, with ordered unique frame indices.
Bounds are1..128 rows,1..120 frames per row and1024 selected frame IDs total.
Every selected frame must exist in the currently pinned exports for **both**
baseline and candidate. Selected coverage can be a subset of the measured range;
it does not claim visual review of other frames or shots. Bindings retain actual
job IDs, source/range/FPS, execution/colour/alpha contracts, native precision and
selected alpha/available straight-linear foreground EXR paths and byte SHA256.
No new display conversion, viewer, background-composite inspection or calibrated
display workflow is implemented.

Version4 can independently include the existing version3 `resources` contract;
if present it requires `observed_memory`, and vice versa. Resource thresholds,
numerical tolerances, quality checks and unprofiled timing remain independent.
A declaration cannot override any of them. Do not use profiled resource timing
as throughput evidence.

### Existing reviewer-authored declaration

First pin the complete version4 policy and capture numerical evidence through
the existing `--capture-numerical` command. The reviewer then supplies a separate
local JSON object with **all** these exact fields already present:

```json
{
  "schema_version": 1,
  "kind": "regression_visual_declaration",
  "reviewer": "<reviewer-provided name>",
  "outcome": "unable_to_assess",
  "policy_sha256": "<exact externally pinned policy byte SHA256>",
  "numerical_evidence_sha256": "<exact numerical receipt byte SHA256>",
  "selections": [{"pair": 0, "job": 0, "source_frames": [12, 15]}],
  "accepted_tradeoffs": ["<exact entries from the pinned policy, in order>"],
  "observations": "<reviewer-provided observations; empty is permitted>"
}
```

This is an incomplete illustrative template, **not an acceptance declaration**.
Use actual hashes, name, selections and the complete policy tradeoff list. Only
the reviewer should deliberately choose `accept`, `reject` or `unable_to_assess`.
Capture does not supply missing claims, infer inspection, fill identity fields,
or automatically turn a template into approval. Name≤256 characters,
observations≤4096 characters, declaration≤256KiB; finite, duplicate-free JSON
with nesting≤64 is required. Receipt metadata is bounded to4MiB. Limits reject
rather than truncate.

```sh
python -m corridorkey.regression_gate --report pair.json --policy policy-v4.json \
  --policy-sha256 <policy-byte-sha> --numerical-evidence numerical.json \
  --numerical-evidence-sha256 <numerical-byte-sha> --capture-visual \
  --visual-declaration reviewer.json --visual-declaration-sha256 <declaration-byte-sha> \
  --output visual-receipt.json

python -m corridorkey.regression_gate --report pair.json --policy policy-v4.json \
  --policy-sha256 <same-policy-byte-sha> --numerical-evidence numerical.json \
  --numerical-evidence-sha256 <same-numerical-byte-sha> \
  --visual-evidence visual-receipt.json --visual-evidence-sha256 <visual-receipt-byte-sha> \
  --output screened-v4.json
```

Python capture is `regression_visual.capture(report, policy, policy_sha,
numerical_path, numerical_sha, declaration_path, declaration_sha, check=...)`.
`regression_gate.evaluate` accepts optional `visual_evidence_path` and
`visual_evidence_sha256`. It reconstructs current evidence and requires exact
receipt equality, not only a self-declared hash. Inputs and actual exports are
protected at early admission and final publication, with existing single-link,
lexical path, file-identity and exclusive-output guards. Cancellation is
cooperative; these guards are not an atomic hostile-filesystem transaction.

### Outcomes and trust boundary

- `bound_declaration_required`: valid `accept` passes **only the bound declaration
  check**; `reject` fails; `unable_to_assess`, absent or invalid evidence remains
  NOT VERIFIED.
- `verified_human_review_required`: a declaration alone **never** satisfies the
  human-review requirement. Valid `accept` remains NOT VERIFIED; explicit valid
  `reject` still fails.
- Every supported result explicitly records `human_inspection_status:not_verified`.
  Reviewer name/authorship, actual inspection, calibrated display, chronological
  predeclaration and approval authenticity are **not authenticated** by local
  hashes. Overall gate PASS in declaration mode is not authenticated visual
  acceptance, representative-footage quality, GPU parity or product release
  certification. Required real visual/target-platform acceptance remains separate.

Portable analytic EXR fixtures in `tests/test_regression_gate_visual.py` exercise
binding and rejection semantics, absolute offsets, both sides, strict JSON,
resource/timing/numerical independence, cancellation, stale inputs and no-overwrite
publication fences. Generated fixture declarations are explicitly labeled as
fixtures; they are not evidence of an actual human inspection.
