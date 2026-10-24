# Recorded realtime feasibility (PERF-I)

The developer benchmark window has a **Feasibility** tab. Inspect an existing
inference/paired report or complete a benchmark, choose an analysis-only target
FPS (0.01..1000), and select **Update feasibility from selected report**. The
frame budget is `1000 / target_fps` milliseconds. Changing this target does not
change engine settings, quality, queue jobs, playback or benchmark protocols.
The projection records the target used; press Update after changing the target.

Inspection, bounded JSON parsing and projection happen on the panel's existing
owned background inspection thread. The UI timer consumes completed bounded
text, not report JSON. Existing close/busy guards wait for inspection/export to
finish; no Qt widgets are accessed by the inspection thread. Newly completed
benchmark reports automatically receive the same background projection.

`corridorkey.feasibility.inspect(path, target_fps)` also provides the same pure
projection outside Qt, using the existing strict report reader and bounded
regular-file reads. Source path/byte-size/SHA256 are retained. The module does
not load models, query hardware, run inference, download data or modify reports.
Rejected/failed samples without measurement runs retain explicit NOT VERIFIED observations; their raw diagnostics remain inspectable. Raw and baseline report inspection is independent of optional feasibility projection, whose failures remain visible in the Feasibility view. Malformed resource/policy/output-telemetry records are unavailable, never recorded measurements.

Inspection is bounded to32MiB input,128 pairs/256 run samples,4096 jobs and4096
stage records per job; oversized individual contextual observations are explicitly
unavailable and the displayed projection is capped at the existing2Mi-character
report-view limit. Invalid schemas/observations fail visibly, not as zero timings.

## What the dashboard shows

- Actual measured runs and both sides of pairs remain separate, preserving
  repetition identity; there is no synthesized cross-machine speedup or pooled
  mixture of cold and warm runs. CPU correctness reports never become FPS.
- Offline throughput is **submitted frames / measured runner wall time** only
  when inference succeeded and every job completed with a positive recorded
  frame count. Cold-process scope includes interpreter/import/model-load/shutdown.
  Warm-session scope follows the runner's unmeasured full-input warmup. A target
  comparison describes only recorded offline throughput, not sustained delivery.
- Per-job processing throughput excludes model loading; model-load and model
  initialization observations are distinct. First rendered batch is measured
  since diagnostics start and means CPU-rendered, **not encoded/durable/presented**.
  Missing counts or durations yield unavailable results, not estimates.
- Exact stage records preserve names, seconds, completion status and host-wall
  versus synchronized-wall method. Foundation, temporal, neural-refinement,
  postprocess/render, writer-submission and writer-finalization lanes show only
  matching observed scopes. Decode, transfer, encoding, delivery, queue and
  synchronization lanes remain NOT VERIFIED unless an exact named stage exists.
  The stage1 inclusive scope explicitly includes downstream stage2/render/submit;
  no stage scopes are summed, subtracted or treated as exclusive frame costs.
- Existing output telemetry is retained with its population/partial/unavailable
  status and exact scope. Queue acceptance is not encoding; EXR native-write
  return is not fsync durability. Render-ready, queue/native-write latency and
  completion-spacing distributions are not camera-to-display or playback latency.
- Existing runtime allocation/resource-policy snapshots retain their own scope.
  Processing allocation peaks are not whole-device usage or a hard memory cap.
  Cache/environment/repository/settings context is recorded rather than measured
  again. Missing observations are explicitly NOT VERIFIED or null in a named
  observation, never silently inferred from another lane.

Sustained presentation and instrumentation overhead are always **NOT VERIFIED**
here. Synchronized profiling perturbs execution; inclusive/overlapped scopes cannot
be added to predict realtime feasibility. No instrumentation is enabled or added
by this dashboard. A matched actual overhead experiment and source-arrival,
delivery and presented-frame observations remain required for stronger claims.
This is a connected evidence-inspection increment, not full PERF-I hardware or
product acceptance.

## Verification

```sh
/private/tmp/corridorkey-compatible-venv/bin/python -m unittest discover -s tests -p test_feasibility.py -v
/private/tmp/corridorkey-compatible-venv/bin/python -m unittest discover -s tests -p test_desktop_benchmark.py -v
/private/tmp/corridorkey-compatible-venv/bin/ruff check corridorkey/feasibility.py corridorkey/desktop_benchmark.py tests/test_feasibility.py tests/test_desktop_benchmark.py
/private/tmp/corridorkey-compatible-venv/bin/pyright --pythonpath /private/tmp/corridorkey-compatible-venv/bin/python corridorkey/feasibility.py corridorkey/desktop_benchmark.py tests/test_feasibility.py tests/test_desktop_benchmark.py
```

Portable fixtures exercise the existing inference/pair report schema, overlap,
missing/failed/CPU reports, target validation, source byte identity and partial
output telemetry. Actual offscreen Qt tests verify background report inspection,
Feasibility-tab connection, target update and source preservation alongside the
existing benchmark/process/export/close lifecycle cases. They make no GPU speed,
quality, memory, Windows or Resolve viewer/export claim.


## CPU stage-scope component observation — 2026-10-24

A separately predeclared, one-shot experiment measured only the existing
`RunDiagnostics.stage` no-op CPU context manager: `gpu=False`, no synchronization
callback, models, input frames or writers. Profiling off/on each executed10,000
scopes with the same name; two warmup pairs preceded five measured alternating
pairs. Diagnostics construction and explicit garbage collection were outside
timers. Timing used `perf_counter_ns` (internal stage clock `perf_counter`);
tracemalloc was disabled for timing and measured separately in three alternating
Python-allocation pairs. Default cyclic GC remained enabled.

On macOS27.0.1/arm64, Python3.12.15,10 logical CPUs, median off/on batch times were
**3.999/5.067ms**. Median paired on-minus-off cost was **113.45ns/scope**, with
**58.56–157.63ns/scope** observed across five pairs. All deltas were positive;
off/on timing sample CVs were4.20%/6.73%, passing the predeclared25%
condition-time/no-sign-change screen. The paired-delta CV was32.89%: magnitude
remains variable, without a confidence interval or a reliable per-frame budget.
This is component overhead, **not an optimisation, neural speedup or pipeline
overhead percentage**. The on/off ratio for an empty scope must not be applied
to real inference.

In the separate tracemalloc observations, retaining10,000 completed stage
records used **2,165,240 retained /2,166,096 peak traced Python bytes** versus
**9,760/10,616 bytes** with profiling off; each paired difference was2,155,480
bytes. These are traced Python allocations for this component, not RSS, native
allocation, RAM/VRAM usage or a hard resource cap. Counts were0off/10,000on;
all on records were `complete`/`host_wall`. Objects were released between cases.
CPU-brand and physical-RAM readouts were unavailable under sandbox permissions;
no escalation or alternate measurement was attempted.

Command: `/private/tmp/corridorkey-compatible-venv/bin/python benchmark-results/product-next/profiling-overhead-experiment.py`.
Exit0; measured experiment wall duration0.287s. Predeclared policy, raw pairs,
source/environment evidence and output are retained locally as
`benchmark-results/product-next/profiling-overhead-{policy,results}.json`,
`profiling-overhead-experiment.py` and `profiling-overhead-output.log`. All445
Python files matched the accepted batch5 source freeze before and after. No
application/test source, dependencies or configuration changed.

Cold startup, CUDA synchronization, full-pipeline instrumentation overhead,
model/frame throughput, representative-footage quality and real-time presentation
remain **NOT VERIFIED**. This limited observation does not replace those missing
measurements in the dashboard above.

## Bounded optional stage observations (PERF-I-001)

New profiled jobs aggregate repeated scopes by `(name, method, status)` rather
than retaining an event for every model/writer batch. The compatible `stages`
list retains `name`, `seconds`, `method` and `status`; `seconds` is now the **sum
of inclusive scope durations** in that group. Each row also records `count`,
`min_seconds`, `max_seconds` and `overflow`. This is not a chronological trace,
a frame-latency distribution or additive end-to-end timing. Nested/overlapped
groups must still not be summed into throughput or a realtime frame budget.

`stage_observations` identifies `aggregated_scopes_v1`, the exact observation and
overflow counts, and the scope/limits. At most64 groups are retained:60 named
groups and four reserved overflow groups separating host/synchronized timing
and complete/failed scopes. Names longer than128 characters, the reserved
`<stage-name-overflow>` name, and excess distinct groups enter those explicit
overflow summaries; unknown names never masquerade as a known dashboard lane.
No raw prefix, event timestamps, per-call ordering or omitted-name inventory is
retained. Public row dictionaries remain mutable for existing explicit CPU test
fixture annotations. Such annotations do not turn CPU observations into CUDA
resource evidence.

Legacy reports without this metadata remain accepted under the existing4096
raw-stage projection limit. New aggregate records retain counts/extrema in the
projection and are checked against their observation/overflow metadata. The
resource gate still requires completed synchronized observations and actual
CUDA provenance; host profiling is never relabelled synchronized. Profiling-off
jobs retain the original empty stage list, no aggregation metadata, and no
per-scope clock/synchronization work.

This bounds **stage telemetry only**, not every other record field, total RAM,
GPU memory or execution time. Worker frames retain their4MiB ceiling and reports
their existing size bounds. Controlled tests demonstrate previously rejected
10K-scope developer projection and60K-scope terminal encoding without running a
long video. Component retained-allocation/serialized-size comparisons use an
explicit pinned baseline and predeclared policy; they are not inference speed,
process RSS/VRAM, neural quality or target-platform evidence.

### Measured retention tradeoff (local CPU component)

The predeclared 10K same-name scope workload passed all three alternating memory
pairs: retained `tracemalloc` Python allocations fell from approximately 1.93 MB to
5,400–5,704 bytes; compact serialized diagnostics fell from 720,742 to 1,142 bytes.
All trials retained the exact 10K count and 10K **synthetic controlled-clock scope
seconds**; those synthetic totals are not real elapsed processing time. Five
separate alternating timing pairs after warmup measured actual workload wall
medians of 5.239 ms before versus 7.517 ms after; paired candidate/baseline times
were 1.338–1.508×, **slower**. The accepted improvement is bounded retention and
smaller records, with additional CPU aggregation cost—not a speedup. These are
Python diagnostics-component measurements, not process RSS, VRAM, GPU/inference
throughput, representative-footage quality or Windows/Resolve acceptance.

Raw policy, baseline bytes, invocation, paired measurements and full-source
provenance are under the ignored local directory
`benchmark-results/product-next/stage-retention/`. Its original `final-scope.json`
identifies the measured source/doc state; `accepted-scope.json` separately records
this later documentation-only results paragraph. No measurement was rerun.

## Optional source-reader CPU scopes (PERF-I-002)

With the existing `profile_stages` option, the shared runtime attaches a private
host observer to its per-job source metadata **after probing**. All interfaces
using that runtime receive the same observations; no reader/queue/default setting
or backend is changed. Private metadata is excluded from serialized source
records and immutable job/recovery/cache identities. Public reader constructor
calls, `include_review`, tensor triples, frame coordinates and caches remain
unchanged. Source-witness hashing is outside these scopes.

The developer projection now separates:

- `source_decode`: actual `cv2.imread`, `cv2.imdecode` and source `VideoCapture.read`
  **attempts**, including discarded video reads. A completed native call may
  return no frame; existing caller errors still fail the job. A call that raises
  records a failed scope. Counts are not proof of successfully decoded frames.
- CPU preprocessing siblings: `source_normalize` (image dtype/alpha/RGB handling),
  `source_colour` (image linear/display/primary conversion),
  `source_model_prepare` (image resampling/padding/model encoding/CPU tensors),
  and `source_preprocess` (video colour/model or draft-view preparation).

These are host-wall observations (`gpu=False`) even in reader/prefetch threads;
they never synchronize CUDA or masquerade as transfer/kernel measurements.
Sibling preprocessing block counts are not frame counts. Cache hits do not add
native decode/preprocessing observations. Repeated RVM/foundation/Stage1/Quick
passes contribute actual repeated work rather than deduplicated export-frame
costs. Existing inclusive parent scopes overlap these children and must not be
summed into end-to-end throughput.

Probe/open/seek, window stacking, hint decoding, source-witness hashing, GPU
transfers and sustained presentation are excluded; unavailable lanes remain
unavailable. Host allocation/dtype conversion is not a GPU transfer. Native
codec work cannot be interrupted inside a call.

A profiling-only short lock protects aggregate updates and consistent row/count
copies during snapshots. No lock is held during native calls, transforms, GPU
work, synchronization or JSON conversion. Disabled profiling has no observer,
per-operation stage contexts, clocks or collector locks; remaining dispatch/check
cost is subject to the predeclared reader-overhead experiment, not assumed zero.
After shared-runtime return, retired reader/prefetch ownership introduces no
retained diagnostic cycle: functional fixtures check immediate diagnostic release
after the caller drops its reference, without cyclic garbage collection. No weak-reference/release framework is
introduced, and one reader's close never disables other passes.

The earlier PERF-I-001 figures describe its pinned **unlocked** component state,
not the current synchronized accumulator. Fresh default-off reader, combined
profiling-on/off reader, and locked/unlocked collector experiments have separate
predeclared policies in ignored `benchmark-results/product-next/reader-profiling/`.
No inference/GPU/VRAM or general-format speed claim follows from that PNG8 CPU
cohort; HDR/video/alpha/source-colour tests establish functional parity only.

### Reader/collector component observations

The predeclared PNG8 CPU reader cohort used 16 generated 64×64 frames and 45 fresh
reader passes per arm (720 frames), warm filesystem caches and identical tensor
byte hashes. Five alternating default-off baseline/candidate pairs had ratios
0.981850–0.996374, meeting the declared 1.10 material-regression cap and 0.10
relative-spread condition (observed maximum 0.093748). Wall medians were 203.353 ms
before versus 201.576 ms after; this small difference is **not a speed-gain claim**.
Three separate reader profiling-on/off pairs observed 2.10–3.15% overhead, with
wall medians 200.107→205.256 ms; these are descriptive observations for this cohort.

Five separate 10K-scope collector pairs measured unlocked/locked wall medians
6.917→8.018 ms, but relative spread 0.219713 makes precise lock overhead
**INCONCLUSIVE**. Their preserved 10K controlled-clock scope units are synthetic,
not actual elapsed seconds. Source pixels, tensor shapes/dtypes and counts were
preserved; hashing used a `uint8` byte view of bfloat16 tensors, not a processing
precision change. The earlier PERF-I-001 retention and 5.239→7.517 ms figures refer
to its historical **unlocked** source, not the current lock implementation.

Raw evidence is in ignored `benchmark-results/product-next/reader-profiling/`:
`policy.json`, `policy-supplement.json`, `measurements.json`,
`experiment-invocation.json`, `experiment-stdout.log` and the exact
`measurement-harness.txt`. The original `final-scope.json` identifies the measured
state; `accepted-scope.json` and `doc-only-delta.diff` distinguish this later
results paragraph. No rerun or threshold change was made. These measurements do
not establish other-format performance, RSS/VRAM, GPU/inference throughput,
representative-footage quality or target-platform acceptance.

### Corrected lifetime-repair attribution

The original I002 candidate met its timing cap, but that was **not proof of
baseline buffer lifetime**. Two introduced helper-call argument references kept
raw integer decode arrays and pre-padding/resizing source views alive longer
than the original inline code. Restoring inline normalization/preparation fixes
those release points without an extra image copy. Weak-reference and exact
original tensor-byte checks verify the lifetime repair; no RSS or peak-memory
gain is claimed.

A separately predeclared reader-only revision retained the original baseline,
source recipe and thresholds. Five default-off pairs had ratios 0.965320–1.011161
and maximum relative spread 0.059562, passing the same 1.10 cap/0.10 noise limit;
wall medians 202.542→203.321 ms are **not a speed-gain claim**. Three profiling-on/off
pairs descriptively observed 1.69–3.45% overhead, with medians 201.597→207.978 ms.
Each arm again processed 720 PNG8 frames with exact tensor bytes and counts.
The unchanged collector was not rerun; its earlier noisy receipt remains the
only collector-overhead evidence.

Revision evidence is in ignored
`benchmark-results/product-next/reader-profiling-lifetime-repair/`, including
`policy-revision.json`, `baseline-parity.json`, `measurements.json`,
`experiment-invocation.json`, `experiment-stdout.log` and exact
`measurement-harness.txt`. Its original `final-scope.json` identifies the measured
repair; separate `accepted-scope.json`/`doc-only-delta.diff` identify this later
paragraph. Earlier evidence remains historical and unchanged. Scope is the
specified PNG8 CPU component, not other formats, RSS/VRAM, GPU/inference
performance, visual quality or target-platform acceptance.


## Optional CUDA upload and synchronization observations (PERF-I-003)

With stage profiling enabled on an available CUDA backend, the shared runtime
captures its selected device index and device-bound synchronizer. A continuous
RVM **cache miss** can now record `rvm_rgb_upload_and_float` when its input is an
actual CPU tensor and its CUDA target has that same explicit index. The scope
retains the original inline `display.unsqueeze(0).to(device,
non_blocking=True).float()` operation, including its dtype promotion and temporary
lifetimes. It is **completed synchronized wall time for upload plus float
promotion**, not pure DMA, bandwidth, all transfers or model inference. Cache-hit
state transfers and other tensor/device transitions remain unmeasured. CPU,
profiling-off, unavailable CUDA, generic callback and mismatched/implicit-index
paths do not produce this transfer observation.

Runtime-owned pre/post stage callbacks also record `cuda_sync_before_scope` and
`cuda_sync_after_scope`: bounded aggregated **host wall waits** for their existing
synchronization calls. Wait observation itself adds no synchronization calls and
never holds the collector lock while waiting. The new upload scope does add its
normal profiling-only pre/post pair; it is not wrapped in another GPU stage.
Existing stage/body/callback failure behavior and generic callback clock behavior
are retained. Pre-waits precede their stage clock, while post-waits overlap stage
time; nested stages can overlap both. These wait lanes must **never be summed**
with transfer, inference or other stages to estimate exclusive costs. They do not
attribute queued GPU work to a particular operation.

The Feasibility transfer/synchronization lanes display these exact names. Raw
snapshots additionally identify the captured device and whether its original
runtime callback remains active in `cuda_stage_observations`. No self-bound
callback or retained diagnostics cycle is introduced. Reader host scopes remain
`gpu=False` and do not synchronize CUDA.

Portable tests verify admission, captured-device mismatch/change, callback/clock
and error populations, lock-free waits, cache-hit/miss CPU parity, inline expression
identity and dashboard mapping. Mocked callback scopes are plumbing evidence, not
GPU transfer measurements. **Actual CUDA timing, profiling overhead, transfer
performance and representative-footage quality remain NOT VERIFIED.** No GPU
speed, RSS/VRAM or default-off performance improvement is claimed.


## Optional native image encode/write service (PERF-I-004)

Profiled shared-runtime EXR writers record `native_image_encode_and_write` in
Feasibility's encoding lane. Each host-wall scope surrounds the existing
`cv2.imwrite` call and its false-return failure check, inside the existing writer
consumer thread. This is **combined native image encoding and filesystem write
service**, not pure encoding, encoder CPU time, fsync durability, queue/backlog
latency or playback. EXR parameters, array preparation, half-float saturation,
queues and native writes remain unchanged. Profiling-off passes no host observer;
no new CUDA synchronization is introduced.

The scope excludes pre-write frame preparation and completion callbacks. Counts
are native calls per output image; restored prefixes are not rewritten or timed.
Failure/exception scopes remain failed and preserve the existing writer error.
The bounded thread-safe stage collector is updated outside native I/O; writer
retirement still precedes final diagnostics. Multiple output writers and other
stages can overlap, so service totals are **not additive end-to-end time**.
Existing render-ready-to-native-completion latency includes queue/backlog and is
therefore a different observation, not this service duration.

Portable actual small-EXR tests check signed/near-zero float pixels, half alpha,
profile off/on byte parity, original array storage/parameters, worker-thread and
lock boundaries, cancellation, false returns/native exceptions, diagnostic
callbacks and observer retirement. CPU-worker integration checks profile-only
wiring. These checks establish coverage and compatibility, not performance.
Instrumentation overhead remains NOT VERIFIED without a separately matched
measurement. FFmpeg stdin submission remains backpressure, not encoder timing;
actual FFmpeg execution and encoder-only attribution remain NOT VERIFIED.
