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
