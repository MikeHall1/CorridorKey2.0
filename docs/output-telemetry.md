# Detailed output telemetry

M39 mechanics are locally verified. Existing `--profile_stages` or the desktop **Profile stages** checkbox activates detailed output observations; the default remains off. Use `--diagnostics FILE` to save CLI records. No new processing option, approximation, colour conversion or inference backend is introduced.

The `diagnostics.output_telemetry` record describes the selected output timeline:

- **Render-ready offsets:** processing-start to the return of a rendered batch, shared by its frames. Quick Key measures final rendered frames, not sparse Stage2 anchors.
- **Queue-accept latency:** render-ready to every requested writer accepting that frame's batch. Acceptance is not encoding completion.
- **Native-write latency:** render-ready to every requested EXR writer returning successfully from `cv2.imwrite` for that frame.
- **Native completion offsets:** processing-start to that all-output native-write event.
- **Completion spacing:** differences between adjacent output-relative frames' all-output native-completion timestamps.

All durations are seconds. Exact nearest-rank p50/p95/p99 use `ceil(p*N)` with no interpolation, alongside minima, maxima and population/coverage counts. Native completion means a successful native EXR write return; it does not promise fsync, power-loss durability, source arrival, per-frame neural service time, display presentation or playback readiness. Encoder queue acceptance does not supply per-frame encoded or presented latency. Those video measurements remain unavailable.

The clock origin is the existing post-model-load processing start. Model initialization, cold startup and preflight timing remain separate parent-record measurements. Output frame indices are relative to the selected output sequence; parent source diagnostics retain source offsets, frame count and FPS. This summary does not record source-arrival timestamps.

Profiling synchronizes existing GPU stage boundaries and adds observation/aggregation overhead. The existing inference and paired throughput runners reject profiled jobs. Use these diagnostics separately from ordinary throughput measurements; no speedup follows from these timings alone. CPU fixture clocks verify mechanics only and do not establish GPU timing, VRAM usage or professional quality.

Registration happens before queue submission. An EXR worker may finish a frame before `write_many` returns; the collector accepts that order. Write duration always starts at the render boundary, so a submission-return timestamp is never substituted as a later write origin. Different writer threads may deliver captured timestamps in a different global arrival order. Native order is validated per output.

Admission defaults to at most 100000 exact frame samples and 4096 in-flight frames, with at most 16 requested outputs. Over-limit or invalid observations produce explicit partial/unavailable telemetry, with no biased prefix percentile presented as a whole-job distribution. These are telemetry memory bounds, not process RAM, VRAM or output-storage limits. Missing output events, cancellation and failures keep incomplete coverage visible. Observation failures do not replace a primary rendering/writer/cancellation error.

Finalization observes the existing owned-writer cleanup; the collector neither joins workers nor starts I/O. Cached snapshots use bounded JSON data and do not query clocks or hardware. Actual regular and Quick Key CPU worker/Qt workflows preserved nine source files and produced byte-identical profiled/unprofiled alpha/foreground EXRs on a five-frame selected range. Complete745CPU tests (48.315s) and full compile/import/dependency/whitespace/static regression gate pass, with zero new static findings; existing backlog remains. Independent review accepted four focused checks plus a concurrent finish/snapshot probe. Evidence: `benchmark-results/m39-verification-summary.json`. These fixtures establish mechanics, not GPU performance or professional quality.
