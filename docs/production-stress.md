# Production stress workflow

The explicit headless workflow and Developer benchmark runner use the existing shared engine and owned workers:

```sh
python -m corridorkey.stress_suite --manifest benchmarks/suites/production-stress.example.json --output /existing/local/results/stress.json
```

M40 orchestration mechanics are locally verified: complete787CPU/no skips, zero new static findings, main/independent review and actual Qt/owned-worker fixture workflows. See `benchmark-results/m40-verification-summary.json`. Production stress acceptance remains unverified. The example names operator-provided local batch manifests; it does not download or create production assets. Each batch uses the existing `inference_batch` schema. The suite captures immutable requested jobs, batch/source hashes, repository/environment provenance and, for production runs, selected model-file hashes. Default and explicit model paths, sources, hints, corrections, original output/temp paths and input manifests are protected from artifact placement. Production model/source bytes and bounded model-directory membership are checked again before complete status. Python-only fixture injection is absent from the manifest schema.

Nine required scenarios remain visible even when not configured: at least ten minutes of selected source, 50 distinct paths and source-content identities, repeated jobs in a persistent worker, two isolated workers with actual engine invocation overlap, cancellation after readable EXR frames, hard interruption followed by explicit recovery and an uncached export oracle, memory growth, encoder failure and disk full. Fixture outcomes verify mechanics only. Engine invocation overlap does not establish overlapping GPU kernels.

Memory checks run warm-up followed by at least ten jobs in the same worker, sampling current process-tree RSS while quiescent. Growth above the fixed maximum of 128 MiB or 10% of warm RSS fails the screen; a positive trend below that ceiling remains unverified pending explanation. Missing RSS is unavailable, never zero. No forced garbage collection or cache flushing is used to manufacture a pass.

Encoder-failure evidence requires the expected typed native failure. A missing encoder is a missing prerequisite. Disk-full requires an explicitly selected, already full disposable volume, with scenario artifacts inside that volume and an observed ENOSPC. The runner never fills a normal filesystem or generates filler files. No physical disk exhaustion is established by injected faults.

Each case gets a fresh UUID directory. Source destinations are templates, never overwritten or implicitly resumed. Recovery uses the suite's own workspace, a new worker, fresh output, unchanged interrupted partial files and a separate uncached oracle. The artifact ceiling is cumulative apparent regular-file bytes across retained suite/case roots, including explicit fault-volume roots. It is cooperative admission, not a filesystem reservation or physical disk quota. The outer terminal report is separately bounded to 16 MiB. Metadata, jobs, cases, raw evidence and directory entries also have explicit bounds. Native I/O and cleanup can outlast a cooperative deadline.

The report distinguishes measured, not_verified, failed and cancelled cases, preserves the planned denominator after aborts and records explicit cleanup evidence. All nine scenarios require successful measured coverage for overall success. `complete` means the plan was processed; it does not mean product acceptance. Exit codes are 0 for successful complete coverage, 2 for complete with unavailable coverage, 1 for failure and 130 for cancellation. The desktop displays exit 2 as unavailable only after strict report validation.

Desktop forced shutdown observes descendants from the owned task thread and pins PID plus native birth identity. It signals only confirmed identities, retires pins only after confirmed exit/reuse and keeps uncertainty after denied/malformed observations. Descendants escaping the CLI process group are included when observed. This is not an OS containment guarantee against an unseen fork race; Windows Job Object containment and native target-platform cleanup remain unverified.

GPU/weights/representative footage, actual long/50-clip workloads, physical fault-volume runs, packaged Windows and Resolve host acceptance still require prepared target environments. CPU fixtures do not satisfy those criteria.
