# Persistent sessions and job diagnostics

The launcher now keeps one model loaded through repeated and directory-batch jobs. It still uses the existing CLI settings/parser and full-rate processing core. Low-VRAM overrides are reset between jobs, and clip-local RVM/hint/window state is recreated each time. Sessions reject concurrent requests; use distinct sessions/output directories for isolation. Failed model execution discards the model before a subsequent job; preflight errors do not discard an untouched healthy model. Normal exit releases it. No output/cache reuse or speedup is claimed from CPU lifecycle tests.

For a single job:

```sh
python infer.py --input /local/dev/clip.mp4 --output_dir out_job --diagnostics out_job/job.json
python infer.py --input /local/dev/clip.mp4 --output_dir out_profile --diagnostics out_profile/job.json --profile_stages
```

Ordinary diagnostics report model-load time, processing time including decode/render/output finalization, first CPU-rendered batch latency (excluding model load), frame counts, source metadata, requested/effective settings and actual per-output codec/precision contracts. `end_to_end_fps` means submitted frames divided by successful processing seconds; startup/load are separate. Failed/cancelled jobs never get a successful FPS. Outputs listed in failed records may be partial. First rendered batch is not first encoded playback. No live/causal, model-only throughput, per-frame latency percentile, confidence or quality claim is implied.

`model_initialization_seconds` includes lazy imports/device setup and model loading; `model_load_seconds` isolates the default loader call plus CUDA completion after imports. A reused model reports zero for both. Missing prerequisites or injected test loaders without an actual model-load measurement leave load seconds null. Cold-process total additionally includes interpreter startup and shutdown; it is not added to these nested durations.

`--profile_stages` adds synchronized wall-clock stage scopes and peak allocated/reserved VRAM after model loading. Stage 1 in normal mode includes streamed Stage 2/render/submit; scopes can nest and must not be summed. Stage 1 also includes per-window foundation extraction. Writer finalization uses host wall time. Profiling changes throughput and is rejected by the ordinary inference benchmark runner. Normal diagnostics are path/settings records; benchmark reports add content hashes and repository/environment identity.

To run a local headless batch, edit `benchmarks/suites/inference-batch.example.json`, then:

```sh
python -m corridorkey.batch benchmarks/suites/inference-batch.example.json --report benchmark-results/batch.json
```

Schema 1 requires `kind: inference_batch`, optional shared `weights`, and a nonempty `jobs` list. Each job requires `input` and distinct `output_dir`; optional `settings` maps to `InferenceConfig`, `writer` maps to `WriterConfig`, with `hint_first_frame`, `hint_video`, `profile_stages`. Paths are relative to the manifest. Settings use core defaults (80/40, auto off); explicit `auto_vram: true` enables automatic selection. Output aliases `raw_fg`/`foreground` map to `fg`. Unknown fields, invalid types/ranges, duplicate output directories and absent sources/hints are rejected. The report is replaced atomically after each completed job and on failure; the batch stops at the first failure and retains prior records. KeyboardInterrupt and cooperative token cancellation record cancelled jobs and stop the batch. A token shared with a later batch job remains cancelled until the caller creates a new token.

API users create `JobRequest.create(...)`, call `InferenceSession.run(request)`, and use `JobResult.to_dict()` for a fresh copy. One session's weights cannot be changed through the CLI. No new runtime dependencies or interface-specific inference implementation were introduced. GPU parity, repeated-job memory growth and real model load savings still require CUDA/weights/local footage.


Job-local IID feature reuse and automatic selection-budget controls are described in `feature-cache.md`. Session persistence does not carry feature caches or recurrent state between jobs. Diagnostics include the effective RAM/disk cache policy, byte-accounting and actual foundation frame counts. A RAM cache requested with auto memory selection is explicitly disabled (with a warning) when auto selects the existing low-VRAM disk-prepass path.


API cancellation and progress use the same shared engine:

```python
from corridorkey.control import CancellationToken

token = CancellationToken()
# A UI/controller thread may call token.cancel("user cancelled").
result = session.run(request, cancellation=token, progress=lambda event: print(
    event.stage, event.completed, event.total))
```

Progress observers run synchronously on the inference thread and must return promptly. Observer exceptions fail the job. Counts are stage-local work units, not an overall percentage; model load is checked before/after, prepasses between chunks, Stage 1 between native windows, Stage 2 between batches, and writer waits repeatedly. Completion is reported only after writer finalization and frame-count checks. Cancellation raises `CancelledError`, records cancelled status with null successful FPS and discards a model touched by the job; a pre-cancelled request loads nothing. Control tokens/callbacks are not serialized into diagnostic JSON.

Encoder pipes use unbuffered short-write handling. Queue and finalization waits default to 60 seconds (`--writer_timeout_seconds` / batch `writer.timeout_seconds`); stalled encoders are terminated then killed if necessary, and reaped. This is a bounded wait policy, not a total job deadline. Native decode, storage, model loading and GPU kernels can finish their current call before cancellation takes effect. A stuck in-process EXR callback can outlive the deadline as a daemon thread and leaves a diagnostic note; Python cannot safely kill that native call. Partial output is retained, never described as a completed render.

Feature prepasses and window processing explicitly flush/close owned writable maps, close read-only maps, wait for the prefetch executor and close its source generator on success/error/cancellation. Prefetched prepasses reject noncontiguous, empty, overflowing or truncated chunks. Cleanup attempts every owned resource and preserves the processing exception. CPU tests inspect actual mmap closure, real child reaping and source-finally execution; actual Windows and CUDA cancellation remain external validation.

Optional `settings.corrections_path` is snapshotted as an absolute local path; a verified immutable document is loaded at job start. Results retain its actual identity, revision, source/asset references and supported policies. Malformed/hash-mismatched corrections fail preflight before loading or touching a healthy retained model. Raster/probe incompatibilities fail before processing stages. Correction hashing appears in `correction_preflight_seconds` and session elapsed/batch wall time; processing FPS excludes that preflight. See `corrections.md` for directional/static/hint semantics, strength equations and whole-shot invalidation.
