# Shared engine worker and render queue

`python -m corridorkey.worker` runs the existing engine in an owned local child process. `WorkerClient` supplies the process boundary for a desktop frontend; it does not introduce another inference implementation. Production remains the existing CUDA runtime with local assets. A ready handshake means the protocol is ready, not that CUDA, model assets or an encoder have passed validation.

## Working commands

Use the same inference-batch manifest supported by `corridorkey.batch`:

```sh
python -m corridorkey.worker run \
  --batch benchmarks/suites/inference-batch.example.json \
  --report benchmark-results/worker-report.json

# Existing recoverable EXR controller, now executed in the child:
python -m corridorkey.worker run \
  --batch benchmarks/suites/inference-batch.example.json \
  --report benchmark-results/worker-recovery-report.json \
  --workspace benchmark-results/recovery --max-cache-bytes 1073741824 \
  --durable-foundations --durable-rvm --durable-hann
# Repeat the recovery command with --resume after interruption.
```

Replace example asset/input paths first. Recovery retains its EXR/explicit scheduling requirements, fresh destinations, content/model identity guards, ownership and journal rules. Recovery flags require `--workspace`. Negative byte budgets and invalid queue capacities reject before child startup. No footage/model download, upload, network listener or alternate CPU/MPS inference backend is selected.

The CLI submits one operation at a time and stops on the first failure, retaining prior results. The library supports a bounded waiting queue for the desktop. Reports persist startup/progress (rate limited), authoritative results/failures, worker generation/PID, cleanup status, a bounded engine-log tail and actual operation wall time. These times are not an uncached GPU benchmark. Recovery returns a compact reference to its full local report, preserving historical byte-replay provenance separately.

## Python client

```python
from corridorkey.session import JobRequest
from corridorkey.worker_client import WorkerClient

with WorkerClient(capacity=4, weights=None) as worker:
    job_id = worker.submit(JobRequest.create('/absolute/input', '/absolute/output'))
    while True:
        event = worker.next_event(timeout=10)
        if event['request_id'] == job_id and event['kind'] in {
            'result', 'failed', 'cancelled', 'rejected', 'lost'}:
            break
    # worker.cancel(job_id) requests cooperative cancellation while a job runs.
```

A timeout is an observation timeout, not evidence of process exit. The batch CLI records waiting state and observes the same child again. Library callers should likewise re-poll or inspect `worker.process.poll()`. `restart()` requires the old leader to be confirmed terminal, creates a new generation, and never resubmits old requests. Recovery is explicit with `--resume`; lost work can have partial output and must not be treated as a completed render. Normal inference output semantics remain those of the existing session; use recovery when preserved fresh destinations are required.

Default `close()` requests cancellation of active/pending work, allows a cooperative deadline, then force-stops the entire owned process group/tree. It waits for the leader and transport threads. A false return or `cleanup_error` means shutdown was not verified clean. Post-kill inspection may be denied by the sandbox/OS; this uncertainty is retained, not presented as successful cleanup. POSIX ignoring-descendant execution is tested through its stopped heartbeat and the reaped leader; a process-table/reparented-group inspection is not always available here. Windows hard-crash descendant containment still needs Job Object/target-platform validation; `taskkill /T /F` is only a best-effort live-tree fallback.

## Ownership and model lifetime

One execution thread owns one lazy `InferenceSession`; normal jobs reuse its model sequentially. IPC input/control and output/log handling remain independent of the model call. At most one job executes; default4 (allowed1..64) wait. Pending cancellation removes a job without loading/executing it. Active cancellation uses existing `CancellationToken`/`JobControl` checkpoints. Cancellation is a request; completion can win if it occurs after the engine has already completed. Native GPU/I/O calls remain noninterruptible until they return; the supervisor provides force termination after the cooperative deadline.

Recovery requests use the existing `run_recoverable` batch/session lifetime. A resident normal session is closed before switching to a recovery batch; each recovery request gets fresh batch model context. This prevents newly fingerprinted disk assets being attached to old resident weights. Normal jobs retain fixed worker-startup weights; keep assets stable and explicitly start a new worker when changing them. No unvalidated cross-batch model/context reuse is added. Each independent worker owns its own model/queue; shared recovery workspaces retain their exclusive SQLite ownership.

Input/configuration data is snapshotted before queueing. Shared job settings/hints/profile flags use `JobRequest` validation, preserving all implemented options. Paths are explicit absolute local paths; temp paths are canonicalized before transfer. A recovery request references a batch file read by the child at execution; keep queued manifests and source/assets stable. These are file fingerprints and validated requests, not a globally atomic filesystem snapshot.

## Version1 wire contract

`serve --capacity N [--weights PATH]` is the low-level child entrypoint. Run it as an owned, supervised process; desktop code should use `WorkerClient`. It consumes one complete UTF-8 JSON object per line on stdin. Python **and native fd1** engine logs are redirected to stderr; a private duplicated descriptor carries only protocol frames. Requests never contain executable module names, pickles or media tensors.

Request envelope has exactly `schema_version` (integer1), `request_id` (1..64 ASCII letters/digits/underscore/hyphen), `action` and object `payload`. Frame limit is4 MiB, including newline. Duplicate fields, nonfinite numbers, incomplete/oversized lines, invalid types/versions and unknown fields/actions reject. Embedded shared settings JSON also uses strict duplicate/finite parsing. Malformed job settings become protocol-domain errors.

| Action | Payload/effect |
|---|---|
| `render` | `job`: complete serialized shared `JobRequest`; executes through resident session. |
| `recover` | Absolute `batch`, `workspace`, nonnegative integer `max_cache_bytes`; optional exact booleans `resume`, `durable_foundations`, `durable_rvm`, `durable_hann`. Executes shared recovery controller with fresh batch context. |
| `cancel` | `target` request ID and optional reason (at most2000 characters). Acknowledgement reports attempt state; target receives its own authoritative terminal event. |
| `status` | Empty payload; reports active/pending IDs, capacity, closing state and history exhaustion. |
| `shutdown` | Optional `drain` boolean. Default cancels; true drains accepted work before closing. Wait for `stopped`/actual process exit when using the low-level draining operation. |

Wire events contain exactly version, random generation, contiguous sequence, kind, request ID (or None) and object payload. Sequence is assigned on actual send, so coalesced progress creates no gaps. Kinds: `ready`, `accepted`, `started`, `progress`, `result`, `failed`, `cancelled`, `rejected`, `status`, `stopping`, `stopped`. The parent additionally synthesizes `lost` on crash/protocol failure, without inventing a job result. Every accepted job has a terminal outcome or explicit connection-loss uncertainty. Retained callbacks cannot publish progress after terminal completion.

Request IDs are never reused within a generation. The bounded4096-ID admission history returns `restart_required` when exhausted; controls/shutdown remain available. Queue-full rejection does not run a model. Failures/cancellation discard contaminated session models using the existing lifecycle, and later accepted jobs can continue. Manual POSIX SIGINT cooperatively cancels/stops the low-level server, flushes terminal records and returns130; a valid stopped handshake is recognized before classifying EOF.

## Bounded backpressure and errors

The client queues outbound frames (8-frame limit) and writes on a dedicated thread, so a peer that stops reading cannot block the submitter on pipe I/O. Server input framing has a bounded8-frame queue. Server/client critical event buffers are bounded by count/bytes (128 events/32 MiB); current progress coalesces instead of displacing terminal records. The parent drains stderr into a64×4096-byte tail. Bounds cover IPC/history only, not all serialization buffers, engine/model RAM/VRAM, OS pipes, source data, temporary files or total process-tree resources.

Critical backpressure, invalid generation/sequence, broken pipes and frame limits fail explicitly. No terminal record is silently dropped and no partial output is relabelled complete. If a valid engine result exceeds the4 MiB frame contract, a small `ResultEncodingError` reports that outputs may already be complete; inspect local outputs/diagnostics before deciding how to recover. Recovery's full report remains on disk. Cleanup uncertainty/failure prevents a CLI batch from reporting successful clean shutdown. Repeated process restarts or concurrent workers are not a measured memory/performance optimization.

## Actual evidence and remaining acceptance

Tests exercise actual owned children, native EXR writes and shared sessions with an explicit test-only deterministic engine. They verify model-session reuse, bounded queues/history, queued/active cancellation, independent status controls, Python/native log isolation, coalescing, partial writes/fragmented Unicode, strict JSON, nonreading peer responsiveness, same-handle timeout observation, actual hard exit78/new-generation restart, SIGINT, recovery byte replay/resume and a signal-ignoring descendant during force-close. The production child also actually rejects unavailable CUDA, with a failed record and no fixture fallback.

`benchmark-results/m17-worker-generated-summary.json` records a generated2-job native EXR fixture with1 load and second-session reuse, recovery render/verified-byte-restore followed by verified-existing resume, child exit codes/clean stops, and the real production CLI's unsupported-CUDA exit1. These are actual controller/export/lifecycle counts, **not GPU inference throughput, real weight-load timings, numerical parity or professional visual-quality measurements**. Tests/fixture code are not a production backend or public CLI option.

GPU/weights/permitted footage, real encoder stress, pretrained cancellation/performance/quality, whole-process resource telemetry, Windows containment/packaging and Resolve SDK/host acceptance remain NOT VERIFIED. The next product increment is a usable desktop shell using this client, followed by previews/composites/range/A-B, persistent queue/settings/corrections/cache controls and a developer benchmark panel backed by the existing runner. Direct seek and Stage2/writer checkpoints remain open; no OFX integration is claimed.


Explicit status now includes [resource snapshots](resource-snapshots.md): current worker lifetime peak RSS where platform units are supported, and already-initialized current-device PyTorch allocator counters. Querying never imports Torch or creates CUDA context. Unsupported/partial/native-probe failures retain null metrics and an explanation. This additive status metadata is unsynchronized, not whole-machine/GPU usage and not a processing budget.
