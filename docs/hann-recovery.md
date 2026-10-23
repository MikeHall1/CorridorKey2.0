# Hann recovery by chronological event replay

`python -m corridorkey.recovery run --durable-hann` optionally persists Stage1 window events. It has a working effect in the existing shared engine and defaults off. Combine it independently with `--durable-foundations` and `--durable-rvm`; repeat the same flags with `--resume`. See `recovery.md` for batch/workspace/byte-budget commands. Production still requires local CUDA assets, EXR and explicit scheduling. Normal inference/interactive/batch defaults are retained.

## Consistent boundary

The exported Stage1 recurrent sites are disabled by the existing loader. Its per-window model result, given exact RGB/hint/quality/foundation inputs, is therefore distinct from Hann's carried hints and live overlapping sums. Hann state also includes accumulated weights, pending streaming source frames and previous segment. A cursor alone cannot authorize skipping this state.

Recovery seals a raw model event and replays events in the original schedule, running the existing conditioning/carry/blending/finalization code. The same source readers refill pending frames. Every earlier window is replayed or recomputed before a later event is eligible. This rebuilds consistent live state and finalized baton ranges without persisting a partial output or trusting stale accumulators. It is an event-log checkpoint strategy; there is no direct arbitrary-window snapshot seek.

The immutable `HannWindowIdentity` version1 binds:

- Complete parent job identity and enabled recovery policy.
- Actual whole-schedule digest (windows, chunk/stride, target anchors, causal context, streaming/disk mode, context radius and segments).
- Window index/start/stop/next boundary, selected-timeline segment, carry policy and baton width.
- Incoming live replay-state digest: every carried hint, accumulator, weight, pending source frame and previous segment.
- Exact dtype/shape/bits of actual Stage1 RGB, hints, quality values, MoGe, optional C-RADIO and optional RVM tensors.

Indices/types, RGB/hint geometry, foundation grid, complete argument inventory and hashes are strict. State dictionaries hash ordered frame/value pairs, not insertion order or mutable object addresses. Actual input tensors are finite; input identity is computed before invoking Stage1. Corrections bind through full parent provenance, actual conditioned input tensors and reconstructed carry state; reset segmentation is part of the schedule. All invalidation is conservative.

## Payload and replay

One sealed safetensors file contains the raw `(1,T,d_16,H/16,W/16)` baton and, if carry is enabled, the used raw auxiliary alpha `(1,T,1,H/2,W/2)`. The remaining auxiliary foreground channels and bottleneck are unused by this boundary and are not persisted. Exact producer float32/bfloat16/float16 dtype/bits are retained; no latent clipping, quantization, activation or interpolation is introduced. Signed/large finite values remain valid. Original float32 carry interpolation/clamping, authored refinement, float32 weighted accumulation and float16 finalization run after every event.

The adapter shares actual loaded-model/context binding, exclusive store ownership, recognized-file-only cleanup, staged atomic completion and the logical sealed-entry/LRU byte budget with IID/RVM/output recovery. Loading validates manifest and immutable payload bytes, exact tensor inventory/shapes/dtypes/finite values and canonical producer provenance before deserialization results escape. No pickle or live file mapping is used. Invalid owned entries are discarded/recomputed; unknown content blocks deletion. Active Stage1 recurrent sites are rejected before lookup, and outgoing recurrent state is rejected at admission.

Publication certifies the raw event, not a completed Hann pass, Stage2 output or writer. If a later correction/blend/render/cancel/failure occurs, the event can remain valid while the job remains incomplete. Resume uses a fresh session/output destination and reconstructs prior state. A missing/corrupt middle event computes the same complete window; later events hit only if the actual rebuilt state and model inputs match. Both normal streaming and low-VRAM baton-file paths use this loop, including sparse/fallback contexts and authored resets.

## Counters and verification scope

`diagnostics.durable_hann` records window hits/misses/stores, Stage1 frame instances reused, corruption, bytes loaded and budget rejection. `execution.stage1_frame_evaluations` and `stage1_windows_evaluated` count actual calls only. Overlapping frame instances are not unique timeline frames. Foundation, Stage2, motion and writer work retain their own counters; output replay origin timings remain historical.

CPU regressions execute the real runtime/source/memmap/EXR/controller with deterministic neural substitutes. They verify 15-frame overlapping carry/weight reconstruction after interruption in both memory paths; zero-call repeated replay; missing/corrupt middle recomputation; full/offline/hold/causal/cut/correction/reset/fallback native output equality; exact dtypes and signed/large values; invalid payload/provenance; model binding/recurrent guards; budget/cancel/disk-full cleanup; and actual hard process exit77/reopen. These prove local cache/control/arithmetic behavior, not pretrained model/GPU acceptance.

Source decode, foundation computation unless independently cached, exact input hashing, CPU/GPU copies and Hann replay arithmetic still run. Stage2, sparse propagation and complete output exports restart. Transient hash/serialization/read/copy buffers, temporary files, journals/reports/render copies, physical allocation and total process memory lie outside the sealed-entry budget. No throughput or memory improvement is claimed from call counts alone.

Pretrained CUDA parity/repeatability/tolerances, transfer costs, end-to-end/cache-aware performance, reference visual quality, Windows directory flush/power-loss and Resolve SDK/host remain **NOT VERIFIED**. Direct range seeking, Stage2/writer checkpoints and desktop/OFX random-access serving remain open. The next product step is an isolated worker/queue calling this shared recoverable engine, followed by desktop previews/editor; required hardware/quality/host acceptance remains part of the full brief.
