# Recurrent checkpoint contract

The shared runtime now supports optional exact-call RVM chunk checkpoints through `python -m corridorkey.recovery run --durable-rvm`. See `recovery.md` for concrete commands, budgets and ownership. The option has a working effect in the existing chronological prepass and defaults off. It can be combined with `--durable-foundations`; both are shared by normal/low-VRAM execution. No pretrained inference, backend or model algorithm is replaced.

## State inventory from the actual runtime

| Component | State and current recovery behavior |
|---|---|
| C-RADIO/MoGe | Independent normalized float32 frames; optional durable IID entries. |
| RVM encoder | Batched over the exact current call's frames. Reuse never changes miss call grouping. |
| RVM recurrent core | One `(1, 1 + grid_h * grid_w, D)` tensor, including CLS. Returned state feeds the next chronological chunk. Exact floating dtype/bits are persisted with the chunk's features. |
| RVM feature prepass | Normalizes float32 using the loaded stats, then stores float16, preserving the existing behavior. Complete selected-timeline memmap is reconstructed on restart. |
| Stage1 recurrent sites | Existing exported-runtime loader explicitly sets `stage1_recurrent_64_sites=()`; no new state is invented or enabled. |
| Hann windows | Carried hints, live float32 accumulators/weights, previous segment and pending Stage2 source frames remain RAM-local. Repeated overlapping window calls and blending are deterministic scheduling, not a resumable checkpoint yet. |
| Stage2/writers | Output batches, sparse anchor maps/fallback state and writer completion are not resumed mid-job. A successful output inventory/journal remains the authority for completed whole-job replay. |

`RVMVideoSiamMAE.forward` tokenizes/encodes all frames in a call, loops its gated transformer over T, drops CLS for returned feature maps and retains CLS in returned state. The exported runtime prepass independently resets at clip/shot boundaries. A selected video range starts its own timeline; checkpoint indices are relative to that selection, with parent config/source identity preserving the original range.

## Version 1 identity and payload

`RvmChunkIdentity` is immutable canonical JSON with exact fields: schema version1, kind `rvm_chronological_chunk`, parent job SHA256, selected shot start, start/stop, original proxy digest/dtype/TCHW shape, incoming-state digest or None, channel width and grid. Ordered integer bounds, positive dimensions, supported input precision and hashes are strict. At shot start the incoming state must be None; continuation requires its exact dtype/shape/bit digest. Full parent provenance binds source/hints/corrections/settings, assets/statistics, implementation, environment and opt-in policy. Keys invalidate conservatively; this is not a minimal mathematical dependency key.

Each bundle has exactly one safetensors file with `features` and `state`. Feature shape is `(stop-start, D, grid_h, grid_w)`, dtype float16. State shape includes CLS and its producer dtype is preserved exactly (float32, bfloat16 or float16). Both must be finite. Canonical provenance identifies the exact stage, parent, feature/state precision and shape, and explicitly says Hann/output resume is false. Missing features/state, bad provenance, nonfinite values and incompatible geometry/dtypes reject the complete entry.

Publication uses owned private staging, checked inventory, atomic rename, completion metadata hashes and existing file/POSIX directory sync. Loading hashes the immutable bytes passed to safetensors; no pickle, path-dependent mmap, asynchronous state alias or partially paired state escapes. Arrays/tensors are independent of subsequent cache eviction. SQLite exclusive ownership, recognized-file-only deletion and cache budgets are inherited from the verified bundle store.

## Resume and invalidation

The fresh session binds actual loaded paths and rechecks current dependencies before adapter use. The prepass decodes each source chunk, applies the same reset segmentation and obtains an identity **before** invoking the model, so a producer mutating its input-state object cannot relabel the outgoing state as incoming. A hit restores both features and exact state. A miss invokes the same complete chronological RVM call and seals both results before the progress checkpoint. Later entries can be reused only if their actual incoming-state digest matches. If recomputation differs numerically, state hashing causes safe misses rather than accepting an incompatible suffix.

All selected frames still decode and all feature ranges are reconstructed. No direct seek/random/reverse access or warm-model persistence across a crash is claimed. Existing/partial outputs remain preserved and a fresh export destination is used. Full job success still requires a complete output inventory, independent of prepass entries. Tiny/zero budgets may reject states, and output publication may evict them. Managed sealed bytes are bounded; serialization/read buffers, GPU/CPU copies, transient staging, journals/reports and render copies are outside the limit. No process-wide memory or total-disk cap is implied.

## Verification and next contract

Tests execute the real prepass/session/controller with deterministic CPU recurrence, plus the actual RVM classes configured with tiny random weights. They verify interruption/reopen, missing-middle recomputation from restored state, verified suffix reuse, resets, dtype/bit equality, identical float16 files, cancellation, disk-full/capacity behavior, context binding, invalid provenance/post-lookup mutation rejection and actual hard exit76/state restore. They are local correctness evidence, not pretrained model quality or GPU performance measurements.

Pretrained CUDA parity across restart, baseline repeatability/tolerances, actual transfer/memory costs and throughput, reference visual quality, Windows directory flush/power-loss and Resolve SDK/host remain **NOT VERIFIED**. Keep the feature opt-in until those runs are completed with real weights and permitted footage.

Full-range recovery next requires a versioned Hann checkpoint that binds window cursor/schedule, carried hints, live accumulators/weights, source/feature identity and pending outputs as a consistent boundary. Sparse contexts, corrections, resets, fallback anchors and writer completion need explicit dependency/invalidation handling. A state file alone cannot authorize skipping preceding work or calling an unfinished render complete. Desktop/OFX random access also needs a verified shot index/cache serving policy and per-instance isolation; this RVM prepass is one implemented component, not a complete host solution.


Milestone16 implements the Hann recovery increment through **verified prefix event replay** (`--durable-hann`). Raw Stage1 events plus original source inputs reconstruct carried hints, live sums/weights and pending frames in the same loop. Schedule/cursor and incoming full replay state are hashed into each identity. It does not serialize a mutable mid-loop snapshot or enable direct range seek, and Stage2/output still restart. Earlier future Hann notes above describe the remaining direct-seek/full-output boundary. See `hann-recovery.md` for actual contracts, verification and limits.
