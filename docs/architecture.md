# Shared engine and plugin architecture

## Entry points and ownership

`infer.py` validates `InferenceConfig`, captures an immutable `JobRequest`, and
calls `InferenceSession.run`. The interactive launcher and batch CLI use the same
engine. The session lazily loads a model and calls `runtime.run_inference`; it
owns model reuse, failure disposal and sequential execution. Public `CorridorKeyV2`
and configuration import aliases remain supported.

The Qt desktop sends validated JSON requests through the existing worker boundary.
It does not have a second inference implementation. Background operations publish
bounded progress/results; source/request/generation fences reject stale completions.
A completed worker does not imply that its queued terminal records were consumed.
Replacement must consume records and retain cleanup-uncertainty safeguards.

## Processing pipeline

1. Resolve local model/foundation assets, checkpoint configuration and selected
   source range. Missing assets or unsupported backends fail without silent fallback.
2. Decode video or naturally ordered frames, preserve native geometry with explicit
   padding, and apply the declared colour/alpha contract. Display-coded video uses
   the legacy SDR path; scene-linear EXR input is explicit. Foundations receive SDR
   proxies rather than unclipped HDR values.
3. Run RVM chronologically over the selected timeline. Recurrent state starts fresh
   at the selected range/shot boundaries; this is not an automatic original-frame-zero pre-roll.
4. Run C-RADIO/MoGe and Stage1 with original RGB, hints, corrections and window context.
   Normal and low-VRAM scheduling retain their documented numeric/precision policies.
5. Blend overlapping Hann batons, then run complete Stage2 groups for alpha and
   foreground reconstruction. Current corrections, refinement and finishing remain
   ordered operations, not new pretrained input channels.
6. Write configured outputs through bounded FFmpeg/EXR transports. Finalization and
   expected coverage must succeed before ordinary export completion is reported.

Source frames/mattes must retain absolute/local mapping. Resets, cut boundaries,
carry hints and full temporal groups are dependencies, not interchangeable quality settings.
See [colour](colour.md), [sessions](sessions.md), [worker](worker.md) and
[Quick Key](quick-key.md) for effective behavior and explicit approximations.

## Corrections, previews and caches

Authored masks, generated tracking/guidance and their provenance stay distinct.
Editing uses the existing editor/service/history and invalidates dependent work;
unknown, stale or corrupt inputs are refused. Diagnostic overlays are display-only
heuristics, not confidence or occlusion ground truth.

Progressive sequence previews retain the complete captured timeline, hints,
corrections and Stage2 grouping, but capture only the requested frame into an owned
workspace. Their distinct preview completion cannot be promoted into a full export.
Each selection currently processes the full timeline; draft scale remains explicit.
See [conditioned previews](conditioned-progressive-preview.md).

Opt-in intermediate reuse qualifies actual complete input/state/group identities.
Current correction/carry reconstruction and full-current admission/output provenance
remain authoritative. IID grouping, temporal schedules and model/backend/precision
boundaries cannot be weakened merely to increase hits. Old render attempts remain
separate. See [Stage2 recovery](stage2-recovery.md),
[upstream finishing reuse](upstream-finishing-reuse.md) and [recovery](recovery.md).

## Cancellation and observability

Sessions own one model and reject conflicting operations. Failed model execution
invalidates potentially contaminated state. Cleanup attempts all owned callbacks;
uncertain cleanup does not silently create a replacement owner. Cooperative
cancellation cannot forcibly interrupt every native codec/kernel call.

Optional instrumentation reports actual available observations with requested and
effective settings. Inclusive/overlapping stages cannot be summed into an exclusive
frame budget. Missing GPU/resource measurements stay unavailable; no CPU fixture
certifies neural throughput or visual quality. [Diagnostics](realtime-feasibility.md).

## Native OpenFX boundary

`native/ofx/` contains C++17/CMake plugin and harness sources. The desktop prepares
and seals completed EXR caches through shared admission; the native adapter reads
that prepared contract, not arbitrary live model state. OpenFX headers, third-party
licences and provenance are build inputs and must remain distributed as required.

[Cache preparation](ofx-cache.md), [upstream capture](ofx-upstream-capture.md) and
[Resolve](resolve.md) describe scope and limitations. SDK/harness compilation is not
actual Resolve host acceptance. Clean Windows/NVIDIA execution and representative
footage/managed-display evaluation require target validation.
