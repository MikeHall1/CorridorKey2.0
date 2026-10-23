# Verified EXR replay and interrupted-batch recovery

`python -m corridorkey.recovery` is an opt-in batch controller on the existing `InferenceSession`. Normal inference/interactive/batch defaults are unchanged. The first implementation preserves completed whole-job EXR bytes and journals attempts across process exits. An unfinished job rebuilds its selected timeline in order. Optional durable IID frames, exact chronological RVM chunk/state checkpoints, Hann event replay and exact Stage2 prediction batches avoid verified neural work; upstream reconstruction, rendering and fresh writers still run. Direct range seeking remains future work.

## Commands

Prepare permitted local inputs/weights in the existing inference-batch format. Every job must explicitly use EXR output and `auto_vram:false`; the batch dataclass default is already false. Choose explicit temporal/chunk settings. This avoids replaying an uncomputed automatic memory-dependent profile. No output mode, scheduling, precision or backend is silently changed.

```sh
python -m corridorkey.recovery run \
  --batch benchmarks/suites/inference-batch.example.json \
  --workspace benchmark-results/recovery \
  --max-cache-bytes 1073741824
# Same batch after interruption, or to verify/retain completed outputs:
python -m corridorkey.recovery run \
  --batch benchmarks/suites/inference-batch.example.json \
  --workspace benchmark-results/recovery \
  --max-cache-bytes 1073741824 --resume
# Read-only metadata listing; no CUDA/model loading or payload verification:
python -m corridorkey.recovery status --workspace benchmark-results/recovery
```

Example batch paths must be replaced before use; no model/source assets are bundled or downloaded. Production `run` requires CUDA and local runtime/foundation/stat assets, GPU/driver/dependency context, and the shared model loader. There is no CPU/MPS inference fallback. `status` explicitly reports payload/backend/outputs NOT VERIFIED. Python library APIs support the CPU storage/export fixtures used in tests; those are neural substitutes, not an alternate model backend.

## Identity and model lifetime

`recovery_identity.job_identity` constructs immutable canonical identities from source/hint content and ordered frame names, all validated config/writer/profiling settings, correction source/manifest/canonical revision and verified asset hashes, all runtime/foundation/stat files, shared/third-party implementation Python files and explicit environment. Production environment includes Python/platform/dependency versions, CUDA/GPU/capability/UUID where exposed, driver/cuDNN, precision and relevant deterministic/TF32/matmul flags. Inputs and settings paths participate conservatively; render destination does not. Changing optional outputs or writer settings can therefore invalidate an otherwise reusable result. It is a completed-job identity, not a proven minimal IID feature key.

Source/assets/implementation are fingerprinted before and after operations; observed changes reject completion/replay eligibility. Actual session-loaded asset paths must match the fingerprinted asset set (required for production CUDA origins), including default legacy/safetensors resolution. Model asset/backend/implementation signatures stay pinned across the batch. Changes stop the batch and require resume with a fresh session, so an old resident model cannot publish data under new weight hashes. Inputs and installed dependencies must remain stable; these checks do not create globally atomic snapshots or independently attest locally modified installed packages.

## Storage, integrity and ownership

An empty workspace gets an ownership marker and immutable content-key entry directories. Existing nonowned/nonempty directories are rejected. Source/hint/model trees and render destinations must be separate from the workspace. A SQLite exclusive lock enforces one controller owner; actual process-exit tests demonstrate OS lock release without deleting or guessing stale PID lock files. The journal is separate from that lock transaction and commits each attempt independently.

Each entry stores regular flat payloads, canonical identity, origin result, size/hash inventory and a completion-record digest in its owned marker. Payload hashes/lengths, full inventory, identity and completion metadata are checked before reuse. Artifact paths reject escapes, reserved names, file-parent conflicts and case/Unicode aliases that could collide on supported filesystems. No pickle or arbitrary Python objects are deserialized; optional RVM checkpoints load explicit floating tensors from verified safetensors bytes. Private staging has workspace-specific ownership in its exclusive directory name, so interrupted owner/completion record writes can be cleaned. Unexpected files or symlinks prevent automatic deletion. Eviction removes only recognized tool-owned derived files; user/source/old render outputs are untouched.

Publication seals a private staged bundle only after copies/checks complete, then renames it atomically into the entry namespace. Restores create an exclusive fresh destination, copy/check bytes and write a completion marker last; partial copies remain untrusted. File sync and POSIX directory sync are requested. Windows directory flush/power-loss and target-platform behavior are NOT VERIFIED; process-crash recovery is locally tested. Durable here includes process restarts, not an independently proven guarantee against every storage/power failure.

`max-cache-bytes` bounds logical managed sealed payload+record bytes. Eviction occurs before allocating another entry; oversized entries are not admitted, while a completed journaled render remains usable. Zero disables stored bundles but retains journal/output verification. Filesystem allocation overhead, atomic metadata temporary copies, source/weights, render copies, reports/journal history and model/RAM/VRAM are outside this cache limit. This is not a total disk/process budget. Hashing all inputs/assets/code can be expensive and is included in controller wall time; no cache speedup is claimed without comparable measurements.

## Recovery semantics and reports

Per-request journals have checksummed envelopes and owned request identities. A successful attempt requires completed engine status, a complete native EXR frame inventory and matching hashes. A previous `running` attempt becomes `interrupted` only after exclusive ownership is acquired; partial files alone never count as completion. An unchanged completed output is verified in place. Otherwise a verified bundle can restore it, or the whole job renders again. Existing requested output directories are preserved; a new sibling `.recovery-<uuid>` is chosen and recorded rather than overwriting old/partial/user data. An invalid known cache bundle is discarded only through the owned-content rules and recomputed. Unknown ownership/content requires inspection rather than a broad cleanup.

Reports distinguish `render`, `verified_cache_restore` and `verified_existing`, with actual output paths, key, cache notes, current operation wall seconds and `new_session_run`. `origin_record` preserves original job timings and settings as historical provenance. Replay is byte restoration, not new neural inference or newly measured FPS. Reports and attempt journals persist under the workspace; failed/cancelled operations preserve original exceptions/notes and partial output paths. Repeated requests require explicit `--resume`; distinct batches/destinations have separate journals but may share compatible bundles. All content remains local.

## Evidence and remaining work

Tests exercise canonical invalidation/detachment, native shared-session/EXR writer reuse and exact byte equality, completed-job resume without model loading, corrupt/stale data, model path/content changes, zero/oversized budgets, cancellation, injected disk-full errors, actual hard process exits, exclusive owners and unknown/user-data protection. Generated CPU fixture `benchmark-results/recovery-m13-generated-summary.json` observed one session load and one export job for two identical requests (second restored), then two verified-existing jobs on resume with equal bytes. This is actual controller/storage work counting, not neural/GPU throughput, visual quality or packaged/Windows/Resolve acceptance.

Recovery remains in progress for durable intermediate IID reuse, validated recurrent checkpoints/range continuation, richer repair/retention/resource policies and desktop queue/cache UX. GPU parity/performance, real assets/held-out quality, target-platform crash/export and legal/package/Resolve validation remain external. Do not expose arbitrary frame-resume controls without implemented state contracts.


## Optional durable foundation frames

Add `--durable-foundations` to both the initial `run` and its `--resume` command to persist independent C-RADIO/MoGe features. It defaults off. The same adapter feeds the existing normal Hann path and low-VRAM disk prepass; no alternate inference engine is introduced. The policy participates in replay identity, so switching it invalidates reuse conservatively. Normal inference/batch/interactive commands retain their existing behavior.

Each frame identity includes the complete parent job identity, native index, original RGB proxy SHA256 bits/dtype/shape, positive channel widths and output grid. Actual bfloat16 proxy bits remain unchanged; float32 and bfloat16 inputs have different keys. Only exact finite normalized float32 CHW features are persisted, including valid signed/large latent values. No precision reduction, RGB conversion, RVM state, baton, hint or stage prediction is stored in these entries. The key is conservative: changing any parent setting can invalidate all frames even when mathematically unnecessary.

The shared session binds the adapter to actual loaded model paths and revalidates dependency identity before its runner can read/write frames. Producer batch membership is recorded. Lookup verifies sealed inventory/checksums, exact dtype/channel/grid and finite values; mapped arrays are copied and closed before eviction, then payload hashes are rechecked. Corrupt recognized entries are discarded and recomputed. Unknown files still prevent deletion. Cache files and source dependencies must remain stable during processing; this is not an atomic snapshot of externally changing files.

Each completed frame seals independently. A later render/cancellation/crash can preserve those frames while the attempt stays incomplete. On resume, a fresh session restarts the selected timeline; valid IID frames can avoid foundation evaluations. With only this flag, RVM recurrence restarts; `--durable-rvm` independently enables the checkpoint path below. Existing/partial render paths remain preserved. Failure reports include the failed session's diagnostics. Frame telemetry distinguishes durable hits/misses/stores/corruption/budget rejection/bytes from RAM counters and actual foundation evaluations. Output replay remains historical byte reuse, never new inference.

Features and completed-output bundles share `max-cache-bytes` and LRU eviction. Owned temporary tensor NPY files are cleaned on failure/reopen; their transient bytes, like other publication temporaries, are outside the sealed-entry budget. Very small budgets can reject frames or evict them while publishing output. NPY loading forbids pickle. Default GPU extraction batches can change when only missing frames are recomputed: **real GPU batch-roundoff, numerical parity, quality, throughput and transfer costs remain NOT VERIFIED**. This feature remains opt-in; CPU deterministic fixtures establish storage/control behavior only.

Milestone14 local fixtures: partial3/5 frames resume with3 hits and2 evaluations; low-VRAM second5-frame prepass makes0 evaluations with identical float32 files; normal7-frame Hann stream second run makes0 foundation evaluations with identical batons; failed2-frame EXR job resumes with1 hit/1 newly computed frame in a fresh session. Actual child exit75 recovers a private tensor stage. These are fixture work counts, not GPU performance measurements. See `tests/test_durable_foundations.py`, `tests/test_foundation_cache.py`, `tests/test_recovery_runner.py` and the checkpoint for final checks.


## Optional chronological RVM checkpoints

Add `--durable-rvm` to both initial `run` and `--resume` to reuse RVM features **together with their outgoing recurrent state**. It defaults off and works independently or alongside `--durable-foundations`. Policies participate in the parent replay identity. Production still requires validated local CUDA assets and EXR/explicit scheduling; no new inference backend is selected. Example:

```sh
python -m corridorkey.recovery run \
  --batch benchmarks/suites/inference-batch.example.json \
  --workspace benchmark-results/recovery \
  --max-cache-bytes 1073741824 --durable-foundations --durable-rvm
# Repeat the same command with --resume after interruption.
```

The adapter is bound to actual session model assets and current dependency identity before the runner. Each typed/versioned key includes complete parent provenance, selected-timeline shot start, exact native chunk start/stop, proxy dtype/shape/bits, incoming state dtype/shape/bits hash (None at shot start), channel width and feature grid. The call grouping is unchanged: an entire chunk hits or the same complete model call is recomputed. At cuts/resets state begins at None and a new chain starts. Missing/corrupt chunks compute from the verified preceding state; later entries can hit only when that actual incoming state matches. No single mutable previous-frame value is treated as a random-access solution.

One atomically sealed safetensors file contains normalized float16 features (the existing prepass precision) and finite outgoing `(1, CLS+grid tokens, channels)` state in its **exact** float32/bfloat16/float16 producer dtype. State is never downcast. Loading verifies bundle inventory/checksums, then verifies the immutable bytes actually deserialized, exact tensor names/shapes/dtypes/finite values and the canonical producer/provenance contract. It forbids pickle and live mapped aliases. Invalid owned entries are discarded/recomputed; unknown files prevent deletion. Features, states, IID frames and output bundles share the same sealed-entry/LRU budget; serialization buffers, transfer copies, temporary files, journals and render copies remain outside it.

Every selected frame is still decoded chronologically and the complete RVM memmap is reconstructed. This avoids RVM computation on hits; it does **not** seek directly into a range, retain a warm model across a crash, or resume Hann blending/carried hints/writer queues. Stage1's exported recurrent sites are disabled by the existing loader, but Hann hint/overlap state is separate and still restarts. Failures remain incomplete until the normal output inventory is complete. Diagnostics report actual RVM frame evaluations plus chunk hits/misses/stores, frames reused, corruption, bytes and budget rejection; origin records on output replay remain historical.

CPU fixtures verify interrupted prepass and fresh-session export recovery, exact state precision, cuts/reset invalidation, missing-middle recomputation with verified suffix reuse, cancellation/disk-full/zero/small budgets, post-lookup mutation, invalid metadata and actual hard exit76 followed by successful state restore. A tiny random-weight fixture executes the actual RVM implementation and verifies identical float16 prepass files on replay and a missing chunk recomputed from restored state. These establish local state/serialization/control behavior. **Pretrained GPU numerical parity, determinism across process restart, performance, visual quality, Windows power-loss and Resolve behavior remain NOT VERIFIED**. See `recurrent-checkpoints.md` for the state inventory and remaining full-range work.


## Optional Hann event replay

Add `--durable-hann` to the initial `run` and its `--resume` command. It defaults off and combines independently with `--durable-foundations` and `--durable-rvm`. All enabled policies participate in the parent identity. The shared session binds the adapter before its normal runner, and active Stage1 recurrent sites are explicitly rejected. The existing exported runtime disables those sites already.

Each entry seals exact raw Stage1 baton and, when carry is enabled, auxiliary-alpha tensors. It preserves producer float32/bfloat16/float16 bits, including signed/large latent values, and applies the original correction/interpolation/clamping/blending after loading. It never stores unused auxiliary foreground channels or the unused bottleneck. Full input tensor precision/shape/bits, complete parent, schedule, window cursor/bounds/segment, next boundary, carry policy and incoming live replay state bind the entry. Replay state covers carried hints, live accumulators/weights, pending Stage2 source frames and previous segment.

Resume walks the actual windows in order, loads verified events or computes missing ones, and runs the same carry/correction/overlap/finalization code. This reconstructs consistent Hann state rather than trusting a cursor or incomplete output file. Missing/corrupt events are recomputed and later events hit only when the actual reconstructed state/inputs match. Both normal streaming and low-VRAM baton-file paths, sparse offline/hold/causal contexts, cuts, authored corrections/resets and neural fallback calls share this boundary. Safetensors payload, exact tensor names/shapes/dtypes/finite values, immutable deserialized-byte hashes and canonical producer provenance are verified before a hit. Model/context/budget/ownership rules remain shared with the other caches.

Source decode, actual foundation inputs/features and Hann replay arithmetic still run. Without the separate foundation/RVM cache flags, their model work still runs too; exact Stage1 feature values are hashed before lookup. All exports restart into a fresh recovery destination; Stage2 predictions can now replay with the separate explicit --durable-stage2 policy (stage2-recovery.md). This is **prefix event reconstruction, not a direct mid-range seek or persistent writer queue**. Hashing/copying raw inputs/events can be costly and uses transient buffers outside the sealed-entry budget. No speedup or process-wide memory cap is claimed.

Actual CPU fixtures: a15-frame/8-frame-window/4-frame-stride stream (three windows) interrupted after the first sealed event resumes with1 hit/2 new Stage1 calls, then fully replays with0 calls. Corrupting the middle event recomputes1 call and matches uninterrupted batons. Both memory paths match. A14-subcase output matrix (full/offline/hold/causal/corrections/full-rate resets/neural fallback in both memory paths) replays with0 Stage1 evaluations and identical36 native EXR outputs per case. A fresh-session export fixture reuses its first sealed event; actual child exit77 releases ownership and restores the event. These are deterministic substitute/CPU work counts, **not pretrained GPU performance or quality evidence**. CUDA parity/performance, Windows power-loss and Resolve behavior remain NOT VERIFIED. See `hann-recovery.md`.


Milestone17 exposes this same controller through `python -m corridorkey.worker run --batch PATH --report PATH --workspace PATH --max-cache-bytes N` with the existing resume/durable flags. Recovery requests use fresh batch sessions; ordinary worker render jobs reuse a session. This avoids stale resident model identity across independently fingerprinted batches. Worker results reference the complete local recovery report, preserving historical origin provenance and all existing verification/fresh-destination semantics. See `worker.md`; no new backend or host integration is introduced.


Milestone21 adds optional desktop **Recovery…** controls and durable queue observations around this same controller. Requests use private canonical batch snapshots, exact managed sealed-cache bytes/durable flags, explicit verify/resume and actual report/metadata inspection. Saved observations/PIDs are not proof of current output/process state, and opening history never resubmits. Current CPU fixtures verify byte restore, verified-existing resume, corrupt-output fresh restore preserving originals, zero-budget journal verification and manifest mutation after queue acceptance. This does not prove pretrained GPU/Windows/host acceptance or direct seek/Stage2/writer resume. See `durable-queue.md`.


M37 adds the independent `--durable-stage2` recovery policy. The actual Qt/ownedworker workflow survives graceful cancellation and a hard child exit, reuses exact sealed rawprediction batches, preserves partial exports and writes9freshverified frames equal to an uncached/pixel oracle. Completedmodelcalls/frames and reusedinstances are reported separately. It does not provide directseek or writerprefix continuation; see stage2-recovery.md and m37-verification-summary.json.
