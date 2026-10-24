# Shared engine

Current extension scope and acceptance are recorded in [product-improvements-brief.md](product-improvements-brief.md)
and [checkpoint.md](checkpoint.md); [review-objective.md](review-objective.md) records the earlier review. The chronological implementation notes below
retain their original verification limits.

Milestone28 adds a companion completed-cache workflow: pure `ofx_record.cache_spec` admission is shared by the desktop and sealer; `CachePreparation` owns immutable bounded record bytes, background path/native I/O, coalesced progress, cancellation and one terminal result. `desktop_ofx.OfxCachePanel` consumes that result through the existing main poll and close ownership. No Qt objects enter the worker, no model loads during UI import, no new inference implementation is introduced. Publication remains authoritative after commit even if owned-stage unlink fails. Exact channel admission is opt-in for the existing reader and required for cache sealing. The version1 native cache/host contract is unchanged; source-content/time/colour binding remains unfinished.

Current working path: `infer.py` validates `config.InferenceConfig`, snapshots a `session.JobRequest`, and calls `InferenceSession.run`. The session lazily loads a model through `runtime.load_model` and calls the existing `runtime.run_inference`. The interactive launcher uses the same CLI parser in process and retains one session across clips; `python -m corridorkey.batch` also uses this path. Single-shot CLI owns and closes its session. `InferenceConfig` remains importable from `corridorkey.runtime`, and `WriterConfig` from `video_io`, for compatibility; both definitions are dependency-free in `config.py`. Public `CorridorKeyV2` loads lazily.

Pipeline inspected end to end:

1. Resolve local runtime/foundation assets and checkpoint architecture, load trainable state, freeze/evaluate foundations, disable stage-1 recurrence sites for the exported pre-recurrence checkpoint.
2. Probe video or naturally ordered image sequence; derive native dimensions, reflect padding (default quantum 128), FPS and selected video range. Display-coded video passes through sRGB-to-linear then signed-asinh model encoding. Foundations get an SDR proxy; EXR can retain signed HDR linear RGB when selected explicitly.
3. Run RVM chronologically across all selected frames in a prepass. State starts at None per clip; normalized features are float16 disk memmaps. Partial video processing starts fresh at the selected range, not at frame zero of the original clip.
4. Normal mode recomputes C-RADIO/MoGe on overlapping temporal windows by default. Optional bounded float32 RAM reuse extracts only missing IID frames and restores them to original window positions. Low-VRAM mode precomputes them once into float32 disk memmaps. Stage 1 consumes the complete native-rate RGB, hints, quality values and features. Predicted carried hints override full hints in overlap unless carry is disabled; first-frame hint overrides frame zero. RVM prepass/state is independent of IID reuse.
5. Blend overlapping /16 baton features with tapers on CPU float32 and finalize to float16. Normal mode streams finalized frames into Stage 2; low-VRAM mode stores a full baton then moves foundations/Stage 1 to CPU and Stage 2 to GPU.
6. Stage 2 batches native-rate RGB plus baton, predicts alpha and reconstructed foreground, decodes asinh/inverts video adjustments and renders only selected outputs. Alpha-only bypasses foreground rendering. Checker construction now occurs only when requested. Submitted batch/frame counts are checked before reporting completion.
7. Bounded writer transport (`lifecycle.WriterWorker`) feeds video FFmpeg or EXR callbacks. Preview outputs clip RGB; EXR float retains signed HDR; half retains existing +/-65504 saturation. No new precision or colour policy was enabled.

Configuration, chronological scheduling/source discovery and failure-aware writer lifecycle are isolated without duplicating inference. Job temp directories are unique and removed through cleanup that attempts all callbacks. Sessions reject concurrent run/close requests and own one model. Inference failures and KeyboardInterrupt discard that model; preflight failures preserve an untouched healthy model. All low-VRAM module chunking overrides are applied/reset on every job, and recurrent/hint/window state remains local to each runtime call. Independent sessions still require separate output directories. Progress and thread-safe cooperative cancellation now pass through sessions, feature prepasses, Stage 1, Stage 2 and writers. Queue/finalization deadlines supervise an owned FFmpeg process; abort terminates and reaps it. Writable/read-only memmaps and prefetched source generators have explicit scoped cleanup on success and exceptions. Actual Windows behavior remains unverified; in-process native decode, EXR storage calls and GPU kernels cannot be forcibly interrupted.

Requests/results contain immutable JSON snapshots. Results include status, requested/effective settings, actual output contracts, model-load and processing times, first CPU-rendered batch latency and submitted frame count. The original exception is raised on failure; `session.last_result` holds its serializable record. Optional stage profiling synchronizes CUDA and reports inclusive scopes and allocation peaks; ordinary throughput avoids those extra synchronizations. `--diagnostics` writes a local atomic record; benchmark provenance/hashing is separate from normal inference.

Foundation reuse is scoped to one job and never shared with another source/model/config. `cache.ByteBudgetCache` owns LRU admission/accounting; `foundation_cache.FoundationWindowCache` owns per-frame CPU copies and full-window assembly. The RAM limit covers retained payload only; an optional automatic VRAM selection cap limits the inherited heuristic's budget without changing allocator policy. Current/final/peak cache residency and actual foundation evaluation counts are diagnostic fields. See `feature-cache.md` for scope, transfer costs and GPU validation limits.

Next boundaries: GPU parity/performance validation, total resource budgets and richer error contracts. Durable cache identities must include source/hints/corrections, weights/stats, colour/preprocessing, precision/backend and temporal policy; recurrent outputs depend on the preceding shot state. Durable reuse must verify checksums and completion markers and publish atomically. Opt-in completed-job EXR replay and durable batch-attempt recovery are now implemented in `recovery.py` (`recovery.md`). Mid-shot recurrent checkpoints and durable intermediate feature reuse remain unfinished. This synchronous session is not a background daemon or multi-job scheduler.

Desktop and future OFX adapters call this core. The optional Qt Widgets frontend uses the validated worker boundary described below. Do not introduce another inference implementation or a second language before measured need.


Opt-in offline Quick Key shares the same runtime/preprocessing/model/render/export interfaces. Interval 1 bypasses the draft path. Sparse mode chooses contiguous native anchor contexts within supplied/detected shot/reset segments, resets chronological RVM at boundaries, blends only required anchor batons and runs Stage 2 on anchors. Job-local float32 anchor-alpha maps feed `draft.offline_frames`; `motion.interpolate_alpha` performs CPU flow/warp/reliability weighting. Current source RGB is read separately at float32 and exported with draft labels. Explicit uncertain-region policies stop, approximate or request additional native-context neural work. Existing full-rate defaults remain unchanged. See `quick-key.md` for costs, diagnostics and implemented cut/adaptive/hold/causal policies and implemented correction inputs and remaining UI/validation work.


Opt-in reduced-proxy cut analysis streams JSONL diagnostics before RVM/context scheduling and unions candidates with supplied cuts. The detector is heuristic and may miss cuts or flag flashes. Maximum-gap insertion supplements regular anchors. Adaptive offline neural fallback promotes anchors for subsequent interpolation; causal propagation promotes requested neural fallbacks and can additionally trigger on motion. Hold performs no flow. Causal Stage 1 owns each past-only context's anchor, preventing later windows from changing earlier batons; its planner does not add previous-final anchors from future cut/end knowledge, and RVM uses one-frame chunks. CPU changed-future tests include an automatically detected new cut and both memory paths. The actual architecture's Stage 2 is per-frame (B×T spatial processing), and RVM's recurrent core iterates T chronologically; GPU/model validation remains external. All modes still precompute the full clip before output; no live/zero-startup claim is made.


Authored corrections use a dependency-free immutable schema/provenance/history layer (`corrections.py`) and a native CPU raster operator (`correction_rasters.py`). Shared `identities.py` preserves the benchmark's source-hash definition. Sessions validate schema/source/asset content before model loading and carry the verified snapshot through diagnostics to runtime; raster decoding/probe metadata is checked before stages. Both-direction authored mattes feed the existing single-channel hint interface after first/full/carried hints, with partial keep/remove edits only where a known base exists. Both-direction constraints refine predicted auxiliary carry and flow anchors. Directional ranges are explicit static raster output constraints; no reverse neural pass or unknown per-pixel hint channels are invented. Output alpha math uses float32 and leaves direct foreground colour reconstruction unchanged. Raw neural alpha, flow conditioning and final ordered constraints remain separate so directional masks do not leak into flow state or apply twice to an anchor. Project shot/reset boundaries segment RVM/Stage 1 even in full-rate mode. Every job recomputes; the history/invalidation API returns whole authored shots for future editor/recovery integration. Real-model quality and GPU acceptance remain external.


The image boundary now preserves decoded uint16 levels in float32, reports observed decoded dtypes across passes, rejects non-finite/geometry-changing inputs and keeps model/foundation bfloat16 conversion explicit. Scene-linear RGBA unpremultiplication and linear EXR premultiplied cutout are optional; legacy ignored-source-alpha/straight-output defaults remain. `refinement.py` optionally adjusts only soft alpha using spatial/RGB-proxy weights before authored constraints, exactly once per neural/output frame. Optional blue/green, linear-Y-preserving and edge-weighted despill shares the existing render helpers. Default green equations are CPU-exact with the previous path. Model/decode overflow fails before export. `colour.md` documents assumptions/domains, numeric bounds, metadata limits and pending high-bit video/managed source colour; there is no automatic ICC/OCIO/PQ/HLG transform or new backend.


The opt-in recovery controller wraps the existing session rather than duplicating inference. Canonical job/content/model/backend/implementation identities select immutable checksummed EXR bundles. Owned staged publication, SQLite single-owner locking and independently committed attempt journals allow completed outputs to be verified/restored after process exits. Partial jobs restart whole in fresh output directories; old outputs remain. Model-loaded asset paths and batch model context are pinned to prevent old resident weights from being labelled with new disk asset hashes. No automatic memory profile, backend fallback or new precision path is selected. Reports distinguish fresh session runs from historical byte replay; native GPU and target-platform acceptance remain external.


Durable IID recovery uses typed frame identities and `DurableFeatureFrames` on the same owned `BundleStore`. Shared-session model binding precedes runner access; `RunDiagnostics` carries the optional live adapter and serializes counters only. `FoundationWindowCache` consumes it in both normal windows and the low-VRAM prepass. Frame entries store exact normalized float32 C-RADIO/MoGe tensors, copied out of closed read-only maps and verified before use. Producer input bits/dtype and complete parent provenance form conservative keys. RVM/Stage1 recurrence restarts; no recurrent state is reused. The recovery CLI alone opts in via `--durable-foundations`; GPU numerical/performance acceptance remains external.


Optional `DurableRvmChunks` complements the IID adapter. Both inherit the same `BoundModelCache` asset/context guard and are bound before the shared runner. RVM entries preserve complete exact-call features and outgoing state as one atomic safetensors payload, keyed by parent job, actual input/state bits, call bounds and shot start. The prepass walks every chunk in order, restores exact state on hits and computes misses from that actual state, resetting at cuts. Both normal/low-VRAM modes use its reconstructed float16 memmap. State dtype is retained exactly; Hann carry/blending/output remain job-local and restart. `--durable-rvm` is explicit/default-off; native CPU/random-weight fixtures are not pretrained GPU or host acceptance. See `recurrent-checkpoints.md`.


Optional `DurableHannWindows` replays exact raw Stage1 events through the existing Hann carry/correction/blend/flush loop. Typed identities bind parent, complete schedule/cursor, incoming live replay-state digest and exact RGB/hint/quality/MoGe/C-RADIO/RVM input tensors. Baton and used auxiliary alpha preserve producer precision in one verified safetensors entry. Both memory paths and sparse/fallback contexts reconstruct state chronologically; active Stage1 recurrent sites reject this mode. Stage1 counters increment only on actual model calls, while durable diagnostics report verified event hits. Foundation work/hashing and Stage2/writers continue; this is not direct range seek. `--durable-hann` defaults off and combines with the prior explicit caches. See `hann-recovery.md`.


`worker_protocol.py`, `worker_core.py`, `worker_transport.py` and `worker_client.py` isolate the existing session/recovery engine in an owned child with one model owner and a bounded waiting queue. Concurrent input controls use the same cancellation token; output carries bounded/coalesced events with generation/sequence and native/Python logs on separate stderr. Normal jobs reuse one session; recovery requests deliberately retain fresh batch lifetime to preserve model identity. The working worker CLI uses this exact client. Uncertain/crashed work is lost, never automatically replayed; full report references retain recovery provenance. POSIX force-close/cancellation is locally tested, while GPU/Windows/host acceptance remains external. Desktop code can now use this boundary rather than running model calls in its UI thread; `worker.md` gives lifecycle and limits.


The optional Qt Widgets desktop (`desktop_window.py`) owns UI state only. `DesktopBridge` uses a dedicated thread to create/control/close the existing `WorkerClient`, with bounded commands/events and coalesced progress; it never runs another inference implementation. `PreviewLoader` owns latest-only CPU source/export decoding off the event thread, with immutable display snapshots and actual shared colour/image adapters. `desktop_state.py` persists strict bounded shared-job snapshots atomically, protecting sources/hints/corrections. Projects do not automatically render or restore queue state. Unique render output children preserve previous jobs; completed views use their original immutable settings. Qt is isolated in optional requirements/modules; desktop CLI help and shared settings remain headless. See `desktop.md` for exact bounds, native decode/shutdown limits and unfinished acceptance.


The optional developer benchmark dialog calls the existing headless runner through `BenchmarkTask`, an owned process with concurrent bounded log collection, cooperative group cancellation and force/reap cleanup. Desktop gating stops the resident render bridge and waits for clean completion before benchmark startup, rejecting concurrent render/benchmark owners. Actual report parsing/export runs off the Qt event thread; strict byte/depth/finite limits and fixed text bounds protect inspection. `benchmark_report` only displays recorded fields and preserves exact source bytes on export; independent baseline inspection does not derive speedups or certify comparable experiments. `benchmark.py` preserves partial CPU cancellation/timeout records and emits optional phase progress outside measured scopes. See `developer-benchmark.md` for scope/platform/native-I/O limits.


The native correction editor separates `CorrectionCanvas` Qt drawing, `CorrectionDialog` UI state, one `EditorService` background owner and a shared-contract `CorrectionEditor` model. Actual source probes/content hashes bind native geometry and selected range before authoring. The model reuses existing document/history/raster operators, publishes unique verified mask assets and immutable recovery/save snapshots, and reports full-shot invalidation. Queued normal jobs keep prior manifest paths; new attachment changes future settings only. Editor128-snapshot history/canvas admission and strict4MiB core document I/O are explicit bounds, not global memory/disk budgets. No duplicate neural path or fabricated partial-hint/trimap channels were introduced; `correction-editor.md` documents scope and unverified host/GPU behavior.


`QueueStore` journals canonical request intents and observed worker events in independently committed SQLite data while a separate SQLite exclusive lock owns the writer. `DesktopBridge` snapshots input before command admission, records before dispatch/delivery and treats post-send persistence errors as uncertain/authoritative-drain outcomes rather than false rejection. `QueuePanel` only inspects history/loads form settings unless activation is explicitly requested. `queue_snapshot` resolves/copies accepted batch settings to owned checksummed files using the existing batch parser, preserving source/asset path references and stable default-path identity. `RecoveryPanel` submits existing worker recover requests and inspects real reports/metadata; the existing controller alone validates outputs/cache/context. Shared bounded local JSON I/O also serves benchmark inspection and strict batch parsing; inference architecture is unchanged. See `durable-queue.md` for acknowledgement, ownership, bounds and unverified native/platform behavior.


`desktop_settings.py` applies complete shared-codec validated core/writer snapshots with original-text stale-edit detection and exact numeric text. `desktop_recipe.py` prepares immutable one-job batches off the UI thread without submitting. `runtime_paths.py` separates bundled code root from optional asset root while re-exporting RuntimePaths through runtime; InferenceSession/SharedEngine/WorkerClient pass selected assets consistently for normal and fresh recovery sessions. The recovery implementation hash remains bound to actual code. `asset_inventory.py` reads metadata and probes backend availability in a bounded child; `desktop_assets.py` activates selected paths only after clean processing shutdown. Presence is not checkpoint compatibility; there is no added inference backend or fallback. See desktop-settings.md for exact scope and native Qt lifetime uncertainty.


Queue history keyset paging uses SQLite admission sequence rather than timestamps, reads at most129 headers for a128-row page, and retains existing selected-detail checksum validation and readonly ownership semantics. `resource_snapshot.py` adds opt-in resource data to worker status without importing Torch/initializing CUDA: process peak RSS and only already-initialized current-device allocator counters. `desktop_resources.py` displays explicit worker snapshots and off-thread containing-filesystem space; parent shutdown waits for its I/O and dispatch remains under existing processing ownership. Replaced correction dialogs are explicitly retired after actor shutdown and the dirty guard. These do not add a processing backend, automatic budget or global memory claim.


`QueueStore.seal` obtains exclusive journal ownership, validates a bounded owner-bound archive marker and publishes complete fsynced marker bytes via an exclusive hard link. Future writers re-check seal state after acquiring ownership; inspection/paging retain original checksums and unresolved semantics. `QueuePanel` performs explicit archive/fresh transitions off-thread, re-reads archive state after errors and activates only through existing clean processing/asset barriers. Old directories are retained in place, with no automatic replay/deletion. Main-window dirty correction admission now precedes shutdown; accepted editor close blocks input even after late events. See queue-archives.md for filesystem/native/backup limits.


`desktop_comparison.py` captures validated actual render ranges and maps source coordinates to immutable export paths. `PreviewLoader` atomically returns both thumbnails for synchronized A/B; mismatches/missing metadata reject rather than guessing offsets. Optional core `start_frame` flows through the shared video/sequence probes and correction binding; selected-clip temporal indices stay local. Desktop range controls write validated future settings only. See source-ranges.md.

`app_process.module_command` preserves Python module launches in source checkouts and selects explicit application-owned worker/backend/benchmark/batch entry points when frozen. `bundle_entry.py` preserves console streams and freeze_support; `desktop_bundle` generates a private Windows PyInstaller one-directory build after preflight/source gates, retaining failure reports and existing notices while leaving model/media assets external. `desktop_smoke` verifies explicit backend/worker launch/pipe lifecycle only. Native bundle/GPU/host acceptance remains separate; see packaging.md and resolve.md.


The experimental native OFX adapter now serves checksummed completed RGBA caches prepared by `ofx_cache.py` from actual shared-engine linear EXR exports. It duplicates no inference. Pinned official ABI headers, strict per-frame/metadata admission, host image leasing, signed-stride/window/association handling and conservative serialized rendering are exercised by native unit hosts and sanitizers. Source matching/colour/time binding is explicit; Windows/Resolve/GPU acceptance remains unverified. See ofx-cache.md.


Producer-bound cache path: optional SourceWitness observes native decoded float32 RGB across all shared full/draft read paths; private observer never enters serialized metadata. Allselected stable digests finalize after owned cleanup into a bounded sidecar; completed record carries its contract/hash. Sealer copies verified digests into CKOFX002 header-bound table. Native effect verifies all mapped Source dependencies before Output lease/write and uses per-instance render serialization for upstream reentrancy. No mutable host source epoch is trusted. The paired benchmark companion calls the existing inference runner, freezes per-side requests and validates actual effective conditions; desktop uses existing background launch/report ownership and explicit frozen/source dispatch. Neither path adds a model implementation/backend/precision fallback.


Managed recovery-cache administration is separate from BundleStore construction so inspection cannot create/prune. cache_maintenance holds the existing rollback SQLite ownership lock, validates bounded strict owned metadata/stat snapshots, and binds one-time trim plans to current inventory. Apply rechecks and removes flat owned entries while preserving all external outputs/journals/reports/staging; interruption reconciliation reports committed/partial/unknown values. desktop_cache owns background immutable I/O and terminal delivery, participates in parent poll/close after the dirty-editor guard; no inference is duplicated or submitted. See cache-management.md.


Recovery transfer is isolated in cache_backup.py, with bounded flat numbered payloads, SHA256 manifest, read-only cooperating source ownership and immutable snapshot recheck. Fresh restore holds its own SQLite lock and atomically publishes complete workspace metadata last. desktop_cache_backup.py owns background I/O and truthful partial/latecommit delivery; desktop_cache.py includes child ownership in its existing main-window shutdown barrier. app_process explicitly admits source/frozen backup commands. Historical dependency paths remain unchanged and explicit shared recovery still verifies compatibility; see cache-backup.md.


queue_transfer.py exports already archived queues under read-only zero-byte-compatible SQLite ownership into bounded checked flat containers, then restores exact history into fresh archived roots with owner marker published last. queue_store.py reads separate queue-transfer.json relocation authority and returns transient load_payload without rewriting historical request/event bytes. desktop_queue_transfer.py owns immutable background operations, cancellation/terminal delivery and explicit history selection; desktop_queue.py retains selectedRows authority and includes child ownership in main-window shutdown. See queue-transfer.md; native matching path flavour only.


asset_audit.py is stdlib-only nonexecuting role/container inspection with streamed hashes, bounded same-descriptor ZIP reads, bounded C-RADIO discovered frontier and lexical/target identities. Exactreport export re-audits then fences observed files/directories/missing/preference state afterstaging beforeexclusivepublication. desktop_asset_audit.py owns immutable backgroundaudit/export withcoalescedprogress/terminal/cancel; desktop_assets.py includeschildownership in existing mainclose afterdirtydraftbarrier. Source/frozen CLI route explicit. Runtime/model loading and legacydefaults unchanged; asset-audit.md details limits.

The deliberate asset downloader is separate from inference and global hub caches. asset_download validates immutable pinned publisher plans, bounds anonymous HTTP and payloads, binds hashed descriptors to staged publications and seals provenance last in unique roots. desktop_asset_download owns background progress/cancellation and explicit selection; desktop_assets excludes sibling operations and drains both children before close. Source/frozen dispatch uses app_process. Existing backend admission, session ownership and legacy source downloader remain unchanged. See asset-downloads.md for integrity, partial outcomes and DNS/production limitations.

Shared runtime scratch admission uses pure temp_budget.MapSpec/TempMapBudget. Actual conditional geometry/model/schedule produces the complete writable-map recipe; a single jobaccountant is passed explicitly through prepasses/Hann/fallback and validates each writable creation. Readonlyreopens are uncharged; claims persist until actual temporary-directory cleanup. Reports are primitive JSON under execution.temp_maps. No global allocator hook, environment setting or parallel inference engine. See scratch-budget.md.

process_memory uses bounded birth-pinned process identities and platform RSS providers, keeping incomplete totals unavailable. memory_guard serializes sampling separately from cached-state reads so finaldiagnostics cannot waiton nativequeries. JobControl optionalcheck_hook runs after cancellation; default-off sessions construct noobserver. Session owns perjobguard or reuses recovery suppliedguard, enforces forcedload/completionchecks and writes onlycached diagnostics. Recovery checks the same guard across hashing/verified-existing/restore/publication and preserves completedhistory on failure. No monitor daemon/globalallocator policy. NativeMac scope verified; Linux/Windows targetvalidation pending. See memory-stop.md.


ExactStage2 replay uses typed stage2_identity.Stage2BatchIdentity and durable_stage2.DurableStage2Batches on the same owned BundleStore. Parent model/assets/backend validation precedes adapter binding and calls. Identity hashes actual RGB/baton args and ordered native subset with complete shape/dtype/execution contract; the cache stores selected rawprediction before any external float promotion/refinement/correction/render. Lookup validates bounded safetensors and returns an independent tensor preservingbits, then unchanged downstream paths run. Counters separate completed actual forwards, reusedbatches/instances and completedpredictions; failedjobFPS staysnull. Opt-in store metadata/inventory and expectedpayload bounds protect entry reads/copies without changing validlegacy defaults. No secondmodelengine/directseek/writerqueue persistence. See stage2-recovery.md.


## Evaluation dossier boundary

`evaluation_dossier.py` reads bounded existing paired reports, maps explicit cases/categories/splits, validates compatible measurements, and produces hashed immutable evidence. `evaluation_dossier_export.py` revalidates inputs, copies only explicitly budgeted measured media and seals fresh exports. `desktop_evaluation_dossier.py` owns background I/O requests and pending terminal results; `desktop_benchmark.py` excludes overlapping report owners and desktop shutdown waits for the child. No model, network, or human-quality verdict is introduced by this workflow. See [evaluation dossiers](evaluation-dossiers.md); M38 mechanics are locally verified; representative quality acceptance remains pending.


## Profiled output milestones

`output_telemetry.py` owns bounded scalar observations only. Runtime registers final output frames before queue submission; optional EXR callbacks record successful native write returns. All-output native maxima and queue acceptance are distinct. Observations and cached snapshots introduce no model, I/O, UI or hardware-query ownership. The existing profile flag exclusively enables collection; throughput runners reject profiled jobs. Details: [output-telemetry.md](output-telemetry.md). M39 locally verified; GPU/production timing remains pending.


Trimap refinement uses correction schema 2 and a strict class asset loader in
`trimap_refinement.py`, separate from ordinary coverage normalization. Bounded
native tiles refine unknown pixels after existing global alpha refinement;
known targets and ordered entry strength remain explicit. Trimaps never enter
neural hints, temporal carry or flow anchors. Full-rate native guidance has an
owned reader; Quick Key uses current native source RGB. Editor import/save/reopen
and attachment reuse the existing immutable correction workflow. See
[trimap refinement](trimap-refinement.md) for the exact admission and algorithm.

The fresh-review input-preservation guard shares deterministic output names in
`output_paths.py` with runtime writers. Bulk source/hint/asset inventory belongs
to `InferenceSession.run` and direct `run_inference`, before model loading or
output mutation; generic `JobRequest.validate_source` stays lightweight for its
GUI callers. Runtime repeats admission at output creation boundaries. Directory
ancestry, resolved paths and existing file identities protect user inputs while
ordinary unrelated rerenders remain supported. This is cooperating-process
preflight, not a filesystem reservation against concurrent hostile changes.


Final review read boundaries: queue SQL CASE projections validate persisted text
types/byte extents before Python receives payloads, on the same SELECT snapshot.
Metadata uses at most two rows; paging retains its129/128 lookahead contract.
Recovery ownership records are bounded before opening/pruning, while generic
completion manifests retain the configured store quota. RVM/Hann payloads use
one pinned regular descriptor, admitted extent/hash and exact header/provenance
checks before safetensors loading. Worker class-owned reaping and root signalling
share one lock; timeout releases it before forced cleanup. See the final
[review evidence and limits](review-2026-10-23.md).


The ten-feature extension starts with `readiness.py`, a Qt/model-independent
metadata service returning named PASS/FAIL/WARNING/NOT VERIFIED observations.
`desktop_readiness.py` owns cancellable off-thread checks, export and immutable
short-job preparation; actual processing uses the existing `DesktopBridge` and
shared engine. Requested/effective settings and unexecuted stages remain distinct.
No new inference implementation or automatic model/backend selection is added.

`playback.py` plans one outstanding source-coordinate frame or atomic A/B pair.
Its caller-bound identity and generation reject obsolete completions after
pause/seek/rebind; it reuses `RenderView` for original recorded export offsets.
`desktop_playback.py` consumes the workspace's existing sole `PreviewLoader` and
50ms timer, retiring pending ownership even when manual previews supersede it.
Initial desktop playback requires source sequences with declared rates and EXR
exports; video/VFR timing is explicitly unsupported. Preview display conversion
and scheduling do not change rendered pixels or future job settings.


Explicit `readiness_capabilities.py` probes use stdlib nonblocking bounded pipe
reads and direct owned subprocess cleanup; they are separate from lightweight
metadata checks and model execution. The readiness panel reuses its existing
single I/O owner for explicit completed-frame decoding, retaining one thumbnail
and immutable matching engine metadata. GUI presentation and JSON evidence are
separate from model/quality acceptance; no pixels enter reports. `desktop.py`
defers first-run introduction through the existing Qt loop/preferences, without
starting work. See [readiness](readiness.md).


The readiness report retains matching failed/cancelled worker metadata and the
last attempted checkpoint, copied before cleanup. Missing/contradictory metadata
remains NOT VERIFIED; partial CUDA work never implies success. A/B preview
surfaces share decoded-pixel zoom and normalized pan with wheel/drag and explicit
Zoom/Fit controls. Display transformations preserve render settings and pixels;
scale refers to bounded decoded previews, not native source resolution. No
inference default, precision, dependency or saved-format change is introduced.


The headless correction_tracking kernel streams immutable float32 candidates from
explicit source/anchor requests. The existing editor owner publishes bounded
sequence previews, then one explicit schema3 history revision. CorrectionRasters
replaces participating constraints once at their original slot for final alpha;
original hints/carry remain separate. Shared asset inventories protect authored
and derived dependencies, and runtime admission checks decode/cut bindings.
[Tracking contracts](tracked-corrections.md). SequenceIndex is a bounded frozen
path/stat manifest consumed by the existing preview decoder; source preparation
shares its single queue and lifecycle, without a second decoding owner.


Clean-plate assistance runs on the existing correction service owner. The pure
linear-RGB kernel produces conservative trimap guidance; explicit Inspect/Accept
uses the existing atomic raster import/history path. Schema4 origins bind local
references, decoded observations and generation settings while the shared runtime
continues to apply ordinary corrections. Source decode contracts are checked at
session preflight and effective runtime boundaries. Generated guidance does not
add a neural channel. The bounded image-admission helper captures immutable
encoded bytes after header geometry checks; `video_io` accepts those bytes through
the same channel/precision/association/color math as its existing path decoder.
See [clean-plate guidance](clean-plate-guidance.md). Optional segmentation has a
headless single-owner provider foundation only; no model is loaded at UI import.


Optional subject clicks connect through the same correction service and native
canvas mapping. Prepare captures bounded immutable source/proxy bytes; Predict
owns one persistent isolated CPU float32 provider child with separate32MiB
framing, bounded queues/logs and deadline/cancel termination. This does not reuse
the render worker's4MiB protocol or GPU ownership. Captured source, exact native
RGB, ordered prompts and provider provenance bind accepted schema4 Keep masks.
The service admits one active operation, preserves cancellation until retirement
and retains child ownership until verified shutdown. See [subject clicks](subject-clicks.md).


Portable project schema2 is separate from legacy desktop schema1 and sealed
queue-history transfer. Typed dependency groups retain complete source ordering
and offsets, immutable file pins/hashes and original job wire. Authored/tracked/
guidance asset relocation reuses the shared correction inventory and verifies
unchanged semantic document identity. Metadata preview precedes explicit bounded
verification/collection. Completion authority is an exclusive zero-byte marker
created after metadata sync/final fences; loader and Apply pin both marker and
manifest. Correction relocation publishes from a private pending file in a fresh
owned scratch directory. A single background UI owner formats bounded paged
summaries and retains late committed results without applying them. Main applies
only a strictly validated acknowledged snapshot, preserving queue/model owners.
See [portable projects](portable-projects.md); cross-path-flavour support remains
unsupported and model runtime compatibility remains unverified.

Progressive preview scheduling is Qt-independent in `progressive_preview.py`:
immutable context/stage/attempt identities, bounded priority work, one retained
owner and exact-provenance thumbnail cache. `preview_execution.py` captures
strict original jobs/source/model metadata, admits selected encoded/native frames
and creates fresh per-request workspaces without importing neural models.
The desktop panel submits the dedicated transient `preview` action through its
existing DesktopBridge/WorkerCore/SharedEngine/InferenceSession. It serializes
preparation, native terminal observation, thumbnail decode and verified cleanup;
exports cancel/disable preview admission and keep their original recipes. Worker
replacement/source/settings/view changes invalidate captured contexts. No preview
artifact is promoted into export/recovery storage. The initial single-frame
sequence path explicitly refuses unsupported temporal/hint/correction contexts;
final labels mean the configured preview recipe, not full-range equivalence.

`tracking_overlays.py` converts actual TrackingCandidate reliability/uncertainty
bytes into bounded immutable display proxies and verified exclusive raw-map
receipts. `correction_tracking_editor.py` captures them only when explicitly
requested in the existing flow iteration, with separate retained-byte accounting.
Inspection revalidates the original source/document/decode/frame/proxy binding,
then composites red RGBA on a copied source thumbnail. The ordinary editor service
returns a distinct diagnostic role; no masks/history/acceptance flags are changed.
Dirty navigation guards, late-cancel display fencing and source reload before
painting preserve ownership. Accepted-mask assets survive separate diagnostic
subdirectory cleanup; missing historical spatial data stays unavailable.


## Machine calibration ownership

`calibration.py` supplies Qt-free immutable verified plans, controlled profile and
workload admission, measured-only summaries, duration assumptions and validity.
`calibration_task.py` owns one WorkerClient at a time, explicit warmup/cold/warm
trials, cancellation/containment, retained raw records and committed reports.
`desktop_calibration.py` performs inventory/hash/report I/O on an owned thread and
uses bounded timer snapshots. DesktopWindow retires progressive/editor ownership
before task creation, blocks competing processing, and waits for verified cleanup
on close. No queue mutation or automatic recommendation application is exposed.
See [calibration](calibration.md) for evidence scopes and unsupported contexts.


## Archived versus operational portable paths

The foreign all-collected reader extension uses
`portable_paths.py` for bounded lexical archival job/settings validation.
`portable_project.py` retains original provenance and exact package layout while
resolving collected dependencies and rebuilding native jobs only after explicit
destination and correction-resource rebinding. Desktop report projection reads
archival settings without constructing an operational native job. Public
plan/inspection/pin association checks precede path-derived filesystem work.
See [review and remaining acceptance](portable-foreign-review.md).


## Immutable photo preview snapshots

`preview_background.py` admits a Qt-free explicit recipe, validates bounded local
header/encoded/decode identity and produces immutable linear709 bytes with cached
identity/profile/geometry/provenance. Compositing creates fresh linear pixels;
`desktop_preview.py` applies the existing display conversion once. The existing
PreviewLoader handles background preparation, ordinary decoding and atomic A/B
on one owner, with capacity1 pending/result queues. Playback carries the same
snapshot and binds by its cached identity. Desktop controls perform no image I/O
on open, fence superseded selection/late results, and wait for loader close.
Displayed metadata commits with accepted pixels so static pins retain the visible
frame/photo identity even while a later frame is pending. No engine/export path
or saved job format changes. [Workflow/limits](photo-backgrounds.md).


## Reusable export playback metadata

`preview_exports.py` captures immutable validated RenderView/mode requests without
I/O. The existing PreviewLoader prepares deduplicated bounded source/FG/alpha
indexes and cached identities. Playback attaches only matching generation/context
metadata and retains source-global/output-local mapping; separate alpha indexes
feed the same native preview decoder and colour path. Repeated frame reads avoid
directory rescans. Sampled file/root stat fences reject ordinary mutation; explicit
Refresh controls membership. One bundle and the existing single pending loader
retain bounded references; no pixel cache, new thread or model owner. Common
terminal acknowledgement precedes metadata routing, including superseded photo
results. [Workflow/limits](export-playback-indexes.md).


## Region effect separation — checkpoint25

Correction schema5 supports `refinement/edge_finishing`, `composite/despill` and `diagnostic/tracking_overlay`. Existing static/accepted-tracked coverage has purpose-isolated max-weight selection; complete anchor bindings preserve purpose/target, and every region stays outside neural hints, artist matte constraints and forced frames. The editor/portable/history/resource machinery remains shared.

Composite weights affect configured RGB despill before compositing/association in existing neural EXR/video and source-draft renderers; alpha/raw foreground remain separate. Active batches use one incrementally packed native float32 coverage tensor, not selective model execution. Diagnostic weights affect only visible opacity of existing captured tracking maps after native-union/area proxy resampling; observations and render pixels do not change. Missing maps stay unavailable; stale bindings and late source/map/mask mutations refuse publication.

The runtime still calls full-tensor `model.stage2(rgb_t, baton_t)` and requires `(1,T,4,H,W)`. `stage2.py`'s optional `mae_mask_16` performs learned-token replacement after full patchification; it is not an ROI/cost-saving interface. Cropping, temporal context/parity and real quality must be separately implemented/validated before any future selective-neural claim.

Full current-request/recovery/writer identities still include correction manifests/provenance. Existing finishing reuse omits only its explicitly typed final fields; broader semantic dependency projection is remaining local E work, not permission to omit an entire correction manifest. Evidence: `benchmark-results/product-next/batch15-summary.json`.


## Final-only identity and renderer boundaries — checkpoint26

Existing independently opted-in Stage2/upstream finishing contexts now emit privatev2, excluding exactly seven global expansion/feather/despill controls. Directv1 retains its original two-field canonical semantics. Full ReplayIdentity, writer-prefix, fresh attempts, journals and final provenance remain complete; all source/hint/correction/profile/model/state/schedule dependencies remain conservative. Typed adapters validate full current context at bind, use bound-owner/cancel checks peraccess, and recovery reidentifies the whole current job before final publication—not full rehashing per hit. No wire/seal migration or deletion.

Renderer candidates are shared only inside one call and one identical working domain. Mixed linear/display domains remain distinct. Globalzero skips composite-mask pack/transfer at the responsible batch helper without waiving fullasset preflight/directmask validation. Removing genuinely unused linear arithmetic repairs the finite-HDR false rejection; actually used linear overflow guards remain. No new cache/service/flags/NN path. Exact/provenance/component evidence: `benchmark-results/product-next/batch16-summary.json`, [guide](render-finishing-performance.md).

LOCAL next: guarded pure composite/diagnostic document projection must retain exact source/shots/presence, since pure documents still affect Hann segments, RVM reset starts and Quick cuts. Mixed/reset/hint/refinement/guidance documents remain conservative. No blind wholemanifest omission, selective neural ROI, hard resource cap or GPU gain inferred.


## Guarded correction-region reuse — checkpoint27

The existing two typed finishing-context factories now produce privatev3 only for strictly parsed nonempty schema5 documents containing exclusively composite/despill and/or diagnostic/tracking_overlay regions (and validated accepted tracks). Shared helper verifies original wrapper shape, identity/source, strict document/track structure; its closed projected record preserves exact presence/source/shots and omits the actual corrections path only in that private context. Pure region masks do not enter hints/carry/forced frames, but document shots affect Hann/Quick/RVM scheduling, so remain exact. Mixed/empty/ordinary/refinement/reset/trimap/guidance cases retain exactv2 factory behavior; directv1/v2 payload/key semantics remain.

Full current Replay identity/path/manifest/assets/admission/bind/final-publication/writer-prefix/journal/profile provenance and every existing raw-input/state/group/precision/model contract are unchanged. Projection records are structural cache identities, not authenticated provenance; valid raw intermediates can survive a later whole-job failure, but no stale completed whole-output publication is allowed. No new cache/service/DAG/flags or runtime scheduler. Existing CLI/UI help only updated truthfully. Details/proofs `benchmark-results/product-next/batch17-summary.json`, [reuse guide](upstream-finishing-reuse.md). Mixed-document dependency reuse remains LOCAL work, not external-only/fullE completion.


## Mixed final-only versus neural edits — checkpoint28

The existing selector now emits privatev4 for schema5 documents with retained nonfinal entries, including no-final-region cases. Every original wrapper/document/track binding is admitted before filtering. All ordered nonfinal entry fields (IDs/revisions/disabled controls/paths/hashes/reset/trimap/refinement), complete guidance, source, shots and presence remain. Only proven composite/diagnostic entries, accepted derived tracks, outer revision and private path are omitted; the filtered trackless revision0 document uses the existing strict parser. Purev3/directv1–v3 and None/empty/older-schema factoryv2 semantics remain.

A pre-implementation state/input proof showed ordinary-track retirement is raw-neural-neutral but changes final alpha. Thus current B is always rendered from reused raw data and must equal clean B—not old A. Full current Replay identity/admission/bind/final publication/writer-prefix/journal/provenance and Editor retirement behavior remain unchanged. No runtime.py/model-forward/adapter/store edits, new service/DAG/flags or seal migration. [Reuse guide](upstream-finishing-reuse.md); `benchmark-results/product-next/batch18-summary.json`.

Genuine neural authored/hint edits still change retained parent SHA even for identical earlier calls. This remains LOCAL incremental work. Independent facts identify complete Stage2 grouping/input/execution keys, RVM explicit state, Hann complete input/carry state plus conservative global schedule, and an IID limitation: producer batch membership is provenance rather than key, with documented GPU partial-batch roundoff. Do not broaden the shared IID scope or claim universal GPU bit-exactness from CPU fixtures. Preserve the trusted loaded-model boundary and all unknown model/config/runtime dependencies in any later stage-qualified design.


## Stage2 actual-input qualification — checkpoint29

The existing Stage2 reuse factory now emits privatev5. Original full correction wrapper/document/tracks/source and hint file/ordered-sequence records are structurally validated before fixed private slots replace only their semantic identity/path. Source/model/assets/code/environment/settings/writer and unknown siblings remain bound. Every hit still requires the complete actual RGB/baton/frame/group/policy/device/autocast identity. This relies on the existing trusted loaded eval-model boundary, not arbitrary mutable/custom models. Directv1–v4 readers/keys remain; current upstream factories v2–v4 are unchanged.

Current upstream reconstruction and full actual job admission/bind/final publication/writer-prefix/journals/provenance remain separate. Late-edit lookahead was observed, not assumed: four early complete calls matched, three affected calls recomputed; initial hint influence can reconverge, so later matches are valid. Bad consumed late hints can fail after earlier valid hits—no completed stale output. No runtime.py/model-forward/adapter/store or new-flag change. Evidence `benchmark-results/product-next/batch19-summary.json`.

The integration exposed an invalid RSS test oracle: a correctly included32MiB child need not make total RSS grow24MiB when the parent shrinks. Test-only same-snapshot additivity and a deterministic root-shrink regression preserve detection/threshold/lifecycle checks. Production memory code unchanged; original RSS movement unobserved, not a memory defect or reduction claim.

LOCAL next: existing upstream reuse_scope can qualify a temporal-only RVM/Hann domain while legacy IID uses unchanged v2–v4. Both IID cache-constructor and direct feature-identity routes must reject that temporal-only parent; no generic cache base rewrite is required. Keep current-input/state contracts, trusted model scope and conservative full Hann schedule on cuts/resets. IID producer-group GPU numerical semantics remain unchanged. This is a factual next direction, not implemented acceptance.


## Checkpoint30 — temporal exact-input boundary

Private upstream v5 paired with rvm_hann_exact_inputs, strict original metadata validation before input-qualified hint/correction projection. Existing default-off policies route temporal factory only to RVM/Hann; legacy IID factory unchanged, both IID constructor and direct feature identity refuse temporal scope. O(1) private readonly scope is not a dataclass/serde field. Whole actual input/state/group/schedule/model/backend/precision/context keys and current asset admission, correction/carry, provenance, fresh attempts, writer prefixes, journals, final re-identification and publication remain. Global cut/reset schedule conservatively qualified; active Stage1 recurrence rejected. Trusted loaded model only. Literal upstream v2-v4 factory, Stage2v5 factory, direct Stage2v1-v5/upstreamv1-v4 keys unchanged. Old IID assertions retained; authorized temporal blanket-forwards assertions replaced with complete entry-key membership and cleanB. Returned consumed aux1 channel compared; producer4 channels remain immutable, no storage change. See `benchmark-results/product-next/batch20-summary.json` for proof, composite focused-run limitations and full gate. Next local ownership repair is conditioned full-context progressive capture through the existing Session/runtime, not another engine.


## Checkpoint31 — conditioned selected preview ownership

Existing shared Session/runtime/worker/desktop progressive path retains complete original sequence range, initial/full hints, corrections/resets, RVM/carry/Hann windows and whole Stage2 groups; after ordinary finishing/render it writes only selected EXR index0. Separate preview_complete status and strict sparse admission leave full-export admission unchanged. Typed selector transported through existing diagnostics control pattern to explicit runtime kw; injected ordinary5arg runners compatible, selected custom runner fails closed. All context/source/hint/correction/tracked/guidance dependencies pinned/rechecked and scratch protected. Existing scheduler/stale/queue/model owner/cancel/cleanup remains. Original requested range is unchanged; full context processed separate from exactly1submitted output; telemetry total1/index0/sourceabsolute separate, no exportFPS claim. Writer finalize must succeed before preview completion. Correction graph capture retains the original parsed manifest FilePin and content hash; later admission cannot substitute a newer document. See [feature guide](conditioned-progressive-preview.md) and `benchmark-results/product-next/batch22-summary.json`; full-timeline/read-budget costs explicit, no GPU/latency claim.


## Checkpoint32 — measured authored-tracking disagreement

Existing authored BOTH-direction interior two-anchor propagation optionally retains actual abs(warped-left-alpha minus warped-right-alpha) in immutable little-endianF32 bytes under existing capture flag. Existing editor/bundle/receipt/loader/UI displays third diagnostic layer; no new motion algorithm. Defaultoff no additional retainedmap. Alpha/reliability/uncertainty/fallback/non-timing diagnostics/provenance and four flow calls unchanged. Measuredzero distinct from absence; authored/singleanchor/stopped-uncomputed unavailable, first actually computed rejected candidate may retain measurement. Old4positional Propagation/default candidate construction and old2-map receipts valid. Strict ordered2 or optional3 maps, finiteunit/nativebytes/geometry/hash/filename/source/request/local+absolute frame binding. Existing native4Mi/proxy256/64Mi diagnostics and512Mi mask budgets, cancellation/source drift/stale/ownedcleanup and diagnosticregion opacity remain. Integrity/binding is not authentication against hostile rewriting. Source/render algorithms and default precision unchanged. See [tracking guide](tracking-overlays.md) and `benchmark-results/product-next/batch23-summary.json`. Remaining provider investigation is [dependency-gated](overnight/cfr-provider-decision.md), not speculative CFR certification.
