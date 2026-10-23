# Existing option compatibility inventory

All CLI flags from the starting revision remain accepted. Runtime defaults are in `corridorkey/config.py`; CLI memory-selection defaults differ as described below. Output aliases `raw_fg` and `foreground` map to `fg`, including comma-separated/case-insensitive lists. Quick Key and authored correction documents are explicitly opt-in; see `quick-key.md` and `corrections.md`.

| CLI option | Default | Declared help / choices |
|---|---|---|
| `--input` | `required` | Input video file or directory of image frames. |
| `--output_dir` | `required` |  |
| `--weights` | `None` | Runtime checkpoint (.safetensors or legacy .pt). Defaults to weights/corridorkey_v2.safetensors. |
| `--hint_first_frame` | `None` | Optional matte image/dir; first frame only. |
| `--hint_video` | `None` | Optional full-frame hint video or frame directory. |
| `--hint_quality` | `0.95` |  |
| `--carry_hint_quality` | `0.95` |  |
| `--no_carry_hint` | `False` |  |
| `--hann_chunk` | `None` | Frames per stage-1 window. Default: chosen automatically from free GPU memory. |
| `--hann_stride` | `None` | Frames between stage-1 windows. Default: half of --hann_chunk. |
| `--native_resolution` | `True` | Accepted for clarity; native resolution is always used. |
| `--low_vram` | `False` | Staged low-memory inference. Passing this or --hann_chunk turns off automatic selection. |
| `--temp_dir` | `system temp / corridorkey_v2` |  |
| `--outputs` | `None` | Any of: alpha fg/raw_fg checker cutout. Default: alpha fg. |
| `--num_frames` | `-1` |  |
| `--start_frame` | `None` | Explicit zero-based source in; requires start_mode=begin; num_frames limits the remaining range. |
| `--start_mode` | `'begin'` |  Choices: ('begin', 'middle', 'random_middle') |
| `--seed` | `0` |  |
| `--model_scale` | `1.0` | Finite `0 < scale <= 1`; values below 1 explicitly select spatial approximation with native-sized exports. Actual padded grid and resamplers are recorded; no automatic fallback. [Spatial draft](spatial-draft.md). |
| `--frame_dir_fps` | `24.0` |  |
| `--frame_dir_linear` | `False` | Treat frame-directory inputs as scene-linear RGB. |
| `--cutout_linear` | `False` | Export cutout as straight linear RGB instead of straight sRGB. |
| `--frame_dir_alpha` | `ignore` | Explicit premultiplied decoding requires scene-linear RGBA frames. |
| `--cutout_alpha_mode` | `straight` | Premultiplied supported for linear EXR cutout only. |
| `--alpha_refine_radius` | `0` | Optional native spatial alpha filter radius 0..4; 0 is off. |
| `--alpha_refine_strength` | `1.0` | Blend strength [0,1]; hard alpha endpoints remain fixed. |
| `--alpha_refine_sigma` | `0.1` | RGB proxy range scale [1e-6,1]. |
| `--despill_screen` | `green` | Green or blue excess-channel suppression. |
| `--despill_space` | `legacy` | Legacy domain or explicit linear processing. |
| `--despill_preserve_luma` | `False` | Preserve weighted linear Y before export clipping; requires linear space. |
| `--despill_edges_only` | `False` | Multiply despill strength by 4*alpha*(1-alpha). |
| `--despill_strength` | `0.5` | Despill strength from 0.0 (none) to 1.0 (full). Default: 0.5. |
| `--linear_brightness` | `1.0` |  |
| `--linear_contrast` | `1.0` |  |
| `--linear_contrast_pivot` | `0.18` |  |
| `--output_format` | `'video'` | video: ffmpeg-encoded preview videos (display-referred, clipped to [0,1]). exr: unclamped scene-linear OpenEXR image sequences (full HDR float). Choices: ('video', 'exr') |
| `--exr_dtype` | `'half'` | EXR pixel type for --output_format exr. Default: half. Choices: ('half', 'float') |
| `--ffmpeg_bin` | `'ffmpeg'` |  |
| `--ffmpeg_codec` | `'libx264'` |  |
| `--crf` | `12` |  |
| `--preset` | `'medium'` |  |
| `--pix_fmt` | `'yuv444p10le'` | Pixel format for x264 preview outputs. Default is 10-bit; pass yuv444p for 8-bit compatibility. |
| `--bitrate` | `None` |  |
| `--writer_timeout_seconds` | `60.0` | Positive finite queue/finalization deadline; also used for EXR writer waits. |
| `--ffmpeg_threads` | `2` |  |
| `--stage2_batch` | `1` | Number of /1 output frames to run/render per stage-2 batch. |
| `--low_vram_foundation_chunk` | `4` |  |
| `--low_vram_mlp_chunk_tokens` | `65536` |  |
| `--low_vram_enc_frame_chunk` | `1` |  |
| `--low_vram_swin_window_batch` | `16` |  |
| `--low_vram_full_attn_query_chunk_tokens` | `2048` |  |
| `--corrections` | `None` | Local versioned correction document; core/batch field is `corrections_path`. |
| `--diagnostics` | `None` | Write an atomic JSON job record, including runtime failures. |
| `--profile_stages` | `False` | Synchronize GPU stages for diagnostic timings; changes measurement overhead. |
| `--quick_key` | `False` | Enable offline source-RGB draft approximation. |
| `--quick_interval` | `5` | Native anchor distance; 1 retains the original full-rate path. |
| `--quick_context_radius` | `2` | Native frames on each side of anchors; 0 is single-frame context. |
| `--quick_flow_max_side` | `640` | Maximum motion-estimation side, >=16; no neural resolution change. |
| `--quick_fallback` | `stop` | stop / explicit approximate / additional neural inference. |
| `--quick_min_reliability` | `0.2` | Minimum diagnostic reliability, finite (0,1]; uncalibrated. |
| `--quick_max_disagreement` | `0.25` | Maximum warped alpha disagreement, finite (0,1]. |
| `--quick_cut_frames` | empty | Known shot-start indices relative to selected clip. |
| `--quick_reset_frames` | empty | Explicit reset/start indices relative to selected clip. |
| `--quick_approach` | `offline` | offline bidirectional / hold-last-matte / causal past-to-current; runtime precompute stays offline. |
| `--quick_auto_cuts` | `False` | Opt-in streaming heuristic shot analysis before neural work. |
| `--quick_cut_threshold` | `0.35` | Candidate score threshold, finite (0,1]; not calibrated. |
| `--quick_analysis_max_side` | `128` | Maximum reduced analysis side, integer >=16. |
| `--quick_adaptive` | `False` | Promote requested neural fallbacks; requires neural fallback, excludes hold. |
| `--quick_max_gap` | `None` | Positive maximum scheduled anchor gap, independent of regular interval. |
| `--quick_motion_threshold` | `0.05` | Adaptive 95th-percentile displacement / image diagonal, finite (0,1]. |
| `--foundation_cache_mb` | `0` | Opt-in IID float32 CPU feature reuse, retained payload budget in MiB; 0 disables. |
| `--auto_vram_budget_mb` | `None` | Cap automatic profile selection budget in MiB; not a hard allocator limit. |

## Effective settings and valid ranges

- `hann_chunk`/`hann_stride`: positive integers, stride <= chunk. Omitted memory flags select automatically per clip; explicit chunk/stride or low_vram disables auto. Missing chunk becomes 80; missing stride becomes half chunk (minimum 1). Core defaults are 80/40 with auto off. The auto table selects the largest estimated fitting temporal context; no measured throughput advantage is implied.
- Hint qualities and despill: finite [0,1]. Carry on by default; disable carry for authoritative full hints. First-frame hints use coverage from alpha if present. Full hint directories align to selected source start. Full video hint frame count is checked when reported, and reported average FPS must match the source (relative tolerance 1e-4 or absolute 0.001 fps); missing FPS is rejected. OpenCV average FPS does not establish variable-rate timestamp equivalence.
- Stage 2 batch and foundation chunk >0. Encoder/MLP/window/query chunks >=0; zero disables that chunking. Core shape quantum defaults to 128, accepts positive multiples of 64.
- RAM foundation cache budget is a nonnegative integer MiB; default off. Explicit low-VRAM plus RAM cache is rejected because low-VRAM already uses disk features. An automatic switch to low-VRAM records effective cache budget 0 and a warning. Cache does not change RVM chronology or skip native-rate keying stages.
- Automatic VRAM budget is null or a positive integer MiB and requires automatic selection. It caps a heuristic selection budget, not allocations. Oversized process allocations/no estimated fitting profile cause an actionable error. See `feature-cache.md` for precise budget scope and unmeasured GPU acceptance.
- Quick Key defaults off. Sparse mode uses contiguous native contexts, full-rate RVM, supplied cut/reset boundaries, current source RGB and explicit uncertainty fallback. Contexts must fit the effective Hann chunk. Cut/reset values must be nonnegative integer frame indices within the selected clip and require sparse Quick Key; interval 1 retains the original full-rate path. Automatic cuts and adaptive/max-gap/hold/causal modes are implemented; authored matte/keep/remove/reset documents are implemented with explicit static-range and conditioning limits. Automatic cuts, maximum gap, hold/causal and adaptive controls require sparse Quick Key; interval 1 retains the baseline defaults. Adaptive requires explicit neural fallback and is rejected for hold. Causal contexts are past-only/anchor-owned and omit final anchors requiring future boundary knowledge; RVM chunks are one frame. Zero propagation look-ahead does not make whole-clip precompute live. See `quick-key.md` for approximation, diagnostics and cost limits.
- Model scale currently accepts only 1.0. Other values previously failed after expensive inference; they now fail before loading the model. Reduced-resolution/Lanczos output rendering is deferred.
- num_frames <=0 retains the legacy all-frames convention; positive limits cap the clip. Video start modes are begin/middle/random_middle with nonnegative seed. Frame-directory middle/random and brightness/contrast changes were silently ignored; they now fail with actionable errors.
- FPS finite and >0; brightness and contrast finite/nonzero so inversion is defined (signed values remain accepted); contrast pivot finite. Video adjustments occur in linear light and are inverted at rendering. Frame-directory linear mode is explicit (CLI default off); no transfer inference from extension.
- Outputs: nonempty subset alpha/fg/checker/cutout. Video or EXR only; EXR half/float. CLI defaults alpha+fg; interactive defaults all four. Preview RGB clips to [0,1]; cutout video is ProRes 4444 regardless of ffmpeg_codec/pix_fmt. Cutout defaults to straight RGB/alpha; explicit premultiplied association is supported for EXR cutout only and labelled in the filename/contracts. EXR is linear regardless of cutout_linear. Half saturation outside finite float16 range is inherited behavior.
- Decoded uint8/uint16 image RGB/grayscale normalizes to float32 without eight-bit quantization; float EXR preserves signed HDR in explicit linear mode. Source/foundation model conversion remains bfloat16. Decoded precision observations are recorded per successful frame read, not original EXR pixel-type claims. Video remains an explicitly reported OpenCV uint8 path without source-depth/PQ/HLG detection; higher-bit video and managed source-colour transforms remain pending.
- ffmpeg codec/preset/pix_fmt/bitrate strings are passed to FFmpeg for validation. crf is integer codec-dependent; thread count is clamped to >=1 by the writer. EXR writing no longer requires FFmpeg. Writer queue/finalization waits default to a finite 60 seconds, configurable with `writer_timeout_seconds` (CLI) or `WriterConfig.timeout_seconds` (API/batch). Stalled encoders are terminated/reaped. Native in-process EXR/decode calls cannot be killed by a Python token; partial outputs can remain.

## Interactive launcher and drag/drop settings

Single and batch workflows keep companion hint discovery, Windows-path mapping, video/frame-directory resolution and per-clip output folders. Launcher flags and file names are retained. Interactive execution now calls the same CLI parser in process and retains one model session across clips; failed inference resets the model. Models remain resident while the interactive session is open and are released on exit. No prompt or output default was removed.

| Setting | Default / choices |
|---|---|
| Input | Drag/drop or entered path; video, sequence or batch folder |
| Hint mode | No hint; discovered companion offered; first-frame/full-video/directory choices |
| Hint trust | Medium 0.75; high 0.95; custom [0,1] |
| Authoritative full hints | Off; enabling disables predicted carry |
| Carry quality | 0.95, editable [0,1] when enabled |
| Outputs | alpha, fg, cutout, checker; numbered custom selection |
| Cutout | Straight sRGB default; straight linear alternative |
| Despill | 0.5, custom [0,1] |
| Image input gamma | Linear default, sRGB alternative (differs from CLI) |
| GPU profile | Auto default; manual low-VRAM toggle (off) |
| Hann manual presets | Default 48/24; 24/12, 8/4, 80/40, custom (48/24 prompt defaults) |
| Temp directory | system temp/corridorkey_v2; optional custom in low-VRAM |
| Advanced settings | Off; custom output parent, all frames default, optional limit and begin/middle/random_middle |
| Run confirmation | Existing launcher prompt, default yes |
| Repeat | Reuse settings for another clip, per-clip hints |

CLI-only controls retain defaults above: weights, seed, frame FPS, scale, linear adjustments, EXR format/type, FFmpeg settings, Stage 2 batch, Quick Key, corrections, association/refinement/screen-aware despill and detailed low-VRAM chunks. Launcher does not expose them.

The headless batch schema uses core configuration field names (`carry_hint`, `outputs`, `auto_vram`, etc.), not CLI flag spellings. Core defaults are 80/40 with auto off; choose `auto_vram: true` explicitly when desired. See `docs/sessions.md`. JSON booleans/numbers are validated; the string `"false"` cannot silently enable a memory mode.

Correction schema/content checks run before model load; geometry/range/FPS/mask checks precede processing stages. Native masks use explicit ordered float32 matte/keep/remove operations; only both-direction entries condition existing hints/carry/flow. Forward/backward ranges are direct static raster constraints, not reverse neural inference. Reset/shot boundaries segment RVM and Stage 1 in both full-rate and draft modes. Active correction/reset frames become Quick Key anchors. Full dependent authored shots invalidate on edits; current jobs recompute instead of reusing output caches. See `corrections.md`.

Colour, alpha-association, invalid-value and optional filter/despill contracts are detailed in `colour.md`. Alpha remains coverage, independent of RGB transfer. Native correction masks are applied after optional spatial adjustment. Input/output NaN/Inf and decoding overflow fail rather than being repaired by preview clipping. Existing finite-data defaults are retained.


Opt-in recovery is a separate entrypoint: `python -m corridorkey.recovery run --batch PATH --workspace PATH --max-cache-bytes N [--resume] [--durable-foundations] [--durable-rvm] [--durable-hann]`, plus read-only `status --workspace PATH`. Existing inference options/defaults are unchanged. Recovery currently requires EXR and explicit scheduling (`auto_vram=false`), validates before backend setup and performs whole-job replay/restart with optional durable IID frames, exact chronological RVM chunk/state reuse and Hann event replay; Stage2/output processing restarts. Cache limits refer to managed logical sealed bundle bytes; journals/render copies/physical allocation are outside that limit. See `recovery.md` for verified completion, model identity and preserved output-path semantics.


The separate worker entrypoint is `python -m corridorkey.worker run --batch PATH --report PATH [--capacity N]`. `--workspace PATH --max-cache-bytes N` selects the existing recoverable EXR controller; `--resume`, `--durable-foundations`, `--durable-rvm` and `--durable-hann` require that workspace. Low-level `serve --capacity N [--weights PATH]` is for owned framed-pipe clients. Capacity1..64 counts waiting jobs; one job executes, and the CLI submits sequentially. No new inference/output default or backend fallback. See `worker.md` for model lifetime, cancellation, bounded events, explicit restart and cleanup uncertainty.


Optional `python -m corridorkey.desktop [--project PATH]` opens the Qt workspace. Its advanced settings use existing core/writer names and preserve all fields in validated job snapshots/projects. Frame-limit0 maps to existing all-frames semantics; start modes retain frame-directory restrictions. Full/draft buttons explicitly select documented policies; preview frame is independent of render range. Each render creates a new UUID child under the chosen output parent. Existing CLI/launcher defaults are unchanged. Qt is optional via `requirements-desktop.txt`; see `desktop.md` for working actions and remaining UI/packaging work.


Desktop **Developer benchmarks…** selects existing benchmark manifests/report parents, runs the shared CLI under exclusive local processing ownership, cancels owned work, inspects actual reports/baselines and exports exact bytes to new JSON files. Profiles/repetitions/model-state/quality remain manifest-defined. No inference default or public backend was added. CPU reports now distinguish complete/failed/cancelled and publish null aggregate times unless all required repetitions succeed; CLI cancellation returns130. See `developer-benchmark.md`.


Desktop **Corrections…** now provides native binary painting/eraser, exact soft mask import, core operation/strength/direction/half-open range, reset/shot/order/delete, bounded committed undo/redo and immutable recovery/save/attach. Existing `--corrections`/core fields and equations remain unchanged. Strict correction JSON read/save admission is4MiB; duplicate/nonfinite values reject. `CorrectionHistory(max_snapshots=N)` is optional/default unlimited, while editor chooses128 live snapshots and64Mi-pixel native admission. See `correction-editor.md` for limits, source/timeline binding and unchanged queued manifests.


Desktop **Queue history…** opts into durable observations for future jobs; Open history/Load settings never resubmit, and saved result/PID metadata is unverified history. **Recovery…** uses actual existing EXR/explicit-scheduling batch semantics, exact cache bytes (0 disables bundles), independent durable IID/RVM/Hann flags and explicit verify/resume. Canonical queue snapshots preserve settings despite original manifest edits. Headless batch input now rejects duplicate/nonfinite/boolean-version JSON and raw files above4MiB/128-depth; valid aliases/settings/relative paths are retained. See `durable-queue.md`.


Milestone22: **Common job settings…** exposes existing core/writer options with shared validation, exact numeric text, advanced-field preservation and explicit stale-edit rejection; defaults are unchanged. **Prepare current job for recovery…** requires deliberate EXR/explicit scheduling and creates a new immutable batch recipe without submission. **Local assets…** selects an existing weights/vfms tree plus optional runtime override, performs metadata/backend checks, and requires a clean stopped worker before Use. Worker `serve --asset-root PATH` and optional WorkerClient/InferenceSession `asset_root` keep bundled code fixed; explicit batch weights override worker defaults, omitted weights inherit them. Selection is app-session scoped and not stored in desktop projects. See [desktop settings](desktop-settings.md).


Milestone23: Queue history **Older page/Newer page** traverse bounded128-row admission-order pages; Refresh returns to latest and navigation never submits. Existing4096-record retention/admission remains. Desktop **Resources…** explicitly samples worker process peak RSS, existing CUDA allocator counters and containing-filesystem free space; unavailable fields remain null/NOT VERIFIED. Status requests add a resources record without loading Torch or initializing CUDA. No profiling, processing, precision, output or memory-allocation default changes. See [resource snapshots](resource-snapshots.md).


Milestone24: Queue history **Archive stopped history** seals the selected stopped workspace for managed writes; **Archive + fresh queue** creates/activates a private sibling without submitting work. **Open last archive** retains read-only paging/settings inspection. Every writer enforces seal state after ownership acquisition. No records/media/outputs are deleted or automatically retried; quotas, deletion retention and portable backup remain unfinished. Main close now refuses a dirty correction draft before stopping owners. See [queue archives](queue-archives.md).

Milestone25: optional core/CLI `start_frame` and desktop **Source In/Out (exclusive)** support explicit video/sequence selections. Null preserves existing video start modes; nonzero explicit starts align hints and correction binding. **Selected render as synchronized A** uses observed source offsets for paired views. Viewer position alone still does not change render range; Mark and Apply are explicit. See [source ranges](source-ranges.md).

Milestone26 adds separate developer commands `python -m corridorkey.desktop_bundle check|build --output-parent PATH` and `python -m corridorkey.desktop_smoke --source|--executable PATH --output-parent PATH`. They do not change inference options/defaults or install/download dependencies. Frozen application children use explicit supported module dispatch; source launches retain Python module semantics. See [packaging](packaging.md).

Milestone27: separate `python -m corridorkey.ofx_cache --record PATH --output-parent PATH` prepares a fresh checksummed native RGBA cache from completed scene-linear foreground/alpha EXRs. It never renders, changes defaults, deletes outputs or performs fallback inference. Source-RGB drafts retain a visible cache filename label. Native CacheFile/CacheStart controls explicitly select prepared bytes and integer host-time mapping; unsupported/missing data fails. See [OFX cache](ofx-cache.md).

Milestone28: desktop **Prepare OFX cache…** acts on one highlighted completed queue render, snapshots its actual record/range and prepares a fresh cache in an explicitly chosen existing folder. Prepare/Cancel/Copy path affect real owned I/O; no render is submitted. Missing/invalid selections disable old preparation. Full-quality inference defaults, CLI options and cache format remain unchanged; host source/time/colour matching is still manual. See [OFX cache](ofx-cache.md).


Opt-in `--source_witness` / core `source_witness=false` records exact consumed source pixels for native version2 binding. Requires scene-linear three-channel RGB frame directory, EXR output and at most16384 selected frames; unsupported layouts/video fail explicitly. Common job settings exposes the same default-off control. Sealing old records still creates manual version1 caches; no posthoc source-witness upgrade. See ofx-cache.md. Developer benchmark manifests may use kind=inference_pair through the new benchmark_pair CLI and existing panel; manual scheduling/matched effective settings required. See benchmark-pairs.md.


Manage recovery cache… adds explicit read-only inspection and one-time sealed-byte trim. Target0..9223372036854775807 (0 removesall eligiblebundles); preview required, editsinvalidate, active/stale plansrefuse, originalexports/journals/reports/stagingpreserved. CLI cache_maintenance inspect/plan/apply mirrors core; future recovery budgets and all inference defaults unchanged. This is apparent sealed-entry accounting, not total resource reservation. See cache-management.md.


Recovery cache management now offers Backup / restore… with explicit operation, source and existing destination parent; every operation creates a unique child. CLI module corridorkey.cache_backup has backup --workspace --output-parent, inspect --backup, restore --backup --output-parent. No persistent settings/default paths or automatic submissions. Use restored workspace selects inventory only. Partial destinations remain visible; no overwrite, source-media copying or cross-machine path rebinding. Limits and actual local acceptance: cache-backup.md.


Queue history adds Export / import history…: explicit archived source/exported container and existing parent, unique destination child. CLI corridorkey.queue_transfer export --workspace --output-parent, inspect --archive, import --archive --output-parent. Use imported history opens read-only; Load settings does not submit. Existing archive/freshqueue controls retained. Same-native-path-flavour transfer only; external media/assets/outputs remain historical paths. Bounds/partial outcomes: queue-transfer.md.


Local assets adds optional Audit selected files… → Audit selected files / Save report… / Cancel operation. Existing presence/backend Check and Use stay unchanged; auditing doesnot selectassets. CLI corridorkey.asset_audit audit --asset-root PATH [--weights PATH] [--output NEWFILE]. Complete meansscanfinished, notallformatsPASS; modelcompatibility/authenticity NOTVERIFIED. Reportexport revalidates sources, neveroverwrites, shows actualsaved/partialpath. Defaults/bounds/unsupportedformats: asset-audit.md.

Local assets → Download model assets opens an offline catalog. Resolve online plan explicitly requests metadata; Download reviewed plan requires an existing parent and exact integer payload budget0–9223372036854775807 (default0). Plan payload is capped at1TiB; download needs a covering budget. Each attempt creates a unique tree; Select downloaded paths only fills fields and clears prior Check. CLI: corridorkey.asset_download catalog|plan(alias resolve)[--output NEWFILE]|download --plan FILE --output-parent EXISTING --max-bytes N [--protect PATH repeated]. No automatic retry/resume/Use/inference. Legacy scripts/download_weights.py preserved. See asset-downloads.md.

Managed scratch ceiling: --temp_map_budget_bytes N / config temp_map_budget_bytes / Common job settings Engine exactbytesfield. DefaultNone/blank unlimited;0 no writablemap payload; max2^63-1. Exact effectivejob plan refuses before maps/writers, excludes outputs/cache/filesystemoverhead/wholeRAMVRAM. Existing autoVRAM/profile/cache defaults unchanged. See scratch-budget.md.

Sampled memory stop: --process_tree_rss_stop_bytes N / sameconfigfield / Common job settings Worker tree RSS stop bytes. None/blank/defaultoff; exact0..2^63-1,0literal. Currentroot/trackeddescendant summedRSS checkedcooperatively (~250ms throttle plusforcedmodel/completionchecks), unavailablemeasurement fails. Sharedouterrecoveryguard coversidentity/hash/verifiedexisting/restore/publish; workspaceinit/teardownoutsideperjob. ExcludesGUI/Resolveoutsideworker/GPU, maydoublecountsharedpages/missspikes/overshootnativecalls; nothardRAMcap. See memory-stop.md.

Exact Stage2 replay: recovery run/worker run --durable-stage2 / recoverypayload durable_stage2 / Recovery checkboxDurable Stage2 predictions. DefaultFalse, independentofotherdurableflags; repeatonexplicitVerify/resume, noautoactivation. Sameparent/exactorderedbatch/inputbits/dtype only; perentrytensor256MiB/header64KiB and metadata4MiB/entryfiles4096 bounds mayrejectoptionalcache work. Whole retainedcachequota shared,0skipsbeforehash/copy. Snapshotdurable_stage2 reportactualcompletedforwardcalls/frames vsreplayedinstances and completedpredictions; source/state reconstruction/render/writers stillrun in freshdestinations. See stage2-recovery.md.


Evaluation dossier developer workflow: `python -m corridorkey.evaluation_dossier build --manifest FILE` or `export --manifest FILE --output-parent EXISTING`. Optional `--include-review-media --max-media-bytes N` explicitly admits exact raw media copying; default false/0 emits a metadata template with review pending. Integer budget range 0..9223372036854775807; no overwrite, automatic export, model execution or upload. Manifest defaults: case `required=true`, `review_pair_index=0`; policy relative regression ceiling0.05, zero-baseline absolute tolerances empty/unassessed. Details and interpretation limits: [evaluation-dossiers.md](evaluation-dossiers.md).

Existing `--profile_stages` / desktop Profile stages now also enables [bounded detailed output telemetry](output-telemetry.md), default off. No new inference flag; exact frame/in-flight admission100000/4096 are internal observation limits. CLI `--diagnostics FILE` retains the report. Video encoded/presented latency remains unavailable.

### Declared sequence primaries

`--source_colour_profile` defaults to `legacy`; explicit `linear_bt709_d65` or `linear_bt2020_d65` require `--frame_dir_linear` and image sequences. BT.2020 linear RGB converts into the existing assumed BT.709/D65 working/output basis; alpha is unchanged. No video/source-witness support or automatic tag/transfer detection. Desktop and interactive choices use the same shared setting; the original interactive linear/sRGB choices remain1/2. See [scope and limitations](source-colour-profiles.md).

Durable EXR writer prefix (M42 locally verified): recovery/worker `--durable-writer-prefix`, wire `durable_writer_prefix`, desktop recovery checkbox; default False. Requires EXR and an explicit workspace/cache budget. Explicit resume copies only verified contiguous all-output frame groups into fresh attempts; all chronological processing still runs and only copied-prefix native writes are suppressed. Capacity stops further capture; full workspaces refuse new attempts before mutation. See [writer-prefix recovery](writer-prefix-recovery.md).

Desktop **Diagnostic ranges…** inspects the selected completed/failed/cancelled normal render with observed source metadata. Suggested ranges/events use absolute source In/Out; **Preview source In** revalidates captured inputs. **Prepare range settings** reserves a fresh folder and loads future settings without enqueueing. **Full-rate neural processing** defaults checked; unchecking explicitly preserves/rebases Quick Key. Correction and first-hint policies default Reject; Drop is explicit for the new job. Bounds/coverage/state-reset/provenance limits: [diagnostic ranges](diagnostic-ranges.md). No new inference flags or automatic rerenders.


Trimap import adds an explicit correction kind in document schema 2, with native
background/unknown/foreground classes. It shares existing strength, direction,
range, ordering, save/reopen and immutable attachment semantics. No extra neural
input or new inference flag is introduced; see [trimap refinement](trimap-refinement.md).

Output safety admission rejects generated paths that collide with input media,
hints, correction documents or their assets, including observed link aliases.
It also refuses output roots inside protected source/hint directories. Select a
separate output folder in that case. Existing unrelated outputs can still be
rerendered. Collision inspection is cancellable and limited to 100000 direct
entries and 1024 path ancestors; it never recursively audits arbitrary trees.
Source CLI/API callers receive this protection through the shared engine.


Queue history reads require UTF-8 SQLite databases and preserve independent4MiB
request/event limits. Oversized or ill-typed persisted fields are refused before
Python text materialization; existing histories are never transcoded. This does
not impose a hard cap on SQLite internal memory.

Durable recovery workspace/owner records are limited to4MiB. RVM/Hann adapters
also enforce4MiB metadata,64KiB safetensors headers and geometry/dtype-derived
payload bounds within the configured cache budget. Invalid cache data recomputes;
unverifiable ownership fails closed before pruning. Generic completed manifests
retain the configured store quota rather than the tighter typed-adapter limit.
No new inference flag or precision change is introduced.


Desktop **Readiness / first render…** opens without automatic checks or processing.
**Check readiness** inspects local metadata off-thread. **Prepare short render**
uses explicit absolute In and 1–120 frames (default8) in a fresh unique output child;
**Run displayed snapshot** submits only after review. Quality/backend/precision
remain requested; supplied cuts/resets rebase, moved first hints and correction
subranges refuse rather than silently change meaning. No new inference defaults
or CLI aliases. See [scope and pending acceptance](product-improvements.md).

Desktop **Play sequence / Pause** requires a selected completed source image
sequence/EXR render with recorded declared timing. Optional loop source In/Out
are exclusive ranges; blank uses the common range. Speeds¼×–4× affect preview
scheduling only. Source frame/time and buffering/drop counters are shown. Manual
preview/navigation/view/selection/settings changes pause playback. Video/VFR
explicitly refuses; 50ms UI polling and per-frame reader reopening are documented
limits. No inference/export setting or persisted project schema changes.


First launch now shows the readiness introduction once (no automatic checks or
render). Desktop `--readiness` forces the introduction; `--no-first-run` suppresses
it for this invocation. Explicit **Probe backend / encoders** runs bounded
subprocess metadata queries. **Inspect completed frame** decodes one matching
completed Alpha/Foreground/composite frame, at most512px thumbnail, with separate
decode/display evidence. It invalidates any newly prepared render whose displayed
details it replaces. No inference defaults/aliases or queued settings change.
See [workflow, bounds and unverified acceptance](readiness.md).


The readiness report retains matching failed/cancelled worker metadata and the
last attempted checkpoint, copied before cleanup. Missing/contradictory metadata
remains NOT VERIFIED; partial CUDA work never implies success. A/B preview
surfaces share decoded-pixel zoom and normalized pan with wheel/drag and explicit
Zoom/Fit controls. Display transformations preserve render settings and pixels;
scale refers to bounded decoded previews, not native source resolution. No
inference default, precision, dependency or saved-format change is introduced.


Tracked corrections use explicit Preview/Inspect/Accept in the existing editor,
with forward/backward/both range, artist anchors and request_anchor/stop/explicit
retain_approximation policies. Persisted schema3 final constraints preserve old
schema1/2 semantics; no new inference default. Video/trimap tracking and crossing
cuts/resets are unavailable. Limits, compatibility and validation:
[tracked corrections](tracked-corrections.md). Source-only sequence Play prepares
a bounded index off-thread; Refresh source index reinspects membership explicitly.
Declared sequence FPS remains separate from observed video timestamps.


Correction editor **Clean plate** is collapsed by default. Explicit reference
colour, source/reference index, background sample mask or whole-frame assumption,
alignment/translation, exposure and BG/FG thresholds affect Preview. Guidance
inspection is required before Accept; diagnostic views remain read-only. **New
mask** reloads source pixels before painting after plate inspection or frame
changes. Schema4 adds generation provenance without changing existing schema1–3
behavior, inference flags or quality defaults. Bounds, detach semantics, formats
and validation: [clean-plate guidance](clean-plate-guidance.md). The optional SAM2
provider foundation is headless; there are no nonfunctional model controls.


Correction editor **Subject clicks** is collapsed by default. **Prepare selected
frame** captures the source with explicit local config/checkpoint; **Predict**
starts the optional isolated CPU float32 provider only on request. Click mode,
positive/negative native XY, Undo click, Clear selection, Inspect, **Accept as
Keep** and Discard affect the captured selection. Keep at strength1 makes its
interior opaque and leaves outside unchanged; existing strength/range controls
apply after acceptance. Limits, cancellation and pending optional-model/frozen
platform acceptance: [subject clicks](subject-clicks.md). No render defaults,
precision aliases, queued requests or dependencies change.


Desktop **Projects…** groups **Open desktop snapshot…**, **Save desktop snapshot…**
and **Portable projects…**. Existing schema1 format and CLI/API remain; Save now
requires readable correction metadata to protect every referenced asset/alias.
Portable collection defaults to reference-only, verification budget256MiB and
copy budget0; dialog limits0–65536MiB are explicit per operation/pass. Explicit
Verify/Publish/Inspect/Relink/Apply never enqueue or change the active model owner.
Apply requires separate output/scratch; desktop preview frame maximum stays1e9
and larger portable frame values refuse rather than clamp. Native path flavour
only initially; no model/cache/environment collection or compatibility claim.
See [bounds, provenance and validation](portable-projects.md).

Desktop **Progressive preview…** adds an explicitly enabled, transient inference
workflow; default off and opening performs no I/O/model work. Capture/start uses
the current job, explicitly selected scratch and Alpha/Foreground/composite view.
Optional draft defaults unchecked, scale0.5 (must be below captured model_scale);
idle surrounding refinement defaults checked. Core scheduler limits and workspace
bounds are fixed explicit admission policies, not hardware memory caps. Normal
exports retain their settings and disable previews. Worker protocol `preview`
accepts the same strict job wire as `render`, shares the same sequential session,
and is excluded from durable render history. No existing CLI flag, inference
precision/colour/backend default or persisted format changes. Initial sequences
only, no Quick Key/hints/corrections; single-frame temporal state restarts.
[Workflow and verification limits](progressive-previews.md).

Correction editor **Show spatial tracking diagnostics** is collapsed by default;
**Capture diagnostic maps on next Preview track** defaults false. Opt-in capture
reuses existing flow and adds a separate64MiB map/receipt admission budget to the
existing512MiB tracking-mask budget. Explicit reliability/uncertainty inspection
and staged opacity (default0.55) affect source-display pixels only. Navigation,
captured range and new-correction actions use existing safeguards/source reload.
No CLI/inference flag, correction schema, persisted track semantics or render
pixel/default change. Saved-track maps remain unavailable. Limits and commands:
[spatial tracking diagnostics](tracking-overlays.md).


## Optional machine calibration

Calibrate performance… opens an explicit metadata → budgeted Verify → Start workflow.
No render defaults or aliases change. Profiles cover scale, temporal window/stride,
Stage 2 batching, explicit memory/cache and Quick Key settings; recommendations
never change jobs. Sequence-only; hints/corrections/video and auto-VRAM/stage
profiling are explicitly refused. See [workflow, limits and verification](calibration.md).


## Foreign portable collection paths

The all-collected Windows/POSIX schema2 reader extension is locally verified. It preserves original archival settings and requires explicit native
absolute output/scratch destinations for foreign Apply; native package destination
behavior is unchanged. Partial/reference-only foreign packages and direct foreign
publication refuse. EXR retains unused encoder references; foreign absolute video
encoders refuse without replacement. See [scope and review](portable-foreign-review.md).
