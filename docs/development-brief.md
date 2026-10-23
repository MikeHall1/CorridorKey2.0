Work in this repository:

https://github.com/MikeHall1/CorridorKey2.0

Upstream reference:

https://github.com/nikopueringer/CorridorKey2.0

Develop this fork into a faster, more capable AI keying engine, with a standalone application and an eventual DaVinci Resolve plugin sharing the same processing core.

This is an implementation project. Inspect the repository, establish a baseline, create a staged implementation plan, and start delivering working, tested changes. Do not stop after producing a roadmap.

Read this entire specification before making architectural decisions.

**1. Overall objectives**

The main aims are:

- Improve performance toward real-time frame rendering.
- Improve output toward professional-level keying.
- Integrate benchmarking into the tool, with its interface behind an optional developer mode.
- Support optional manual corrections and correction states at selected frames.
- Use GPU compute, VRAM, system RAM, and temporary storage more efficiently.
- Preserve all existing options and workflows, while adding useful new controls.
- Introduce a Quick Key mode for fast visualisation.
- Support a standalone desktop application.
- Prepare a shared engine suitable for a future DaVinci Resolve OpenFX integration.

Treat real-time performance and professional quality as measured engineering targets. Do not claim either has been achieved without suitable evidence.

The existing model is the starting point. A full rewrite or model retraining is not a prerequisite.

**2. Initial repository investigation**

Before changing behaviour:

- Read any existing AGENTS.md instructions.
- Inspect the current branch, working tree, dependencies, licence, third-party notices, and available tests.
- Record the actual starting commit.
- Preserve existing user changes.
- Determine which components can run in the available environment.
- Check whether CUDA, a suitable GPU, FFmpeg, model weights, and reference footage are available.
- Inspect the current pipeline end to end, including model loading, preprocessing, foundation extraction, recurrent state, temporal windows, blending, final prediction, rendering, and output writing.
- Inventory every CLI option and interactive-launcher setting.
- Identify and document defaults, aliases, valid ranges, and output-affecting settings.

A previous review examined upstream commit:

485f5e9ab759ead6138cc990776184b59d6ea971

That review identified potential opportunities and defects listed below. Revalidate them against the actual checkout.

Do not assume documents or implementation files discussed in an earlier chat exist in this fork. Create the necessary project documentation here.

**3. Preserve existing capabilities**

Retain the existing functionality, including:

- Command-line inference.
- Interactive and drag-and-drop workflows.
- Video and frame-directory inputs.
- Native-resolution processing.
- No-hint inference.
- First-frame hints.
- Full hint videos and hint directories.
- Hint quality and carried-hint controls.
- Temporal window and stride controls.
- Automatic GPU-memory selection.
- Low-VRAM processing.
- Stage 2 batching.
- Partial-clip processing.
- Existing brightness, contrast, and scaling options.
- Alpha, foreground, checkerboard, and cutout outputs.
- Existing output aliases.
- Video encoding controls.
- Linear-input handling.
- HDR EXR exports and precision controls.

Where an option is broken or misleading, repair it or reject unsupported use early with an actionable message.

Do not silently remove options, change colour interpretation, downgrade output precision, or enable approximations.

If an option must change, provide a documented mapping or compatibility layer.

New draft modes must be explicitly selectable and must not silently replace full-quality processing.

**4. Architecture**

Keep the processing engine independent of the GUI, CLI, benchmark interface, and Resolve SDK.

Prefer a reusable architecture with:

- Validated job configuration.
- Model lifecycle management.
- Device and memory budgeting.
- Source readers.
- Preprocessing.
- Foundation-feature extraction.
- Temporal scheduling.
- Inference.
- Optional refinement.
- Output rendering and writing.
- Cache management.
- Progress and cancellation.
- Diagnostics and benchmark instrumentation.
- Serializable manual corrections.
- Stable output and error contracts.

The CLI, desktop interface, benchmark runner, and future OFX adapter should call the same core.

Avoid duplicating inference implementations across interfaces.

Start by retaining the Python/PyTorch model. Most expensive operations already execute in native GPU code.

Consider C++ for the eventual OFX adapter and selected inference components where measured benefits justify it. Consider Rust for application infrastructure only if it provides a concrete advantage.

Do not introduce multiple languages merely to make the architecture appear more sophisticated.

Keep dependencies and module boundaries proportionate to the project.

**5. Define real-time performance correctly**

Report these separately:

- Application startup and model loading.
- Warm time to first usable frame.
- Sustained uncached inference throughput.
- End-to-end throughput including decode and output.
- Interactive request latency.
- Initial buffering.
- Look-ahead latency.
- Cached playback.
- Causal live processing.

Cached playback is not evidence of real-time uncached inference.

Offline throughput above the source frame rate is not evidence of low-latency live processing.

The target machine has not yet been specified. Make targets configurable. Use 1080p at 24 fps as an initial benchmark case, not an agreed hardware-independent guarantee.

Also support benchmark cases for:

- 1080p30.
- 1080p60.
- 1440p.
- 4K24 and 4K30 where hardware permits.

For every performance claim, name the hardware, resolution, frame rate, quality settings, precision, hint mode, output format, and cache state.

**6. Performance improvement backlog**

Investigate all 20 opportunities below. Prioritise using actual profiling, expected value, implementation effort, and output risk.

1. Cache C-RADIO and MoGe features for frames shared between overlapping windows.
2. Keep Stage 1 feature blending on the GPU when memory permits.
3. Keep models loaded across clips using a persistent worker.
4. Share decoded frames between compatible stages through bounded caches.
5. Evaluate GPU preprocessing, including colour conversion, padding, normalisation, and resizing.
6. Overlap transfers and computation where useful, using pinned memory and appropriate stream/event coordination.
7. Reuse buffers for stable input shapes.
8. Evaluate inference_mode in compatible sections already using no_grad.
9. Tune temporal windows for throughput, quality, and latency separately.
10. Benchmark and tune Stage 2 batch sizes per device and resolution.
11. Evaluate 16-bit foundation-feature caches instead of float32.
12. Select GPU, RAM, or disk caching according to explicit budgets.
13. Combine compatible prepasses while preserving chronological recurrent-state behaviour.
14. Evaluate compilation of selected components.
15. Evaluate exporting compatible model stages to an optimised inference engine.
16. Evaluate hardware video decoding with colour and precision validation.
17. Offer hardware-accelerated preview encoding where appropriate.
18. Skip rendering and preprocessing work that selected outputs do not require.
19. Implement a clearly labelled reduced-resolution draft path, repairing existing scaling limitations.
20. Support resumable processing and verified intermediate-cache reuse.

Important constraints:

- Do not assume larger batches or larger temporal windows are faster.
- Do not assume compilation or export will support every operation.
- Account for compilation and engine-build time separately from warm execution.
- Preserve the reference colour and resizing contract when comparing implementations.
- Validate lower-precision caches against suitable reference outputs.
- Do not assume non_blocking transfers overlap computation without verifying the required conditions.
- Keep caches bounded and evict safely.
- Avoid excessive GPU allocation alongside Resolve or other applications.

Start with the simplest high-value changes, particularly feature reuse, persistent model loading, unnecessary-work removal, and transfer reduction.

**7. Quick Key**

Implement an optional draft mode that performs expensive keying on selected anchor frames and estimates mattes for intervening frames.

The purpose is quick visualisation and responsive previews. It must not be represented as equivalent to full-rate neural inference.

For interval N, regular anchors are:

0, N, 2N, 3N, and so on.

For example, interval 5 means frames 0, 5, 10, 15, rather than 0, 4, 8.

Include the necessary final anchor for each shot.

Support three explicitly named approaches:

- Hold-last-matte: a cheap diagnostic baseline.
- Offline motion-guided interpolation: use previous and next anchors.
- Causal motion-guided propagation: use only available past information.

Implement the offline prototype first unless profiling or environment constraints justify a different sequence.

Offline interpolation should:

- Estimate motion between relevant source frames.
- Warp surrounding anchor mattes toward the requested frame.
- Blend using temporal distance and propagation reliability.
- Handle uncertain regions conservatively.
- Avoid blindly crossfading unaligned mattes.

Causal propagation should:

- Track forward from the latest available anchor.
- Report drift and uncertainty.
- Request a new anchor when necessary.
- Expose the tradeoff between inference frequency and responsiveness.

Do not silently describe interpolation using future anchors as a zero-look-ahead live mode.

**8. Quick Key foreground handling**

Start by propagating alpha and combining it with the current source RGB.

Do not freeze the image between anchors.

Do not interpolate complete source frames as a substitute for processing the actual current frame.

Clearly label source-RGB draft output because it is not the same as the model’s reconstructed foreground RGB.

An optional future implementation may propagate reconstructed foreground colour, but it must independently address:

- Ghosting.
- Disocclusions.
- Transparency.
- Motion blur.
- Newly revealed pixels.
- Colour contamination.

For a fair benchmark, separate quality loss from sparse alpha prediction from quality loss caused by using source RGB.

**9. Quick Key temporal semantics and adaptive anchors**

The model was not necessarily trained for arbitrary sparse temporal sampling.

Do not simply remove frames from its input timeline and assume equivalent behaviour.

Compare these strategies:

- Independent single-frame anchor inference.
- Short native-rate context windows around anchors.
- Chronological recurrent foundation extraction across all frames, with sparse expensive keying stages.

Document which strategy is active and its costs.

Preserve RVM chronology and reset state appropriately.

Adaptive anchor insertion should consider:

- Scene cuts.
- User reset points.
- Manual correction frames.
- Fast movement.
- Occlusions and disocclusions.
- Unreliable flow.
- Large disagreement between propagated mattes.
- Maximum permitted anchor gap.

Call propagation reliability a diagnostic unless it has been calibrated as a confidence measure.

Expose:

- Enable/disable.
- Anchor interval.
- Propagation approach.
- Motion-estimation resolution.
- Adaptive anchor toggle.
- Maximum gap.
- Reliability thresholds.
- Fallback policy.
- Foreground mode.
- Diagnostic overlays.

Fallback policies should be explicit:

- Run additional neural inference.
- Return a visibly labelled approximation.
- Stop with an actionable error.

Do not silently violate a requested latency budget.

Interval 1 should route to the existing full-rate path for a meaningful baseline.

Record actual anchor positions, extra inference frames, fallback counts, and motion-processing costs.

Do not promise an N-fold speed increase for interval N.

**10. Quality improvement backlog**

Investigate these 20 improvements:

1. Correction hints at arbitrary frames.
2. Scene-cut detection and temporal-state resets.
3. Foreground/background protection masks.
4. Trimap-assisted refinement or conditioning.
5. Edge-aware alpha refinement.
6. High-resolution boundary refinement.
7. Motion-aware temporal stabilisation.
8. Occlusion-aware temporal handling.
9. Optional bidirectional offline processing.
10. Selective multi-scale inference.
11. Green/blue screen-colour-aware despill.
12. Edge-weighted despill.
13. Foreground colour protection.
14. Spill removal with luminance preservation.
15. Explicit colour management.
16. Higher-precision input paths.
17. Correct alpha association and metadata.
18. Targeted transparent-object refinement.
19. Diagnostics and selective reprocessing of difficult ranges.
20. Fine-tuning on difficult reference material.

Separate improvements possible with the existing model from those requiring a new refinement model, changed conditioning, or training.

Do not add UI controls with no implemented effect.

Do not assume that a smoother or sharper matte is a more accurate matte.

Preserve hair, motion blur, partial transparency, and fine geometry.

**11. Manual correction states**

Implement optional saved corrections at selected frames.

A correction state is a user-authored hint or reset, not a change to model weights.

Design for:

- Matte corrections.
- Keep/remove regions.
- Optional future brush strokes and trimaps.
- Correction strength where meaningfully supported.
- Shot boundaries.
- State-reset markers.
- Propagation direction and range.
- Undo/redo in the eventual UI.
- Versioned persistence.
- Reopening the project.
- Cache invalidation.
- Promoting correction frames to Quick Key anchors.

The model’s existing hint interface may not accept every proposed correction type. Implement the necessary conditioning or refinement behaviour before exposing it.

Changing a correction must invalidate all dependent outputs. With causal recurrence, the affected range may extend to the remainder of the shot.

Do not assume corrections only affect a fixed small neighbourhood.

**12. Colour and output correctness**

Make colour handling explicit.

Distinguish:

- Display-coded sources.
- Scene-linear sources.
- Transfer functions.
- RGB primaries.
- Model working representation.
- Foundation-model SDR proxies.
- Preview outputs.
- Production outputs.

Maintain unclipped HDR where supported.

Avoid silently reducing high-bit-depth inputs to 8-bit.

Preserve negative and above-one linear values where the existing HDR output contract supports them.

Keep alpha coverage separate from RGB transfer functions.

Specify straight versus premultiplied outputs correctly.

Verify composites over black, white, grey, saturated colours, and photographic backgrounds.

Do not treat EXR export alone as proof of correct colour management.

**13. Reliability fixes**

Revalidate and address the previously identified problems:

- Writer threads can exit after an error while producers or shutdown remain blocked on a full queue.
- Temporary filenames derived only from clip basenames can collide between jobs.
- Image sequences may use lexical ordering and case-sensitive extension matching.
- Non-native model scaling may be accepted initially but fail during rendering.
- Cleanup errors may obscure the original processing failure or prevent remaining cleanup.

Also inspect:

- Invalid chunk/stride combinations.
- Empty or damaged sources.
- Hint/source frame alignment.
- Hint alpha-channel handling.
- Cancellation during every processing stage.
- Partial output reporting.
- Encoder failure.
- Disk exhaustion.
- GPU out-of-memory recovery.
- Worker crashes.
- Cache corruption.
- Multiple simultaneous jobs.

Use unique job directories, bounded queues, cancellation-aware operations, explicit worker lifecycle management, and dependable cleanup.

Do not silently return partial or stale output as a successful complete render.

**14. Integrated benchmarking**

Build benchmarking into the engine.

Provide:

- A headless benchmark command.
- A versioned manifest format.
- An optional developer-mode interface using the same runner.
- Baseline-versus-candidate comparisons.
- Exportable reports.

Developer mode is a diagnostic interface, not a security boundary.

Useful controls include:

- Select suite.
- Select baseline.
- Select processing profiles.
- Run.
- Cancel.
- Inspect stage timings.
- Inspect resources.
- Compare output.
- Inspect Quick Key anchors.
- Export results.

Start with an actual runnable command, then integrate the interface once the application shell exists.

Do not create a decorative benchmark screen backed by invented numbers.

**15. Benchmark reproducibility**

Record:

- Commit and dirty-working-tree identity.
- Model and foundation-weight hashes.
- Input, hint, and correction identities.
- Requested and effective settings.
- Seeds.
- Dependencies.
- GPU, VRAM, CPU, RAM, and storage.
- CUDA and driver versions.
- Operating system.
- Relevant power settings.
- Source timing, dimensions, and colour assumptions.
- Output format.
- Cache state and policy.
- Available memory budget.
- Errors and warnings.

Keep source footage local unless the user explicitly authorises uploading it.

Use separate development and held-out evaluation footage.

Record reference provenance and permitted usage.

Missing reference data should produce unavailable metrics, not fake zero errors.

**16. Benchmark suites**

Create a structure supporting:

A. Quick development suite

- Six approximately five-second 1080p clips.
- Short enough to run during iteration.

B. Quality suite

- Approximately 24 clips lasting five to ten seconds.
- Selected 4K variants.
- Hair, fur, motion blur, glass, smoke, reflections, poor lighting, shadows, spill, matching foreground colours, occlusions, camera motion, and cuts.
- Green and blue screens.
- SDR video, image sequences, and linear HDR material.

C. Production stress suite

- A ten-minute clip.
- A batch of 50 short clips.
- Repeated jobs.
- Concurrent isolated jobs.
- Cancellation.
- Disk-full and encoder failures.
- Interrupted-run recovery.
- Memory-growth checks.

D. Future Resolve suite

- Sequential playback.
- Random frame access.
- Reverse scrubbing.
- Multiple plugin instances.
- Trims and retimes.
- Proxy and resolution changes.
- Colour-managed projects.
- Export/viewer agreement.
- Cancellation and worker failure.

Do not download large datasets or fabricate professional reference footage merely to fill the suite. Implement manifests and fixture generators, then clearly identify which real assets are still required.

Synthetic fixtures are useful for correctness, but are not a replacement for real keying evaluation.

**17. Benchmark measurements**

Performance:

- Cold-start time.
- Warm first-frame latency.
- End-to-end FPS.
- Model-only throughput.
- Per-stage timings.
- Frame latency p50, p95, and p99.
- Initial buffering.
- Look-ahead.
- Peak allocated and reserved VRAM.
- Whole-process GPU memory where available.
- Process-tree RAM.
- Temporary storage peak.
- Disk bytes read and written.
- Cache hit rate.
- Success rate.
- Recovery time.
- Batch completion time.

Account for child workers.

Use GPU events or appropriate synchronisation for asynchronous operations.

Do not sum overlapping stage times as if they were wall-clock time.

Separate detailed profiler runs from ordinary throughput measurement.

Quality:

- Normalised alpha MAE/SAD.
- Boundary-region error.
- Alpha gradient error.
- Foreground-colour error over valid regions.
- Composite error against reference.
- Motion-compensated temporal error.
- HDR clipping and invalid-value checks.
- Alpha-association correctness.
- Blinded A/B review.

Document metric definitions, masks, ranges, and units.

Do not compare unnormalised aggregate errors across different resolutions or frame counts.

A temporally frozen matte can score well on naive flicker measurements while being incorrect. Evaluate accuracy and stability together.

Report category-level and worst-case regressions.

**18. Benchmark protocol and acceptance**

For each change:

1. Run baseline and candidate on the same hardware and assets.
2. Hold settings constant for a direct comparison.
3. Benchmark automatic selection separately.
4. Separate cold-process and warm-worker conditions.
5. State filesystem and model-cache conditions.
6. Warm up where appropriate.
7. Run at least three measured repetitions.
8. Alternate baseline and candidate order.
9. Report medians and variation.
10. Test individual changes, then the combined implementation.

Run both diagnostic processing without output encoding and realistic end-to-end exports.

Report speedup as baseline elapsed time divided by candidate elapsed time, with raw times included.

Do not add individual speedup multipliers together.

Acceptance categories:

- Behaviour-preserving optimisation.
- Precision/backend change.
- Explicit speed/quality approximation.
- Quality enhancement.
- Reliability fix.

Define numerical tolerances before judging results, using baseline repeatability and intended precision.

For optional approximations, report the measured speed/quality tradeoff.

A result without a hardware run is unmeasured, not passed.

**19. Quick Key benchmark matrix**

Compare:

- Full-rate inference.
- Hold-last-matte.
- Offline motion interpolation.
- Causal propagation.

Test intervals:

1, 2, 3, 5, 8, and 10.

Include:

- Adaptive anchors enabled/disabled.
- Static and fast-moving subjects.
- Occlusions.
- Cuts.
- Fine hair.
- Motion blur.
- Transparency.
- Correction frames.
- Short clips and final partial intervals.

Record actual neural frames evaluated, including context and prepasses.

Show the cost of motion estimation and adaptive fallback.

Separate source-RGB foreground evaluation from reconstructed-foreground evaluation.

Confirm frame count and timing remain correct.

**20. Standalone application**

Once the processing foundation is stable, build a usable desktop interface with:

- Drag-and-drop inputs.
- Source preview.
- Alpha preview.
- Foreground preview.
- Checkerboard and custom-background composites.
- Range selection.
- A/B comparison.
- Draft/full-quality switching.
- Render queue.
- Progress and cancellation.
- Saved settings.
- Correction-state persistence.
- Cache controls.
- GPU and memory status.
- Developer-mode benchmarking.

Choose a practical GUI framework based on maintainability, packaging, and performance.

Load models once per session where appropriate.

Package dependencies so normal users do not need to manage a Python environment manually.

Provide deliberate model-download and asset-validation behaviour.

Keep experimental options in an advanced section with clear descriptions.

**21. DaVinci Resolve integration**

Prepare for a native C++ OpenFX adapter sharing the engine.

The initial target is Windows/NVIDIA with Fusion-page use.

Do not claim macOS support until an Apple GPU backend has been implemented and validated. A language change does not remove CUDA dependencies.

Resolve can request frames out of order. Design around:

- Deterministic temporal window boundaries.
- Shot analysis.
- Cached results.
- Recurrent-state checkpoints.
- Safe recomputation.
- Per-instance state isolation.
- Concurrent requests.
- Cache invalidation.
- GPU memory shared with Resolve.
- Colour-management boundaries.

Never use one mutable previous-frame state as the complete random-access solution.

Investigate host SDK capabilities before promising arbitrary controls, background analysis, or GPU-buffer sharing.

A practical initial workflow is analyse/cache a shot, then serve processed frames through the effect.

Expose foreground RGB and alpha with suitable preview modes. Do not assume an OFX effect can provide arbitrary separate output clips.

Keep benchmarks, downloads, training, and complex asset management in companion tooling where appropriate.

The official CorridorKey project already has a Resolve plugin. This fork’s purpose is enhanced performance, quality, control, and experimentation.

**22. Future research ideas**

Document these as experiments, not guaranteed deliverables:

- Smaller distilled preview models.
- Training specifically for sparse temporal anchors.
- Learned matte propagation.
- Adaptive compute focused on uncertain boundaries.
- ROI refinement that retains sufficient global context.
- Reduced-resolution context with high-resolution edge reconstruction.
- Smarter anchor scheduling.
- Separate interactive and final-render models.
- Scene-level automatic parameter suggestions.
- Reusable subject hints across related shots.
- Carefully validated quantisation.
- Additional GPU backends.
- Apple Silicon support.
- Multi-GPU offline scheduling.
- Compatibility with other OFX hosts.
- Benchmark history and regression dashboards.
- Fine-tuning tools using appropriately licensed training data.
- Optional user-reviewed automatic correction suggestions.

For each experiment, record the hypothesis, expected benefit, dependencies, validation method, and stop criteria.

**23. Documentation and implementation phases**

Create or update:

- AGENTS.md.
- Development brief.
- Architecture overview.
- Existing-option compatibility inventory.
- Benchmark specification.
- Quick Key specification.
- Manual-correction format.
- Performance experiment log.
- Quality evaluation guidance.
- Resolve integration notes.
- Implementation status and next steps.

Keep documents consistent with actual implementation.

Suggested phases:

Phase 1: baseline and reliability

- Repository audit.
- Compatibility inventory.
- Benchmark harness skeleton with real CPU correctness checks.
- Baseline metadata.
- Writer and temporary-file fixes.
- Input/settings validation.

Phase 2: measured efficiency

- Foundation-feature caching.
- Persistent workers.
- Removal of unnecessary work.
- GPU blending and transfer experiments.
- Resource budgets.

Phase 3: Quick Key

- Deterministic anchors.
- Hold-last baseline.
- Motion-guided offline propagation.
- Cut boundaries.
- Adaptive anchors.
- Interval comparisons.
- Explicit draft output labelling.

Phase 4: quality and corrections

- Correction states.
- Despill improvements.
- Edge and temporal refinement.
- Colour and precision work.

Phase 5: application interface

- Standalone GUI.
- Preview and queue.
- Developer benchmark panel.
- Packaging.

Phase 6: Resolve

- OFX prototype.
- Shared-engine connection.
- Analysis/cache workflow.
- Host correctness and resource tests.

Implement coherent increments. Do not attempt all features simultaneously or create disconnected scaffolding with no working path.

**24. Testing and development standards**

- Preserve readable code and explicit interfaces.
- Use focused tests for meaningful behaviour.
- Add deterministic CPU fixtures for scheduling, cuts, cache keys, option compatibility, corrections, and error handling.
- Gate GPU tests appropriately.
- Test CPU/GPU numerical agreement where a relevant reference exists.
- Test temporal ordering and random-access behaviour.
- Test cancellation and cleanup.
- Test final partial windows and short clips.
- Test alpha, RGB, timing, precision, and metadata.
- Avoid brittle tests that only mirror implementation details.
- Never present static inspection as successful model inference.
- Never invent benchmark results.
- Record unavailable tests and their exact prerequisites.
- Preserve upstream attribution, licence, and third-party notices.
- Check applicable licence requirements before packaging or distributing integrations.

**25. How to begin and report progress**

Begin by inspecting the repository and environment, then implement the first useful milestone.

Proceed autonomously on reversible engineering work. Ask only when a missing decision materially blocks progress.

Do not stop at “this is possible” or a design-only response.

If GPU hardware or weights are unavailable:

- Continue with configuration, scheduling, reliability, cache contracts, benchmark infrastructure, documentation, and meaningful CPU tests.
- Clearly mark GPU performance and visual quality as unmeasured.
- Provide exact commands for the remaining validation.

At the end of each milestone, report:

- What changed.
- Why it matters.
- Existing behaviour preserved.
- New options added.
- Tests actually executed.
- Measured results, if any.
- Known limitations.
- Next implementation step.

Start with a trustworthy baseline and working infrastructure, then improve the engine through measured changes.