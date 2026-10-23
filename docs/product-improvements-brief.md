Extend CorridorKey2.0 with the ten product improvements below, integrating them into the existing desktop application, shared processing engine and supported Resolve workflow.

Repository:
`/Users/mike.hall/Documents/vsCode/corridorKey2_fork`

GitHub fork:
`https://github.com/MikeHall1/CorridorKey2.0`

Implement working, connected features. Continue autonomously through a persistent RALPH implementation, review and verification loop. Commit and push meaningful checkpoints to the verified development branch in my fork.

**1. Establish the current state**

Read:

- `AGENTS.md` and applicable nested instructions.
- `docs/development-brief.md`
- Current checkpoint, status, architecture, options and code-health documents.
- Existing acceptance records.
- `.ralph/plan.json`
- `.ralph/issues.json`

Inspect the worktree, recent commits, existing interfaces and verification evidence. Preserve uncommitted work.

The latest reported implementation includes desktop projects and queues, source-aligned A/B inspection, correction editing, trimap refinement, Quick Key, resource controls, recovery, benchmark tooling, diagnostic rerenders and a prepared-cache OpenFX adapter.

Treat that summary as an orientation, not proof. Verify what exists before adding replacements.

The last reported local review passed 985 tests. Do not assume that result covers the current worktree or establishes real GPU performance, visual quality, Windows packaging or Resolve compatibility.

Resume from current repository evidence rather than an assumed milestone number.

**2. Scope and compatibility**

Implement all ten features below. Preserve existing options, saved projects, correction files, full-quality processing, command-line workflows, headless operation, attribution and third-party notices.

Keep the shared engine independent of Qt and Resolve. Desktop and host interfaces should consume shared functionality rather than duplicate it.

Version persisted formats where necessary. Support migration or explicit compatibility errors. Never silently reinterpret existing projects.

New optional dependencies must remain optional. Missing capabilities must produce accurate explanations.

Do not expose controls that have no implemented effect.

**3. Feature: first-run readiness and guided real render**

Add a first-run and on-demand readiness workflow.

Check and distinguish:

- Application and dependency availability.
- Configured FFmpeg capabilities.
- Asset paths, required files and available integrity metadata.
- Selected backend availability.
- Model compatibility and successful loading.
- Available resource information.
- Output destination permissions and collision risks.
- Actual inference.
- Successful output decoding and display.

Provide actionable results and a local diagnostic report.

Separate lightweight checks from expensive model loading and inference. Do not trigger large downloads, installation changes or expensive processing merely by opening the wizard.

Offer an explicit short-range render using user-selected local footage. Show requested settings, expected operations and output destination before execution.

Do not silently switch backend, resolution or processing quality to obtain success.

Acceptance:

- Users can identify the exact failing stage.
- Checks and renders can be cancelled safely.
- Source files remain unchanged.
- Reports distinguish checks, mocked tests and actual inference.
- The workflow integrates with existing job and asset management.

**4. Feature: tracked manual corrections**

Extend corrections so an authored mask can follow the subject through motion.

Support:

- Selecting a correction frame and operation.
- Forward, backward or bidirectional tracking.
- Explicit frame-range limits.
- Motion-guided mask propagation.
- Occlusion and propagation-reliability diagnostics.
- Stop, request another anchor or explicitly retain an approximation when tracking becomes unreliable.
- Preview before committing tracked results.
- Additional artist anchors.
- Undo/redo, save/reopen and reproducibility.
- Cache invalidation for all dependent results.

Reuse suitable Quick Key motion infrastructure where practical, while preserving distinct semantics between tracked corrections and final matting.

Keep original authored masks immutable. Persist tracking configuration, anchor identities and relevant provenance. Store derived results separately.

Do not treat temporal range application of a stationary mask as tracking.

Do not propagate across cuts or explicit reset boundaries without deliberate user action.

Acceptance:

- A moving-object correction follows motion on suitable reference sequences.
- Cuts and unreliable regions trigger the configured behaviour.
- Editing an anchor invalidates affected results.
- Saved projects reproduce corrections.
- Existing untracked correction behaviour remains available.

**5. Feature: clean-plate assistance**

Allow users to attach an empty-screen reference image or sequence.

Implement an initial practical workflow for:

- Inspecting the reference.
- Declaring or checking alignment.
- Optional alignment estimation with diagnostics.
- Sampling spatially varying screen colour.
- Comparing source and clean plate.
- Generating optional rough background/foreground guidance.
- Previewing and accepting the resulting guidance.

Address exposure differences, misalignment, shadows and foreground colours similar to the screen.

Do not present image subtraction as a guaranteed solution. Keep assistance optional and reversible.

Integrate generated guidance with supported hints or refinement interfaces. Do not claim the existing neural model accepts new clean-plate channels unless that conditioning is implemented and trained.

Acceptance:

- Poorly aligned references produce warnings or rejection.
- Generated guidance can be inspected and adjusted.
- Disabling the feature preserves established behaviour.
- Reference changes invalidate dependent caches.
- Real quality benefits remain unverified until representative footage is evaluated.

**6. Feature: click-to-select subject guidance**

Add interactive subject selection using an optional segmentation provider.

Support:

- Positive and negative clicks.
- Clear selection and iterative refinement.
- Preview of the coarse mask.
- Conversion into a supported hint or correction.
- Saving selection provenance and relevant parameters.
- Optional tracked guidance through the selected range.

Define a provider interface. Evaluate suitable implementations such as SAM 2 or Cutie against available hardware, dependency requirements, licences and inference cost.

Do not bundle multiple large models without evidence of a useful purpose.

Keep segmentation distinct from alpha matting. Coarse masks guide the keyer; they do not automatically replace hair, blur or transparency.

Acceptance:

- Clicks affect the selected subject predictably.
- Guidance integrates with existing jobs and corrections.
- Missing providers produce an actionable unavailable state.
- Provider operations cancel safely.
- Existing workflows work without the optional model.
- Actual model execution is distinguished from provider-interface fixtures.

**7. Feature: synchronized playable composite previews**

Extend existing frame inspection into synchronized playback where not already implemented.

Support:

- Source, alpha, foreground and composite playback.
- Source-aligned A/B playback.
- Looping a selected range.
- Matching zoom, pan and frame navigation.
- Checkerboard and custom backgrounds.
- Playback speed controls.
- Visible frame identity and source timing.
- Dropped-frame and buffering diagnostics.
- Responsive pause, seek and cancellation.

Use recorded source offsets and timestamps. Do not assume matching frame indices imply matching source time.

Bound decode and preview caches. Reuse existing readers where practical.

Acceptance:

- Compared views display the same source time within the defined tolerance.
- Seeking and loop boundaries remain aligned.
- Variable-rate or unsupported timing cases are handled explicitly.
- Preview settings do not modify render outputs.
- Playback does not block the UI or grow memory without bounds.

**8. Feature: progressive preview refinement**

Add an explicitly controlled preview scheduler.

Prioritise:

1. The current playhead or selected frame.
2. A small surrounding range.
3. Upcoming playback frames.
4. Further refinement during idle time.

Show a cheap draft first where configured, then replace it with a higher-quality result.

Clearly identify pending, draft, refined, final and stale results.

Cancel or supersede obsolete requests after settings changes or rapid seeking. Ensure late results cannot replace newer valid results.

Reuse compatible work only when provenance and settings permit it.

Full-quality export must use the requested processing mode. Do not silently export draft previews.

Acceptance:

- Latest requests take priority.
- Stale results are rejected.
- Scheduling respects resource budgets.
- UI interaction remains responsive.
- Progressive preview can be disabled.
- Draft and final outputs remain distinguishable.

**9. Feature: visual problem-region overlays**

Extend diagnostics into spatial and temporal overlays.

Where supported, visualise:

- Motion-propagation disagreement.
- Unreliable flow.
- Occlusions and newly revealed regions.
- Unexpected motion-aligned matte changes.
- Difficult boundary regions.
- Correction anchors, cuts and fallback events.

Let users:

- Toggle overlays.
- Inspect their meaning.
- Navigate to affected frames.
- Select a region or range.
- Add a correction or prepare a separate rerender.

Distinguish measured diagnostics from heuristic suspicion. Do not label heuristic scores as calibrated confidence or proof of a visual defect.

Avoid running expensive diagnostics unnecessarily.

Acceptance:

- Overlays align with the displayed source frame.
- Diagnostics have provenance and documented limitations.
- Existing render pixels remain unchanged.
- Suggested actions use established correction/rerender workflows.
- False positives and missed defects are evaluated on reference footage.

**10. Feature: machine-specific performance calibration**

Add an optional calibration workflow using a short user-selected sample.

Measure configurations under controlled conditions, including relevant combinations of:

- Processing scale.
- Temporal window and stride.
- Stage 2 batching.
- Memory profiles.
- Quick Key settings.
- Cache policies.

Record cold and warm behaviour separately.

Produce recommendations for preview and final processing, showing:

- Measured throughput and latency.
- Resource use.
- Quality/approximation settings.
- Failures and excluded configurations.
- Estimated render duration with assumptions and uncertainty.

Do not extrapolate one sample into a universal hardware guarantee. Do not label a memory-feasible configuration real-time without throughput and latency evidence.

Acceptance:

- Calibration is explicit and cancellable.
- Results are tied to hardware, model, settings and sample identity.
- Recommendations do not silently alter queued jobs.
- Results expire or become suspect when relevant conditions change.
- Missing hardware produces no fabricated measurements.

**11. Feature: safe project portability**

Add a project collection and relinking workflow.

Support:

- Portable project manifests.
- Relative paths where suitable.
- Required and optional dependency inventory.
- Correction and settings preservation.
- Missing-media detection.
- Deliberate relinking.
- Source identity and compatibility checks.
- A preview of what will be copied or referenced.

Offer reference-only portability and optional deliberate collection of selected assets.

Do not automatically copy large footage, model weights or caches.

Handle filename collisions and path traversal safely. Never overwrite unrelated files.

Acceptance:

- Projects reopen from a moved location with deliberate relinking.
- Corrections, timing and settings retain their meaning.
- Missing resources are identified precisely.
- Original files remain unchanged.
- Collected projects contain no unintended credentials or unrelated files.

**12. Feature: Resolve-assisted cache preparation**

Extend the current prepared-cache integration into the best supported host-assisted workflow.

Investigate actual SDK and host capabilities first.

Target a workflow that can:

- Identify the selected clip or supported range.
- Capture required source/range information.
- Submit a preparation job to the shared engine.
- Show progress and cancellation.
- Validate the completed cache.
- Refresh or attach it through a supported interaction.

If OFX alone cannot support the full workflow, assess a companion command or supported host mechanism. Document what is automatic and what still requires user action.

Preserve the distinction between prepared-cache playback and live neural inference inside Resolve.

Account for source offsets, retiming, proxies, colour management, multiple instances, cancellation and project persistence.

Acceptance:

- The implementation uses actual supported host interfaces.
- Prepared frames correspond to the intended source range.
- Original projects and media are protected.
- Failures leave the previous valid cache usable.
- Actual Resolve viewer/export tests remain required.
- SDK mocks or standalone cache tests are not presented as host acceptance.

**13. Implementation order and acceptance tracking**

First map existing capabilities against these requirements.

Use the existing acceptance matrix and issue ledger. Add stable feature/task IDs, dependencies, acceptance scenarios, evidence and external prerequisites.

Prioritise:

1. Readiness and a reproducible short real-render path.
2. Playable synchronized previews.
3. Tracked corrections.
4. Clean-plate and click guidance.
5. Progressive refinement and overlays.
6. Calibration and portability.
7. Resolve-assisted preparation, progressing earlier in parallel where independent SDK investigation is useful.

Adjust ordering using dependencies and actual findings.

Do not require unavailable real inference to block unrelated local implementation. Conversely, do not mark inference-dependent criteria accepted without execution.

**14. Parallel RALPH execution**

Use one main coordinator and up to four native subagents where supported and useful.

Assign independent tasks with clear file ownership, interfaces and acceptance criteria. Suitable parallel work includes UI work, engine services, portability, SDK investigation and independent review.

Subagents must read applicable AGENTS.md instructions.

Only the main agent updates authoritative ledgers and integration records.

Avoid concurrent edits to the same files. Test stable snapshots rather than code changing underneath the test process.

For each work package:

1. Define acceptance conditions.
2. Inspect interfaces.
3. Implement connected behaviour.
4. Run targeted verification.
5. Review against AGENTS.md and the specification.
6. Address findings.
7. Integrate and update records.
8. Checkpoint, commit and push.
9. Continue.

Checkpoint after approximately three to five meaningful cycles or a substantial coherent batch. This is not a stopping limit.

Do not create competing supervisors or recursive continuation loops.

**15. Verification and reviews**

Preserve existing behaviour through proportionate regression checks.

Add meaningful tests for new scheduling, tracking, alignment, persistence, cancellation, invalidation and ownership behaviour.

Use existing test selection. Do not run the entire suite after every small edit.

Run complete applicable CPU/static checks at substantial integration boundaries and final local acceptance. Reuse evidence only when it covers the relevant unchanged state.

Use independent review subagents for substantial changes when available.

Review:

- Correctness and failure paths.
- Compatibility.
- Architecture.
- Concurrency and resource ownership.
- UI responsiveness.
- Source preservation.
- Temporal alignment.
- Cache invalidation.
- Colour and alpha.
- Licensing and privacy.

Resolve blocking findings before accepting the batch. Record reviewed diff identity and evidence.

Never fabricate speed, quality, GPU or host results.

**16. Checkpoint commits and pushes**

Pushing meaningful checkpoint commits to my fork is authorised.

Verify the remote is `MikeHall1/CorridorKey2.0`. Use an appropriate development/review branch.

At each checkpoint:

- Inspect the exact diff.
- Stage relevant files deliberately.
- Exclude footage, model weights, secrets, caches, large generated files and unrelated changes.
- Commit with a descriptive message.
- Push normally.
- Confirm success and record branch and commit ID.

Do not force-push, rewrite published history, merge into the default branch, publish releases or change repository permissions.

If authentication, protection or divergence prevents pushing, document the blocker, preserve a local checkpoint and continue safe independent work.

**17. Environment, privacy and external prerequisites**

Keep footage and generated media local.

Preserve existing user data, attribution, licences and third-party notices.

Do not install/upgrade dependencies, change interpreters, download large assets or alter the user’s environment without authorization. Read-only dependency, repository and SDK investigation is permitted.

For missing hardware, weights, footage, optional models, SDKs or target environments:

- Record the exact prerequisite.
- Identify affected acceptance criteria.
- Prepare reproducible validation commands.
- Continue independent local work.
- Leave dependent criteria explicitly unverified.

Do not repeatedly retry an unchanged unavailable prerequisite.

**18. Context management and stopping conditions**

Maintain concise persistent checkpoints. Compact at safe boundaries using a supported native action when available. If user action is required, save a handoff and request it once.

After compaction, resume from repository records without restarting completed work or rerunning unchanged checks.

Continue until all ten features are implemented, integrated, reviewed and accepted, or all remaining work is genuinely blocked.

After three materially different unsuccessful approaches to one problem, reassess the root cause and seek a focused second opinion. Record a blocker if no justified next experiment exists.

Do not add speculative work merely to keep the loop running.

If local implementation is complete but real GPU, quality, packaging or Resolve validation is unavailable, report “local implementation complete; external acceptance pending”. Do not declare full product completion.

At every checkpoint report:

- Features and acceptance criteria completed.
- User-visible changes.
- Review findings and disposition.
- Verification performed and applicable reused evidence.
- Measured results and limitations.
- External prerequisites.
- Commit and push result.
- Next concrete action.

Begin by auditing the current implementation, creating the feature dependency plan and starting the highest-value locally feasible work.