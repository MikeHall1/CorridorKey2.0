# Source-bound shot profiles (opt-in first increment)

**Shot profiles…** analyses the selected source In/Out on a CPU I/O thread; it does not load a neural model or replace the production CUDA backend. Analyze, inspect/edit the JSON rows, optionally Save as a **new** local JSON, then explicitly **Apply to future job**. Render normally when ready. Existing queue snapshots, source media and earlier renders are not modified. Open a saved document with its matching base recipe to inspect/edit/apply it again. **Disable future profiles** clears the optional payload; saved documents remain intact.

## Supported controls and coordinates

Each enabled row has an absolute source-frame half-open `range: [In,Out)`, a `stage2_batch` cap, `alpha_expand_radius` and `alpha_feather_radius`. Caps must be positive integers no larger than the explicitly selected global Stage2 batch. Expansion is an integer in [-32,32]; feather is an integer in [0,32]. Rows must be sorted and nonoverlapping; gaps and disabled rows use global settings. The document's `enabled: false` disables all effects. Runtime uses maximal valid contiguous prediction groups, not a job/model reload at every detected shot. Equal adjacent caps do not fragment batches just because row metadata differs.

This remains **one original complete timeline job**. Source mapping, initial/full hints, correction assets, Quick cuts/resets, recurrent/Hann chronology, model ownership and writer settings are preserved. Detected cuts only propose row boundaries: they do **not** become neural cut/reset markers. Final expansion/feather retain the existing order before authored constraints and compositing. Authored refinement regions continue to gate these effects. Finishing never enters propagated anchors or carried hints.

Per-shot spatial scale, Hann windows, low-VRAM mode, precision, Quick policy and ordinary alpha refinement are **unsupported**, not silently approximated. In particular, ordinary alpha refinement can affect Quick anchors; it is not a final-only control. Stage2 batching is frame-independent in the implementation, but actual GPU batch-size numerical agreement still requires target validation. No speedup is promised.

## What recommendations mean

Analysis reuses the streaming histogram/adjacent-RGB cut heuristic. It records mean adjacent display-RGB change (not optical flow) and mean adjacent-pixel RGB difference (image-edge/detail proxy, not subject-boundary complexity). Cut transitions are excluded from the new shot's motion proxy. Flashes/exposure changes can appear as cuts; hair and translucency can be missed at the reduced proxy scale.

Version1 suggests feather radius1 only for smooth/stable source shots (`mean_edge_detail < .02`, `mean_rgb_change < .01`) when global feather is0; other finishing settings and the global batch cap are retained. These thresholds are **uncalibrated heuristics**, not quality improvements. Feather can harm fine detail: inspect/edit/disable it before accepting. Resource fit is explicitly **unavailable**; source statistics do not establish observed VRAM, memory caps or optimal throughput. Use the whole-plan paired comparison below with actual target footage; calibration sample derivation still rejects enabled profiles.

## Measured complete-plan candidate (explicit inspection)

**Inspect measured candidate…** reuses the existing paired benchmark, numerical
regression gate and separate CUDA allocator-evidence phase. It does not run a
second calibration engine or infer memory from source heuristics. Initially this
workflow accepts **one image-sequence job per side**, covering its entire selected
source timeline. Existing numerical evidence bounds still limit that selection
to120 frames. A/B must share every base setting, model, writer, conditioning path,
row boundary, enabled flag, signal and edge control; **only row Stage2 caps may
differ**, and at least one cap must differ. Global caps remain explicit ceilings.
No rows are merged, no source subrange is derived and no temporal context restarts
at row boundaries.

1. Save two complete bound plans and embed each in otherwise identical existing
   batch manifests. Declare `changed_settings: ["shot_profiles"]` in an existing
   `inference_pair` manifest. Run the ordinary paired producer, with profiling off:
   `python -m corridorkey.benchmark_pair --manifest pair.json --output new-pair.json`.
   Output restaging alone does not change profile recipe identity.
2. Use an explicit version3 (or version4) [regression policy](regression-gates.md)
   requiring **both** numerical parity and observed memory, with numerical
   tolerances, repeated timing/noise limits, observed job0 allocator constraints,
   attempt budgets and accepted tradeoffs. Capture numerical evidence and run
   `python -m corridorkey.regression_resources --report new-pair.json --policy policy.json
   --policy-sha256 EXPECTED_SHA --output-parent NEW_ATTEMPT_PARENT`. This is a
   separate profiled phase; its timings are not throughput measurements.
3. In **Inspect measured candidate…**, provide the explicit local paths and
   expected lowercase SHA256 values in the shown JSON request:
   `report_path`, `policy_path`, `policy_sha256`, `numerical_evidence_path`,
   `numerical_evidence_sha256`, `resource_evidence_path`,
   `resource_evidence_sha256`. Optional `visual_evidence_path` and
   `visual_evidence_sha256` retain version4 visual requirements. Missing or
   unverified human review is never bypassed. The request editor is bounded
   to64KiB, rejects duplicate/nonfinite JSON and does not save a new metadata format.
4. Off-thread inspection calls the actual existing gate and rechecks current
   source/recipe and receipt fences. Failed, noisy/inconclusive, stale, CPU/unknown
   execution and explicitly fixture-produced resource receipts cannot qualify.
   A passing candidate is put in the ordinary editor with receipt identities,
   actual tested candidate/baseline ratios, allocator observations and tradeoffs.
   **Nothing is applied automatically.** Inspect/edit, optionally Save to a new
   file, then deliberately Apply to future settings. Editing a measured plan
   invalidates any inference that its changed combination was measured.

This recommends only a **gate-eligible complete candidate versus its one tested
baseline**, not the fastest/best configuration or independently optimized rows.
A policy may deliberately tolerate a slowdown; inspect actual ratios and tradeoffs.
Allocator observations concern the **whole selected timeline**, not individual
shot peaks, predictions, RSS, device-wide usage or hard resource caps.
`analysis.resource_fit` remains `unavailable` and `calibrated` remains false.
These local receipts are not authenticated hardware/human-inspection attestations.
The reported models/environment remain the applicability scope, not a promise
about later hardware, assets or longer footage. Saving/reopening an ordinary plan
does not persist a qualification claim: inspect the original receipts again.

Resource attempts alone may explicitly clone the exact immutable job into owned
scratch/output locations and switch profiling on. The shared helper validates the
original recipe and rebinds **only** its recipe digest to those execution fields;
source/range/hints/corrections/resets/rows/controls stay unchanged. Runtime still
validates source content; resource admission retains original full pair context
and checks the exact rebound request. Arbitrary profile projection or stale base
settings remain rejected. Existing no-profile resource requests and persisted
formats are unchanged. Disabling profiles remains the rollback.

Portable tests use actual EXR bytes and real gate/receipt code with explicitly
labelled structural worker fixtures; they prove admission/plumbing, **not actual
CUDA measurements or representative-footage quality**. Real CUDA whole-plan
recommendation acceptance remains NOT VERIFIED locally.

## Binding, persistence and limitations

The optional embedded `shot_profiles` configuration is version1 and canonical, with a complete base-recipe SHA256 excluding only its own payload and the output destination. Temporary location, source range, colour/alpha interpretation, hints/corrections paths, temporal controls, writer settings and stage-profiling choice remain bound. Full sequence membership metadata plus selected encoded bytes (or complete video-file bytes) bind source identity. Changing the base recipe or source requires explicit disable/reanalysis; the system does not silently project settings onto foreign frames. Output destinations may change because ordinary desktop queue submission creates a fresh destination.

Shared configuration/project/batch/worker/queue codecs preserve the payload. Missing legacy fields mean `None`/off. Rollback is clearing/disabling profiles without deleting media, cache or history. Recovery identities remain conservative: profile edits invalidate full and intermediate identities; this does **not** expand finishing-only Stage2 reuse exclusions.

Derived single-frame progressive previews, guided short renders, calibration samples, standalone diagnostic ranges, OFX capture rebinding and portable relocation currently **reject enabled profiles explicitly**. Disable first; no caller silently strips final effects or claims full temporal context. Source-only editor inspection remains available. Saved ordinary desktop snapshots at the same source/base locations remain supported.

Bounds: at most100000 selected frames,256 rows and1MiB profile/document metadata. Source capture uses a conservative encoded-read budget of2GiB (four times admitted encoded sizes plus two first-frame sizes); larger sources are explicitly refused. Sequence analysis explicitly admits up to256MiB encoded bytes and16Mi-pixels per image, covering3840×2160 UHD and4096×2160 DCI4K RGB/RGBA and supported HALF/FLOAT EXR. Existing native-admission workflow defaults remain64MiB/4Mi-pixels; only shot analysis opts into the larger hard ceilings. Each analysis frame is decoded from the exact immutable bytes whose header was admitted, using the existing shared source-colour/alpha reader; actual native RGB/proxy geometry must match. Each admitted byte SHA must match the selected-file content receipt in the captured source binding; later source drift refuses a completed plan. This is cooperating-process validation, not an atomic filesystem transaction. Only the documented analysis proxy is reduced; original source/neural settings, colour and precision are not resized or changed. The native codec and working RGB/display arrays still allocate at native resolution, may have multiple simultaneous copies and cannot be interrupted inside a decode call. These limits are not a hard process-memory bound. Video uses the existing OpenCV probe/reader; first-frame probing still allocates a native decoded frame. These are admission limits, **not** RAM/VRAM or wall-time guarantees. Reads are cancellable; source/settings/edit revisions fence stale completions. Saves reuse the existing exclusive, fsynced diagnostic export helper: existing files/aliases and source/output/correction paths are refused. A failed/crashed write may leave an incomplete new document, which bounded parsing refuses; nothing is overwritten.

CLI: add `--shot_profiles /absolute/path/profiles.json` to `infer.py` with the matching captured source/range/base settings. This is explicit admission, not automatic recommendation acceptance or queue mutation. Without the flag/embedded payload, behavior is unchanged.

## Verification boundary

Portable CPU fixtures exercise structural/temporal contracts, native EXR comparisons and UI ownership. They do not establish pretrained quality, GPU batch parity/performance, Windows packaging, variable-rate timestamp equivalence or Resolve host acceptance. Those remain NOT VERIFIED until actual target runs.
