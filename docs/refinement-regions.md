# Refinement regions — expansion/feather only

This guide covers the existing correction editor’s **Refinement region — expansion/feather only** purpose: an explicit final-alpha treatment region, not a neural crop, quality measurement or performance optimisation. Refinement coverage does not affect guided alpha refinement, foreground reconstruction, despill or ordinary authored corrections. Other supported non-neural purposes have separate guides: [composite regions](composite-regions.md) restrict existing despill, and [diagnostic regions](diagnostic-regions.md) clip captured tracking-overlay display opacity. Neural ROI remains unsupported and rejected; no purpose establishes lower neural computation cost or verified real-footage quality.

## Workflow

Choose the region kind, paint a native binary mask or import native grayscale PNG/EXR coverage, then use the existing Commit, frame/range/strength editing, overlay, delete, undo/redo and Save-and-attach actions. Authored regions retain static `both`-direction ranges contained in a declared shot. Optional motion tracking now reuses the existing coverage tracker; see the explicit inspection/acceptance workflow below. The editor labels them inactive when the captured job's expansion and feather controls are both zero. Saving only changes future captured jobs; existing queue snapshots retain their original correction document.

Strength zero disables a region. An **enabled all-zero mask** deliberately suppresses expansion/feather over its active frame range. With no enabled region on a frame, the established full-frame operation is unchanged. Where regions overlap, their weight is the maximum of `coverage × strength`, independent of entry order. Finishing is evaluated once with full-frame filter context, then blended using that weight: zero copies the original alpha exactly, one copies the finished candidate exactly, and fractional weights blend. There are no crop boundaries or lost filter context; this does **not** reduce neural computation or establish lower resource use.

## Order and temporal safety

1. Existing neural/draft processing and optional guided alpha refinement.
2. Native expansion/contraction, then Gaussian feather, gated by active refinement regions.
3. Existing ordered trimap/matte/keep/remove constraints.
4. Existing foreground/export operations and alpha association.

Regions never become neural hints, carry mattes, Quick Key forced anchors, artist alpha constraints or neural propagation inputs. Optional region tracking uses separate authored coverage anchors for final-finishing weights only. Full/low-memory/spatial/Quick output paths consume local frame indices within the original selected-source mapping. Original signed/above-one foreground RGB is not changed by region processing; the established output pipeline owns premultiplication.

## Optional moving refinement weights

Select one or more committed region entries in the existing coverage tracking
workflow. Anchors must occupy contiguous composition positions and share kind,
strength, purpose `refinement` and target `edge_finishing`. Select Start/End and
**Region track direction** (forward/backward/both), Preview track, inspect a
retained frame and explicitly Accept. Ordinary brush Direction and matte/keep/
remove tracking keep their established behavior. Region authoring remains
`direction: both`; the separate tracking choice does not alter that static rule.

The existing CPU motion estimator warps finite float32 coverage, retaining soft
masks. It does not track a neural quality region or request new model inputs.
Accepted derived coverage gates only final expansion/feather. Within its accepted
range it replaces participating authored anchors exactly once, including at anchor
frames; original strength is applied once after tracking. Overlapping active
regions use maximum `coverage × strength`, never summed/double-applied strength.
Outside that range, original authored static ranges retain their old behavior.
In particular, frames with **no enabled active region** still retain full-frame
finishing; tracking does not silently change that default.

The selected region overlay resolves the accepted derived coverage at the
inspected local/source frame, even where the track extends beyond the anchor's
static range. This overlay displays mask coverage, not a new matte or quality
measurement. Accepted overlay inspection captures the existing bounded sequence index before source decoding, then performs one content-hash verification plus final indexed-stat/full-membership and authored/derived-asset fences. Cancellation remains active. These cooperating-process checks are not an atomic filesystem transaction. Authored and generated assets remain separate. Undo restores the
prior static document, redo restores the accepted immutable track; editing an
anchor conservatively invalidates derivatives. Save/reopen, portable collection,
source/decode binding, cut/reset barriers, reliability/occlusion stops, complete-
range acceptance, hashes and cancellation use the existing tracking contracts.
Partial or stale previews cannot be accepted; approximation remains an explicit
existing reliability policy, not a silent fallback.

Tracking retains its existing smaller limits:300 frames,32 anchors,4Mi native
pixels,64MiB anchor payload,512MiB logical preview mask/encoded charge and512MiB
accepted encoded assets. These do not establish hard RAM limits or native-call
interruptibility. Region imports may exceed tracking limits; that does not imply
they are trackable. Video tracking remains unsupported. No inference-cost,
pretrained GPU quality or real occlusion-handling guarantee is claimed.

## Persistence and compatibility

The first region expands a correction document to schema **5**, adding `kind: region`, `purpose: refinement`, `target: edge_finishing`. Existing schemas1–4 retain their shape and semantics. Tracked regions reuse existing schema5 track fields and complete anchor bindings (including purpose/target); no new wire field or schema version is added. An earlier schema5 reader that supported only static regions rejects tracked-region anchor kinds, rather than misapplying them as matte constraints. Schema5 retains existing tracking/guidance fields; guidance attachment must not downgrade it to schema4. Regions reuse immutable hashed assets, relocation/portable collection, document revisions, output protection and conservative whole-shot invalidation. Any region edit changes the full correction identity; no new recovery-cache exemption is introduced.

Authored edits retain the existing conservative derivative invalidation and immutable undo/history snapshots. Mixed-version operation is fail-closed: an older application rejects schema5 rather than interpreting a region as an alpha-removal operation. To roll back the application, reopen the preserved pre-region document/snapshot; disabling a region leaves a schema5 document and is not a downgrade migration. No old assets or renders are deleted.

## Admission and bounds

New region assets reuse trimap encoded/header admission **without** the three-class restriction: native single-channel8/16-bit PNG or supported zero-origin scanline HALF/FLOAT EXR, finite coverage `[0,1]`, at most64Mi pixels and256MiB encoded bytes. Geometry/header/hash/regular-single-link checks precede decode; no implicit resizing. Existing categorical trimap rules remain strict. Unsupported RGB/RGBA region imports, codec layouts, aliases, changed hashes and resource-limit violations produce visible errors. These are work/admission limits, not a hard process-RAM cap.

## Local verification

```sh
QT_QPA_PLATFORM=offscreen OPENCV_IO_ENABLE_OPENEXR=1 \
  python -Werror -m unittest tests.test_tracked_refinement_regions tests.test_refinement_regions \
  tests.test_desktop_corrections tests.test_trimap_refinement \
  tests.test_edge_finishing tests.test_correction_editor -v
```

Generated CPU fixtures verify exact outside identity, full-candidate equality, fractional/overlap behavior, unchanged hints and foreground bytes across full/low/spatial/Quick-fallback paths, bounded admission, legacy formats and editor history. They are **not** real pretrained GPU, hair/blur/transparency quality, Windows or Resolve acceptance.

## Explicit non-neural purposes

Schema5 now also admits `composite/despill` and
`diagnostic/tracking_overlay`; the existing `refinement/edge_finishing` default
and serialized fields/bytes retain their meaning. The editor derives target from
its purpose choice; existing entries cannot be relabeled as another purpose.
Unsupported pairs (including neural ROI) fail closed. Earlier schema5 readers
reject the new purpose pairs explicitly. Region assets/tracks/strengths are never
interpreted as matte/hint/carried-alpha/forced-anchor inputs.

Each purpose independently combines active static/accepted tracked coverage ×
strength by maximum. No enabled entry means the old whole-frame operation;
strength0 disables an entry, while an active zero mask suppresses only its own
purpose. `CorrectionRasters.composite_region(frame)` and
`diagnostic_region(frame)` return native float32 HW coverage or None.
Composite coverage is intended only for the existing despill in cutout/checker;
other outputs and despill strength0 are inactive. Consumer integration and its
native export tests are required before composite acceptance. Diagnostic coverage
has a display-only consumer described in [diagnostic-regions.md](diagnostic-regions.md).
