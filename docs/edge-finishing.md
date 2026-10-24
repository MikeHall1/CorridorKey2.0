# Opt-in native alpha edge finishing

`alpha_expand_radius` / `--alpha_expand_radius` is an exact integer in
`[-32,32]` native output pixels: positive values take the square-neighbourhood
maximum (expand coverage), negative values take the minimum (contract).
`alpha_feather_radius` / `--alpha_feather_radius` is an exact integer in
`[0,32]`: Gaussian support radius, kernel width `2*radius+1`, sigma
`max(0.5, radius/2)`. Borders replicate edge coverage. Both default to zero;
disabled jobs do not call the new finishing operation. These are artistic
coverage changes, not a neural reconstruction or quality guarantee.

The common desktop settings form exposes both controls. Apply changes future
immutable jobs only. Shared job/worker/project JSON uses the same validated
fields; historical settings without either field load zero defaults. Diagnostics
record requested/effective settings and `alpha_refinement.edge_finishing`
operation order, evaluated/changed final frames and CPU seconds.

## Ordering and colour

1. Native output alpha resizing and existing optional source-guided refinement.
2. Expansion/contraction, then feather.
3. Existing ordered trimap refinement and authored constraints (artist wins).
4. Existing rendering, configured despill, compositing and alpha association.

Finishing occurs exactly once per final emitted frame, including full-rate,
low-VRAM, spatial draft, Quick Key offline/hold/causal and neural-fallback
outputs. It never modifies Stage 1 hints, carried hints, propagation anchors,
flow inputs or cached raw Stage 2 predictions. Quick Key remains its existing
source-RGB approximation, not reconstructed foreground.

The operation accepts finite native coverage in `[0,1]` without transfer
encoding. It touches no RGB: signed/HDR straight foreground is unchanged.
Existing premultiplied EXR multiplies foreground by the final coverage; no
division by small or zero alpha is added. Expansion does not reconstruct
missing foreground colour. Existing reconstruction/despill controls remain
responsible for RGB; this increment adds no replacement reconstruction.

Filtering uses 256×256 tiles with a combined expansion+feather halo (at most
64 pixels). Extra filter buffers and validation/statistic comparisons are tiled;
one full float32 result and the existing native output alpha are retained.
Cancellation is checked between tiles and before returning. Individual native
OpenCV calls are not interruptible. This is not an aggregate RAM/VRAM cap.

## Existing despill controls in the common form

The common desktop settings form now places **Despill domain**, **Preserve linear
luma (linear domain only)** and **Weight despill toward partial-alpha edges** next
to the existing strength and green/blue screen controls. These expose the existing
validated `despill_space`, `despill_preserve_luma` and `despill_edges_only` settings;
there is no new renderer, setting schema or changed default. Defaults remain
strength0.5, green screen, `legacy` domain, luma preservation off and edge weighting
off. Advanced/unexposed settings survive form round trips.

- `legacy` retains historical output-dependent processing. EXR cutout/checker
  paths already despill linear RGB; display-referred video cutout/checker paths
  retain their existing legacy handling. Explicit `linear` selects linear-domain
  processing for display paths too; it does not change source colour declarations
  or introduce automatic colour management.
- Luma preservation requires explicitly selecting `linear`; an incompatible
  legacy+luma selection is rejected on Apply rather than silently changing domain
  or clearing the checkbox. The existing operation preserves weighted working
  linear Y (BT.709 weights) **before export clipping**, not guaranteed displayed
  brightness or a managed-display appearance.
- Edge weighting multiplies configured strength by `4*alpha*(1-alpha)` using final
  coverage: zero at alpha0/1, maximum at alpha0.5. This is a partial-alpha boundary
  emphasis, not a spatial brush/region selector, blur, neural foreground
  reconstruction or reduction of inference work. Fully opaque spill is not
  suppressed by this mode. Existing raw foreground and alpha outputs are not
  newly despilled by exposing these controls.

Load, validation, serialization and Apply use the existing common form and shared
config contract. Apply emits a validated **future-job** snapshot with its original
settings token; existing queued jobs stay immutable. No engine or colour/alpha
operation order is changed here. Standalone offscreen Qt form tests verify defaults,
linear/blue/luma/edge roundtrip, invalid combinations, unexposed-setting retention
and future-job signal semantics. Real GPU/footage quality and managed-display,
Windows or Resolve acceptance are **NOT VERIFIED** by these form tests.

## Local verification and limits

`QT_QPA_PLATFORM=offscreen OPENCV_IO_ENABLE_OPENEXR=1 python -m unittest
tests.test_edge_finishing tests.test_refinement tests.test_spatial_draft
tests.test_trimap_refinement -v`

Tests use actual local EXR source/export I/O and CPU neural substitutes, with
independent morphology/whole-image Gaussian oracles, native-size export pixel
comparison, all memory/draft propagation paths, forced neural fallback,
unchanged signed/HDR foreground, premultiplied coverage math, ordered artist
constraints, cancellation and legacy/CLI/desktop settings roundtrips.
Real pretrained/GPU quality, target Windows/Resolve and magnified dark/light/
custom preview acceptance remain **NOT VERIFIED** by this increment.
