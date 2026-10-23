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
