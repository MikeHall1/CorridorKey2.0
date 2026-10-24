# Spatial despill regions

An explicit correction **region**, purpose `composite`, target `despill`, limits
only the existing configured RGB despill used by cutout/checker outputs. It does
not crop neural inference or reduce model work. Refinement and diagnostic regions
remain separate purposes; the editor offers these choices explicitly.

Coverage is native-resolution float32 in `[0,1]`. Active overlaps combine by
maximum coverage × authored strength, including tracked masks. No active region
retains whole-frame despill. Strength zero disables that region; an active all-zero
mask instead suppresses despill. Frame/range indices remain local to the selected
clip; tracked provenance retains the absolute source-frame offset.

The consumer blends the ordinary configured despill candidate with the original
foreground in the **same existing working domain**, before checker compositing or
cutout alpha association. Zero/full weights retain the exact respective RGB
endpoints; fractional weights use `original*(1-weight) + candidate*weight`. Existing
screen channel, global strength, edge-only weighting and linear-luma preservation
still apply. EXR stays scene-linear, including signed/HDR values; legacy display
processing and explicitly selected linear preview processing are unchanged. No
alpha division or near-zero foreground reconstruction is added.

Normal, low-memory and explicit spatial-draft neural output, and Quick Key source-
RGB draft output (including neural fallback frames), share these semantics. Alpha,
raw foreground, hints, carry state, matte constraints, model reconstruction, batching
and inference chronology are not changed by this region purpose. Alpha/raw-FG-only
requests do not rasterize composite weights. A wholly inactive batch allocates no
region mask; an active neural batch transfers one native coverage tensor shared by
its cutout/checker consumers. Inactive frames within that batch receive weight one.
Malformed geometry, precision, nonfinite values, out-of-range coverage or a tensor
on the wrong device are rejected rather than resized or silently converted.

## Verification and limits

```sh
QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 \
python -m unittest \
  tests.test_composite_region_rendering -v
```

CPU tests exercise actual decode/shared orchestration/float and half EXR writes,
tracked and static regions, selected-source offsets, both memory paths, spatial
output, Quick Key/neural fallback, colour-domain references, alpha endpoints,
cancellation and bad-mask rejection. Video checks use an in-memory writer, not
FFmpeg. Fixture models establish plumbing, not pretrained model quality. Captured
no-region baseline arrays and scoped static evidence live locally under ignored
`benchmark-results/product-next/composite-regions/`.

Actual CUDA transfers/peaks, pretrained weights, representative footage quality,
FFmpeg encoding, Windows and Resolve remain **NOT VERIFIED**. No throughput, VRAM
cap, selective-neural or quality improvement is claimed. Existing whole-frame
pointwise colour work remains; the legacy neural video cutout still computes its
pre-existing unused linear candidate. Removing that work is a separate measured
change, not part of this compatibility-preserving region consumer.
