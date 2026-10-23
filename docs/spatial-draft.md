# Reduced-resolution spatial draft

This implements the contract for brief §6 item 19 (CK-P06-19).
M45 is locally verified through the shared engine and desktop. Target pretrained
quality, GPU benefit, Windows packaging and Resolve-host acceptance remain
unverified.

The endpoint is explicitly requested spatial approximation through the shared
engine, with native-sized exports and unchanged source timing. It must work
with video and sequences, normal and low-VRAM execution, full-rate inference
and Quick Key, existing hints/corrections/refinement, and all selected outputs.
Temporary partial integration does not satisfy this endpoint.

## Configuration and coordinates

`model_scale=1.0` remains the default and retains its existing arithmetic and
output behavior. The range is finite numeric `0 < model_scale <= 1`;
booleans, nonfinite values, zero, negatives and enlargement reject before
inference. Selecting a smaller value explicitly requests approximation. No
resource failure automatically selects it.

Keep four distinct geometries: native source, rounded unpadded model image,
padded model tensor, and native export. Existing architectural multiples still
apply to the padded tensor. Record requested scale and actual dimensions; a
smaller requested scale can leave the padded tensor unchanged. Report that
case honestly without promising a memory or throughput improvement.

For the new spatial branch, interpret source colour and alpha at native size,
then resample scene-linear RGB for model encoding. Preserve signed HDR values.
Crop model padding before rendering. Decode reconstructed foreground and
invert model adjustments before resizing linear foreground to native size.
Resize alpha separately as coverage; clamp alpha, not HDR foreground.
Despill, transfer, composition and optional premultiplication operate after
RGB and alpha have compatible native dimensions. Record resampling policy.

Authored raster assets retain native geometry and source-frame binding. Model
conditioning and carried hints use explicit conversions between coverage
grids. Final authored constraints apply exactly once at native output size,
after alpha upsampling, so small native masks do not disappear in a lossy
round trip. Preserve operation order, direction, strength and reset semantics.

Refinement radius retains native-pixel meaning. Native source guidance needs
an owned reader independent of model scheduling. Existing forward-only readers
cannot serve an earlier fallback frame after reading future anchors; close and
recreate an appropriate bounded reader when chronology requires it.

Quick Key continues using native current RGB, native motion coordinates and
native anchor alpha. Neural anchors and additional fallback work use the same
explicit spatial scale, then upsample alpha before propagation. Spatial and
temporal approximation remain separately identified.

## Workflow and provenance

Full-quality actions reset both spatial scale and temporal approximation.
Settings, projects and queued jobs preserve deliberate scale selection.
Output names and contracts visibly identify spatial approximation; scaled
reconstructed foreground must not be described as Quick Key source RGB.
OFX cache preparation must retain the draft classification.

Recovery identity must distinguish scale and implementation/resampling policy.
Writer-prefix replay must not mix native and spatial draft outputs. Existing
map budgets account for actual model dimensions; native alpha, flow, export
and guidance allocations remain native-sized. Retained-map savings do not
establish a total RAM or VRAM reduction.

## Local verification scope

- Use shape-aware CPU model substitutes and actual odd-sized uint16, signed
  HDR and premultiplied sequences, plus an actual supported video fixture.
- Verify native counts, dimensions, frame offsets and FPS across both memory
  paths, hints/corrections, refinement and Quick Key including neural fallback.
- Check every output's alpha/colour/association contract with independent
  expected values, including linear despill and native fine-mask constraints.
- Preserve scale-1 bytes and existing workflows; test explicit admission,
  UI full-quality reset, immutable queue settings and truthful draft labels.
- Verify cross-scale recovery isolation, exact scratch admission, cancellation
  and owned-reader teardown on failure and nonmonotonic fallback.
- Run independent review and applicable integrated CPU/static checks on the
  final stable implementation. Keep actual pretrained quality, GPU benefit,
  Windows packaging and Resolve-host acceptance explicitly unverified until
  supported by target evidence.

Read-only architecture audit: `cache_maintenance`; independent design review:
`cache_review`. Implementation ownership and current evidence are recorded in
the checkpoint. All verification claims remain limited to their recorded scope.

## Current evidence and limits

The final affected batch passed 61 checks; the final test-only scalar typing
change also passed its affected HDR case. All cases are included in the final
916-test complete suite (65.759 seconds, no skips). Actual Qt/shared-worker processing
of three uint16 129×197 frames used padded model grids 192×256 at scale 1 and
64×128 at scale 0.5; both produced native 129×197 EXRs. A constant-alpha CPU
fixture matched the independent 0.4 output oracle within 1e-6. Full quality reset
scale 1 without submission, queued snapshots stayed immutable, actual A/B
source alignment worked, sources were preserved and all owned work terminated.
Evidence: `benchmark-results/m45-spatial-draft-ui-generated-summary.json`.
The fixture uses explicit CPU model substitutes and establishes mechanics only.
The first demo attempt exposed a fixture foundation grid fixed at 1×1; the
corrected ignored fixture derives its grid from the actual input.

Independent review corrected inconsistent raw-alpha clamp ordering and added
a finite fence before proxy generation. The latter was verified by injecting
nonfinite resampler output; a separate small native extreme-HDR OpenCV probe
did not overflow on this build. No observed real-footage overflow is claimed.
Review evidence: `benchmark-results/m45-review-summary.json`.

The first complete 913-test gate failed one obsolete UI-label assertion and
reported four new Pyright diagnostics against M44. These original reports are
retained in `benchmark-results/m45-final/`; they are not passing acceptance.
The corrected final integration passed in `benchmark-results/m45-final-fixed/`:
314 Python files, compilation/import/dependency/whitespace checks, and no new
static findings against the historical baseline or actual M44. Existing
519 Ruff findings and 686 Pyright errors/17 warnings remain; this is not a
static-clean result. Final identities and evidence scope are in
`m45-verification-summary.json` and `m45-final-source-verification.json`.
The final runtime/video type-assertion changes were independently reviewed as
numerically unchanged from the source-pinned actual Qt proof; that proof is
reused with the explicit identity addendum, rather than silently relabelled.

Additional cases verify first/full hints at selected source start 1/count 3,
premultiplied HDR RGBA and native cutout association, and exact scratch
admission at 75870 bytes with rejection at 75869 before map/writer creation.
The HDR oracle checks the linear resampling boundary separately from BF16
encoding; its round-trip interval derives from encoded half-ULP spacing and
float32 arithmetic rather than an arbitrary observed linear error.
Video verification includes an actual AVI input and encoder-array rendering;
it does not establish target hardware encoding or packaged FFmpeg acceptance.
