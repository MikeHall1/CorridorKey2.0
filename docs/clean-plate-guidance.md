# Clean-plate assistance

The correction editor now has an optional, collapsed **Clean plate** panel. It
does no work until Preview. Select a local still or indexed sequence frame,
declare its sRGB or scene-linear BT.709/BT.2020 interpretation, and supply native
binary background samples. The whole-image background assumption is an explicit
alternative; it can mistake similarly coloured foreground for background.
Source frame coordinates include the job's selected start frame.

Preview uses the existing native decoder and working scene-linear BT.709/D65
RGB. The reference must match source geometry. Alignment is either an explicit
translation or estimated translation validated against the samples. Optional
scalar exposure compensation is explicit. The spatial screen field and relative
RGB/chroma residual produce rough background/unknown/foreground classes, with
an unknown band and erosion. This is heuristic guidance, not a new neural input,
an alpha estimate, calibrated confidence or demonstrated quality improvement.
No source, export, neural precision or render default changes.

Inspect guidance before Accept becomes available. Screen field, residual and
reference inspection do not count as inspecting guidance. Unusable candidates
remain inspectable and cannot be accepted. Accepted classes are an ordinary
native trimap applied by the shared engine, initially over the selected frame.
Existing strength, direction and range controls govern its application. Undo,
remove, save and reopen retain the normal correction behavior. Generated roles
are read-only; load an authored mask to paint. Generation-frame/kind/asset changes
require regeneration or explicit Detach origin. Detaching retains the mask and
deliberately removes its reference dependency.

Schema4 preserves schemas1–3 and adds generation origins and reference inventory.
Bindings include source identity, exact decoded RGB hashes, source/reference
decode declarations, mapping, settings, producer/version, reference hashes and
the accepted raster hash. Paths are excluded from semantic bindings. References
remain local dependencies and changed/missing files refuse reuse until deliberately
detached or regenerated. Final masks do not require an optional model at render
time. Reports carry bounded counts and manifest identity; complete observations
remain in the local manifest. Output protection and recovery identities include
these assets through the shared inventory.

Initial workflow limits: source sequences only, one selected source/reference
pair per Preview, translation rather than perspective/rotation, no implicit
resizing, max4Mi native pixels and a conservative768MiB working-array estimate
(which can refuse smaller images). This estimate is not a process RAM cap.
Encoded source/reference admission is64MiB per image, geometry is checked before
native decode, and only captured immutable bytes enter the decoder. Accepted
formats are static8/16bit grayscale/RGB/RGBA PNG,8bit grayscale/RGB JPEG and
zero-origin single-part scanline HALF/FLOAT grayscale/RGB/RGBA EXR. Palette,
animation, CMYK, tiled/deep/multipart/layered/subsampled EXR and differing EXR
windows refuse. Existing source decoding outside this workflow is unchanged.
Reference inventory is limited to32 origins/300 resources,64MiB each/512MiB total.
Private failed-copy cleanup retains visible ownership, retries and blocks growth
at eight pending paths. Errors never convert a committed edit into a failed edit.

Focused validation (compatible Python/NumPy environment):

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen python -W error -m unittest test_clean_plate test_clean_plate_editor test_desktop_clean_plate test_guidance_provenance test_native_image_admission -v
python -m unittest discover -s tests -v
python -m corridorkey.benchmark
```

Tests use local generated images and toy models. Representative moving plates,
shadows, hair, occlusion, exposure drift and actual pretrained GPU matting quality
remain NOT VERIFIED. No footage, model or dependency was downloaded or installed.

## Multiple source screen-colour samples

The same assistance panel optionally builds an **explicitly synthetic** spatial
screen reference without an empty plate. Enable source screen samples and enter
one native `x,y` coordinate per line (up to32 unique points). Coordinates refer to
the currently selected original source frame, not the thumbnail. Each colour is
the channel-wise median of a clipped native square patch (radius1–31pixels,
default3), after the existing explicit source transfer/primary interpretation.
Signed and above-one working-linear BT.709/D65 values are not clipped. The RGB
reference is inverse-square distance interpolated using bounded128-row tiles,
not a stack proportional to samples × all pixels. This is an **uncalibrated
colour heuristic**, not a measured subject boundary or learned foreground model.

`Seed reach` (1–32768native pixels, default256) limits **foreground/background
seed classes** to pixels near a sample; farther pixels remain unknown. It does
not crop inference or guarantee unchanged output there: accepted trimaps retain
existing full-frame unknown-class refinement. Screen-coloured foreground,
contaminated sample patches, lighting changes and interpolation between different
backgrounds may produce wrong guidance. Inspect rough guidance, spatial field,
residual and the synthetic-reference thumbnail before accepting. Insufficient
sample support or failed sample agreement remains unusable; no fallback plate,
registration or exposure fit is invented. Actual-plate controls cannot be combined
with synthetic mode; switch modes deliberately. Existing empty-plate behavior and
defaults are unchanged.

`Guidance strength` (0–1, default1) is passed to the existing final trimap
correction operator, blending current alpha toward its target once. Zero persists
an explicitly disabled entry, with no alpha effect. Later range/strength edits,
undo/redo and save/reopen use the existing correction history. This creates no
new pretrained input channels and does not automatically update queued jobs.

Acceptance retains the exact generated FLOAT EXR reference as a hashed private
workspace asset before attaching the generated trimap origin. Provenance uses
the existing `clean_plate` producer contract but explicitly records the synthetic
kernel, native coordinates, patch medians, radii, source/decode/frame binding,
output hashes and limitations; it never claims an observed empty plate. Generated
reference files join normal correction asset collection, output protection and
recovery identity. Preview files remain disposable; accepted resources survive
preview discard and undo/redo. Failed copies/checkpoints clean only files created
by that attempt; locked cleanup has bounded ownership and blocks further growth.
Original source images and real plates are never changed.

Focused synthetic-fixture check (no GPU/weights needed):

```sh
python -W error -m unittest tests.test_screen_samples -v
```

Real video timing, plate/background-change detection from absent data, moving
subjects, representative hair/transparency quality and pretrained GPU benefit
remain **NOT VERIFIED**. No speed or memory-saving claim follows from this feature.
