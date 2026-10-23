# Photo composite previews

Photo… opens session-only background controls without image I/O. Select a
completed foreground/alpha render, choose a local RGB image and explicitly
Prepare selected photo. Declare either assumed sRGB PNG/JPEG or scene-linear
BT.709/D65 EXR. The preview worker authenticates bounded encoded bytes and native
header/decode shape before publishing an immutable background snapshot. Photo
composite then uses that snapshot in still, synchronized A/B and sequence playback.
No render job, saved engine option, queued job or export file changes.

The initial geometry policy is native exact: the photo must match the exports'
native dimensions. It is composited before the viewer thumbnail resize. There
is no automatic crop, stretch, resolution downgrade or profile fallback. RGB only;
alpha/grayscale backgrounds refuse. PNG 8/16-bit, JPEG 8-bit and EXR half/float
storage decode into float32 working RGB. Assumed sRGB ignores embedded ICC/colour
metadata and EXIF orientation; stored pixel order is used. Linear EXR must already
have the explicitly declared working interpretation. ICC/OCIO conversion, automatic
orientation, colour-managed display calibration, background movies/sequences,
resize policies and project-persisted photo selection are not implemented.

Compositing uses straight foreground and coverage in linear light, then applies
the existing sRGB display conversion once. HDR/negative values remain in linear
composite math until viewer clipping. Existing checkerboard, solid backgrounds,
source and alpha viewing keep their behavior. This does not establish a model's
colour primaries or managed monitor accuracy. The core returns fresh pixels and
preserves caller foreground, coverage and source files.

Initial admission bounds are 32 MiB encoded, 4,194,304 pixels, extent8192 and a
512 MiB conservative work estimate. They apply together: the default estimate
`192 × native pixels + 2 × encoded byte limit + 2 MiB` admits1080p but permits at
most2,435,754 pixels at the default encoded limit, so native4K is unsupported.
This estimate is not a hard process RSS or codec allocation cap. One active
background snapshot is retained by the workspace. An outstanding decode can
retain the prior snapshot until it retires; no sequence background cache is added.
Static pinned A retains its displayed thumbnail and captured identity.

The snapshot records encoded and linear-pixel SHA256, declared profile, geometry,
working-basis/ICC/orientation policy and work limits. After capture, external file
changes do not alter it. Re-prepare explicitly to read changes. Selecting another
path/profile clears the active snapshot and hides stale photo pixels. Cancel and
newer requests fence late results. Pending preparation overwritten in the bounded
loader queue cannot leave the controls stuck waiting. Closing waits for the existing
preview worker to retire native reads; no additional decoder owner is created.

Live A/B photo views share one prepared object and retain existing source-frame
mapping, offsets, zoom/pan and playback timing. Their visible captions carry the
captured photo identity. A fixed pinned A remains its captured pixels/identity
when the live photo changes. Use checkerboard explicitly clears photo selection
and changes live photo composite modes to checkerboard. Pinning while another
frame is pending uses the caption committed with the visible pixels, rather than
the newest submitted frame's caption.

Focused verification in the existing compatible CPU environment:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen /private/tmp/corridorkey-compatible-venv/bin/python -W error -m unittest test_preview_background test_photo_preview_pipeline test_desktop_photo_background -v
```

Core13actual-image/analytic cases,25new/existing pipeline cases and7actualQt cases
pass. They prove bounded local mechanics and small-fixture colour arithmetic,
not display calibration, representative quality, GPU speed or native Windows.
Final frozen **1434CPU tests PASS103.846s/no skips**;406Python files/full compilation/import/dependency/static regression/whitespace gate PASS,0new vsbaseline AND actual03577e0. Independent review passes; see
[review](photo-backgrounds-review.md). All ten feature workflows remain incomplete.
