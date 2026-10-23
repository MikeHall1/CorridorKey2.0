# Tracked correction previews

The existing correction editor now offers explicit **Preview track** for local
image sequences. Author or import a matte/keep/remove mask, select it and choose
Start/End exclusive and Direction. Additional artist anchors are added with
**Add selected anchor**; they must occupy contiguous composition positions and
have the same kind and strength. Trimap classes and video tracking are unavailable.
The ordinary directional range still applies a stationary raster.

Tracking runs on the editor's background owner. It reads native source frames
through the existing colour/alpha decoder and uses CPU Farneback flow directly
from artist anchors, or two-anchor interpolation between surrounding anchors.
It never chains a retained approximation as a new artist anchor. Backward
tracking uses exact indexed sequence files and explicit source offsets. No model
is loaded; no extra neural conditioning channel is introduced.

Cuts, authored shot boundaries and explicit resets bound the range. Crossing a
boundary is refused in this persisted version: narrow the range and author a
new anchor. Reliability is a flow-consistency/photometric heuristic, not calibrated
confidence or proof of occlusion. `request_anchor` and `stop` retain no mask for
unreliable frames; the interval needs another anchor. Explicit
`retain_approximation` permits a labelled conservative approximation. These
settings describe guidance, never professional-quality alpha acceptance.

The preview changes no document/history. **Inspect tracked frame** displays
retained coverage over the selected source frame in a read-only canvas. At least
one successful inspection and complete coverage of the requested range are
required for **Accept inspected track**. An incomplete preview requires a shorter
range or additional anchors. **Discard preview** changes no authored work.
**Cancel operation** cooperatively cancels the owned operation; native decode or
flow must finish before the owner can observe cancellation. A subsequent request
reuses the same editor owner. Closing retains ownership until terminal cleanup.

Acceptance creates one history revision. Schema3 preserves the authored entries
and bytes, and stores separate native float32 grayscale lossless EXR masks.
Within the accepted span, participating constraints are replaced exactly once at
their first original composition slot. Later operations retain their order.
Outside the span the original ranges still apply. Neural hints and temporal carry
continue to use the original authored semantics; tracking affects final constraints
only. Schema1/2 retain their meaning; older readers refuse schema3 explicitly.

Undo/redo and save/reopen retain accepted assets and content identities. Author,
anchor, ordering or shot edits conservatively drop all derived tracks; undo can
restore the earlier immutable snapshot. Clear accepted tracks creates a reversible
history revision. Source, anchor revisions/hashes, boundaries, decode/colour
interpretation, motion settings and implementation/estimator identity are bound.
Per-frame status, source coordinates, proxy/decoded-mask hashes and heuristic
diagnostics persist in the manifest. Terminal diagnostic records keep bounded
correction summaries and the immutable manifest identity/path; large legacy
entry fields are explicitly omitted with counts, and per-frame tracking payloads
are not duplicated into the4MiB worker envelope. Mismatched live decode/cut settings and newly detected cuts
refuse reuse rather than silently applying stale results. Output/recovery guards,
diagnostic snapshots and benchmark inventories include derived dependencies.

Admission bounds are300derived frames per document,32tracks/anchors,4Mi native
pixels per mask,64MiB artist mask bytes,512MiB logical preview mask plus encoded
bytes and512MiB accepted encoded bytes. The manifest remains bounded to4MiB.
The sequence index admits at most100,000entries/frames and32MiB path bytes. The
kernel streams one candidate; the raster reader retains at most two masks.
These are payload/work bounds, not hard whole-process RAM or native OpenCV
allocation guarantees. No implicit scaling or precision reduction obtains success.

Private cleanup failures retain bounded retry ownership and visible warnings,
including when the primary operation fails;
committed edits and a newly published preview remain valid. Eight pending private
cleanup directories block more preview creation. Accepted assets are never removed
by preview cleanup. Original media and authored mask bytes remain unchanged.

Validation: `PYTHONPATH=tests python -m unittest tests.test_correction_tracking
tests.test_tracked_corrections tests.test_tracking_editor_integration
tests.test_desktop_tracking tests.test_tracking_runtime -v` (one shell command).
Actual CPU flow on a synthetic moving texture, native PNG/EXR publication,
transaction/cancellation faults, Qt ownership and shared toy-model runtime paths
are covered. Real moving/occluded footage quality, long ranges, pretrained CUDA,
Windows/native interaction and video timing acceptance remain NOT VERIFIED.
