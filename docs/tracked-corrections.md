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


## Refinement-region coverage tracking

The same tracker now accepts schema5 authored `region` coverage, restricted to
`purpose: refinement`, `target: edge_finishing`. This is an extension of existing
kind admission and routing, not a new tracking algorithm or wire format. Complete
anchor fields remain in provenance; all anchors must share purpose/target/strength.
Regions stay both-direction when authored; **Region track direction** separately
selects forward/backward/both without changing ordinary brush Direction.

Inspect/Accept/save/history/resource/failure rules above apply unchanged. Accepted
region masks are read only by final expansion/feather weighting, with participating
anchors consumed once inside the accepted span and static behavior outside. They
never become final artist matte constraints, hints, carried alpha, forced neural
anchors or foreground operations. The editor shows accepted derived region
coverage at the inspected frame. Maximum overlap and strength-once semantics
are described in [refinement regions](refinement-regions.md).

Earlier schemas1–5 keep their existing behavior. Old static-region schema5 readers
fail closed on tracked-region anchors; retain the pre-tracking snapshot for rollback.
Portable CPU tests exercise soft forward/backward/both coverage, publication,
relocation/collection, native output math, failure fences and Qt selection. Their
synthetic motion/structural neural substitutes do not establish pretrained/real-
footage/GPU quality, Windows or Resolve acceptance.

### Region purpose isolation

Tracked regions retain their explicit schema5 pair: refinement/edge_finishing,
composite/despill, or diagnostic/tracking_overlay. Anchor binding includes the
same purpose, target and strength across all selected anchors. Isolation reads
that pair's coverage, never reinterprets regions as matte, and acceptance keeps
original immutable assets, derived masks, source/decode bindings and existing
cuts/resource/cancellation/invalidation/history/portable behavior. Inspection
labels the actual pair. Diagnostic coverage gates captured overlay display only,
not measurements or propagation. Neural ROI remains unsupported. Composite
runtime acceptance additionally requires its despill consumer tests.
