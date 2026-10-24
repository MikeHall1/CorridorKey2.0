# Spatial tracking diagnostics

The correction editor's
**Show spatial tracking diagnostics** section is collapsed by default. Its
**Capture diagnostic maps on next Preview track** checkbox defaults off. Explicit
Preview track reuses maps from the existing flow computation; enabling capture
adds no second flow pass or neural inference. Original authored masks and render
pixels remain unchanged.

Inspect **Flow reliability heuristic**, **Uncertain fraction**, or optional
**Warped-anchor coverage difference**. Maps are
area-averaged, quantized8-bit display proxies with maximum side256; they are not
native correction masks. Red opacity means low algorithmic reliability or the
fraction flagged uncertain in a proxy cell. Opacity affects the next explicit
inspection only, and0 shows the unchanged source thumbnail. These flow/photometric
heuristics are not calibrated confidence, proof of occlusion or a detector of
all visual defects. Disagreement is the actual absolute difference between the two warped authored
anchor coverages already evaluated for a bidirectional interior target, not a
map invented from a scalar statistic. It is not confidence, occlusion truth or
final matte error. Equal warped anchors yield a genuinely measured zero map.
Single-anchor/directional targets, authored anchors and uncomputed stopped
intervals have no disagreement map; selecting it reports unavailable. An actually
evaluated first rejected target may retain its measured map.

Artist anchors are declared events with no spatial measurement: their initialized
all-one reliability is not displayed as evidence. A first failed propagation
can have actual maps even though its mask was rejected. Subsequent stopped or
anchor-required intervals with no computed maps explicitly report unavailable.
The bounded frame-event list shows local correction and absolute source frames,
status and map availability. Existing cut/reset/shot barriers still refuse
crossing ranges; capture does not bypass them or invent boundary measurements.

**Go to diagnostic frame** and **Inspect overlay** verify document, source, decode,
frame/proxy and receipt identity before displaying a source copy. They never mark
a tracked mask inspected or authorize its acceptance. **Source / authored view**
removes the diagnostic display through the existing source-view workflow.
**Use captured range** sets the normal correction Start/End controls without
editing history or submitting work. **New correction at shown frame** returns to
the exact displayed local/source frame and reloads the real source before native
painting. Commit remains explicit. Dirty drafts refuse navigation/range/inspection
changes; completed late diagnostic pixels are suppressed after cancellation.

Maps are private to the current tracking preview. Saved/accepted tracks retain
scalar observations and final masks, not these spatial maps. Old projects cannot
reconstruct them without a new deliberate tracking preview. Authored edits discard
preview derivatives; stale source/document/decode/mask/receipt data refuses.
Acceptance still requires ordinary complete tracked-mask inspection. After commit,
only the private diagnostics subdirectory is removed; accepted mask assets remain.
Cleanup failures retain visible pending ownership/warnings and never turn a
committed correction into a reported absent edit.

The existing512MiB logical mask/encoded-byte preview budget remains separate from
an additional64MiB logical map/receipt budget when capture is opted in. Existing
tracking limits remain300frames,4Mi native pixels and32anchors. Two display maps, plus an optional third disagreement map for evaluated
two-anchor targets, fit within256×256 bytes each. Old version1 two-map receipts
remain readable; new three-map receipts retain strict ordered kinds, source/request
binding and per-leaf authentication. Capture is off by default and does not change
tracked masks, reliability, uncertainty, fallbacks or the number of flow passes. Receipt metadata is bounded64KiB,
provenance32KiB, with exact declared dimensions, aspect, hashes and safe exclusive
leaf names. These are retained data/admission bounds, not whole-process RAM or
disk reservation guarantees. Failed publication removes only its newly owned
leaves; blocked cleanup preserves the original exception and owned paths.

Explicit inspection authenticates the full frozen source sequence and selected
native proxy before/after compositing. This can read substantial local media;
checks are cancellable but native I/O/flow can delay cancellation. There is no
automatic expensive diagnostic scan or continuous filesystem watcher. Ordinary
mutation is fenced; hostile concurrent filesystem owners are outside scope.
Display conversion uses the existing source colour/alpha contract, including
linear EXR handling. The display-only red layer never enters corrections or the
shared render pipeline.

Focused reproduction in the existing compatible environment:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen python -W error -m unittest test_tracking_disagreement test_motion test_correction_tracking test_tracking_overlays test_tracking_overlay_editor test_desktop_tracking_overlays test_desktop_tracking -v
```

Representative false-positive/missed-defect rates, real motion/occlusion quality,
pretrained CUDA behaviour, Windows packaging and Resolve viewer/export acceptance
remain NOT VERIFIED. This initial slice does not add saved-track spatial maps,
matte-boundary/motion-aligned-alpha-change diagnostics or a proven occlusion map.
Quick Key spatial disagreement capture remains unsupported; this extension is
limited to the existing authored two-anchor correction-tracking producer.
