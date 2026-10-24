# Diagnostic regions: captured tracking display only

Region purpose `diagnostic`, target `tracking_overlay`, clips **display opacity**
of the existing explicitly captured reliability/uncertainty maps. Select Region
and its Diagnostic purpose in the correction editor, import/paint native coverage,
then prepare a tracking preview with diagnostic capture enabled. Capturing first
and editing the document afterwards invalidates that preview; prepare it again.
Authored-anchor/uncomputed maps remain unavailable. Saved tracks have no backfilled
maps. This does not run additional measurements or change tracking decisions,
scalar statistics, source RGB, alpha, foreground, saved observations or exports.

Active native coverage × entry strength is combined by maximum within this purpose,
then area-resampled to the actual captured map proxy (maximum side256). Only RGBA
alpha is multiplied; proxy RGB and map values remain unchanged. An enabled all-zero
region hides the map. Strength0 disables an entry; no enabled region retains the
whole-map display. Full coverage preserves the unscoped layer pixels. Fractional
coverage is a display weight, not a native-pixel reliability measurement.

Static authored coverage remains effective outside an accepted track. Accepted
tracking uses existing motion/assets/bindings and replaces anchors only inside its
accepted interval, with identical strength/purpose/target requirements. Cuts,
partial tracks, stale document/source/decode/candidate bindings, corrupt coverage,
cancellation and late asset/source mutation fail closed through existing admission
and publication fences. Undo/redo/save/reopen/portable collection retain purpose.

Maps remain uncalibrated heuristic area-average proxies: not confidence, quality,
verified occlusion, native localization, or evidence that every defect was found.
No neural ROI, extra model channel, changed neural computation or new diagnostic
service/cache/decoder is introduced. Real-footage quality is NOT VERIFIED.

Focused CPU validation:

```sh
QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 python -m unittest \
  tests.test_region_purposes tests.test_diagnostic_regions \
  tests.test_region_purposes_ui tests.test_tracking_overlay_editor -v
```
