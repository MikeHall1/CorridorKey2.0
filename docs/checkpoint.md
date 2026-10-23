# Current checkpoint — photo composite previews, 2026-10-23

Twelfth connected increment locally accepted. All ten workflows/full product
acceptance remain incomplete; [extension brief](product-improvements-brief.md).

Photo… explicitly prepares a bounded local RGB image with assumed sRGB or linear
BT.709/D65 interpretation and exact native geometry. Immutable snapshots feed the
existing single preview loader for still inspection, synchronized A/B and declared
sequence playback. No engine setting, queue or export pixel changes. Preparation
is cancellable/superseded, stale photo pixels hide, native reads retain ownership
through close, and static pins retain captured photo and visible-frame identity.

Independent substantial review and main distinct review accepted339–350: strict
public prepared proof/budgets, file-fence ordering, readonly/strided input support,
metadata terminal acknowledgement/narrowing, physical fixture roots and playback
photo captions. Actual pendingframe3/visibleframe2 reproduction now pinsframe2
pixels withframe2caption andcachedphotoidentity. Core13, pipeline25, newQt7 and
existingpin1 pass; independentfinal7Qt andexact350replay pass. Other20existing
playback/sourceQt cases pass, finalfullsuite coversall. [Review](photo-backgrounds-review.md),
[workflow/limits](photo-backgrounds.md).

Final **1434CPU tests PASS103.846s/no skips**;406Python files/full compilation/import/dependency/static regression/whitespace gate PASS,0new vsbaseline AND actual03577e0. Existing519Ruff/684Pyright errors17warnings
remain; static005/nativeQt126 stay open. Frozen source identity:
`67910e8b41c6ce2f95545e57c98e1f05018b3e8de95b50dd142b76281fed17c1`.
Evidence: `benchmark-results/ext05-photo-connected-final/`. Prior unchanged-driver headless benchmark3x1137CPU reused
for applicable orchestration scope only; new photo source independently full-suite/
static verified. No inference speed or quality claim.

Session-onlyRGB images, native-exact dimensions and explicit assumptions only.
Combined defaultwork limits admit1080p but not native4K. No ICC/OCIO/EXIF automation,
alpha/video backgrounds or resize fallback. One active snapshot plus an outstanding
prior decode may retain bounded bytes; work estimate is not a hard RSS cap. Native
Windows/managed-display quality, pretrained CUDA/VRAM, optional SAM2 and Resolve
SDK/host acceptance remain NOT VERIFIED. No dependency/user-environment, render
precision/colour/default change, download or asset publication.

Prior03577e05a240a1e45d2ab2f5d383980f1b9b94b4 pushed/confirmed to
MikeHall1/CorridorKey2.0 review/corridorkey-audit-20261023. This publication's identity
follows in ignored `benchmark-results/ext05-photo-publication.json`.

Next: Implement bounded reusable foreground/alpha sequence indexes for export playback on the existing loader, preserving source-offset A/B, native geometry/colour and explicit refresh/stale ownership; continue remaining feature contexts and supported Resolve preparation.
Five cycles is an interval; continue without signoff or another supervisor.
