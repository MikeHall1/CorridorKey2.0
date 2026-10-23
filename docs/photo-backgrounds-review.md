# Photo preview integration review

This initial EXT-05 photo workflow is locally accepted; full product and external
acceptance remain incomplete. Scope: an
explicit bounded immutable background recipe/snapshot and linear compositing;
one existing preview loader; still/comparison/playback propagation; desktop
Prepare/Cancel/Use checkerboard controls and captured display provenance.

Core13PNG/JPEG/uint16PNG/floatEXR/analytic cases PASS0.531s/-Werror, including
exact geometry, native header/storage admission, malformed/nonfinite/alias/budget
refusal, no source/array mutation, readonly/strided inputs, public dataclass proof
and last callback/file fence. Pipeline6new+19existing cases PASS0.257s/-Werror.
Main7actualQt cases PASS2.216s/-Werror; existing pin workflow1PASS0.783s. Earlier
26Qt broad run25PASS/1known349caption failure preserves20existing playback/source
cases; subsequent349 and350 regressions pass. Focused Ruff/Pyright0. Failed and
superseded evidence remains in `benchmark-results/ext05-photo-focused/`.

Findings and dispositions:

- 339/340/344/345: narrow new fixture/import/type fixes. 340unusedimport was
  author-fixed before coordinator ledger transcription; no checker weakening.
- 341: final cancellation callback precedes final file-pin validation, preventing
  callback mutation after the last identity fence.
- 342: public prepared instances validate exact geometry/profile/storage/work
  metadata, immutable float32 payload hash and finite values before use.
- 343: every loader terminal acknowledges prior playback ownership before routing
  photo metadata, including newer preparation that supersedes a paused decode.
- 346: nonlinear foreground transfer uses an owned contiguous copy, preserving
  readonly/negative-stride caller buffers under warnings-as-errors.
- 347: explicit photo-result narrowing prevents metadata reaching pixel access.
- 348: the new pipeline fixture uses its physical macOS temp root and registers
  cleanup immediately; production alias refusal is preserved.
- 349: playback captions include cached photo identity before the final path
  component, preserving it through caption shortening and static pinning. The
  first fix put it in the discarded component; actual failed retests are retained.
- 350: actual visible frame2/pending frame3 reproduction confirmed pinned frame2
  pixels were labelled frame3. A separate displayed description now commits with
  accepted pixels; pin uses that caption/tooltip, retaining exact frame/photo identity.

Main distinct self-review against AGENTS.md and both briefs: optional controls
perform explicit real work, no implicit background resize/colour fallback,
precision/export/job/source preservation, one shared bounded loader, latest
serial fencing and close ownership. Dataclass instance/class, heterogeneous-set,
canonical JSON and caller-forged metadata contracts were inspected explicitly.
The main thread accesses cached identity/geometry/profile fields only; it never
serializes/hashes image payloads. A prior in-flight request may retain its old
immutable snapshot until terminal acknowledgement; the active bound engine field
is cleared on invalidation, preventing unbounded retained generations.

Default combined work limits admit1080p but not native4K. Background state is
session-only; no ICC/OCIO/EXIF automation, alpha backgrounds, video background,
resize policy or display-calibration claim. Existing sequence playback remains
CFR-declared; video/VFR and persistent reader work remain unfinished. Actual
pretrained CUDA/quality/Windows/Resolve acceptance is NOT VERIFIED.

Final independent substantial review accepted the frozen11-file mapping
`749aab4d982bf8e9314968a293ed17f90b379f6a8371425b67a59184162e07eb`.
All7finalQt cases independently PASS1.690s/-Werror; the exact350 reproduction now
pins visibleframe2pixels withframe2caption andphotoidentity whileframe3pending.
No unresolved local blocker. Final frozen **1434CPU tests PASS103.846s/no skips**;406Python files/full compilation/import/dependency/static regression/whitespace gate PASS,0new vsbaseline AND actual03577e0. Repository source identity `67910e8b41c6ce2f95545e57c98e1f05018b3e8de95b50dd142b76281fed17c1` remains unchanged through all checks. Prior unchanged-driver3x1137CPU headless benchmark reused for applicable orchestration scope only, no inference speed/quality claim. No dependency/user-environment, engine model,
render-default/precision/colour changes, download or asset publication.
