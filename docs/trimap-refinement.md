# Trimap-assisted native refinement

The trimap operation has portable generated-fixture and Qt/shared-worker regression
coverage. This does not establish
pretrained quality improvement or GPU/Windows/Resolve acceptance.

An explicitly imported trimap adds an ordered native correction operation.
Background, unknown and foreground are distinct classes. They are not soft
coverage values. No trimap is inferred from an existing matte and no new model
channel is introduced.

## Operation

Each active trimap entry uses the current alpha produced by preceding entries.
Known background and foreground seed zero and one for the local neighbour
context. A fixed 3×3 RGB/spatial weighted filter adjusts unknown centres only,
including centres whose predicted alpha was exactly zero or one. Its native
radius is one pixel and its display-proxy RGB sigma is 0.1. Known target pixels
remain zero or one. The entry's existing strength blends the original alpha
toward that target once; strength zero is a no-op and strength one gives exact
known constraints. Later authored entries can override earlier ones.

The operation runs once per active entry on final native alpha. Existing global
alpha refinement is a separate, unchanged earlier operation. Trimaps do not
enter neural hints, carried hints, or flow-anchor conditioning, including their
known regions. This is refinement, not learned conditioning or reverse neural
processing. Existing range/direction rules continue to select final output
frames; static directional masks do not become motion-tracked masks.

Normal and low-VRAM execution use the shared engine. Spatial draft first
upsamples prediction alpha to native dimensions. Trimap classes are never
resampled for processing. Full-rate refinement uses an independently owned
native source-guidance reader; Quick Key final output uses its current source
RGB. Raw anchor/fallback alpha remains separate from final authored operations.
Foreground reconstruction, transfer, despill and alpha association retain their
existing behavior and operate with the resulting native alpha.

## Assets and saved states

Trimap entries require correction document schema version 2. Version 1 retains
its existing kinds and rejects trimaps. Version 2 permits mixed existing kinds
and trimaps. Importing a trimap upgrades the edited snapshot; undo can return
to the prior version-1 snapshot. Identity, relocation, history and conservative
shot invalidation continue through the existing correction document mechanism.

Import accepts a single-channel PNG or supported single-part scanline EXR with
exact classes:

| Decoded precision | Background | Unknown | Foreground |
|---|---|---|---|
| uint8 PNG | 0 | 128 | 255 |
| uint16 PNG | 0 | 32768 | 65535 |
| Floating EXR | 0 | 0.5 | 1 |

Validation precedes class mapping. No thresholds, RGB luminance inference,
RGBA-alpha extraction, geometry resizing or colour transfer apply. The internal
read-only labels use canonical uint8 class codes 0/128/255. This class mapping
does not normalize the asset as coverage. Cache keys distinguish trimap class
decoding from ordinary mask decoding even when path and content are identical.

Admission checks an owned regular, single-link file with a bounded read and
unchanged identity. Admission caps are 256 MiB of encoded bytes, a 1 MiB EXR
header, and 64 Mi native pixels. Header geometry/channel checks precede native
decode. Deep, tiled, multipart and unsupported EXR layouts reject explicitly.
The narrow EXR admission follows the official
[OpenEXR file layout](https://openexr.com/en/latest/OpenEXRFileLayout.html);
the codec remains responsible for validating and decoding actual pixel data.

The editor preserves admitted encoded bytes in a new immutable asset. A
separate import action and class legend identify trimaps. Class preview uses
nearest-neighbour sampling, distinct foreground/background colours and
transparent unknown regions. Binary coverage painting and kind conversion do
not silently overwrite trimap semantics. Existing save/reopen/attach steps
remain explicit and do not enqueue work automatically.

## Verification scope

- Verify schema admission, mixed documents, undo/redo, relocation, asset/source
  changes, immutable identity and whole-shot invalidation.
- Exercise actual uint8/uint16 PNG and floating EXR, malformed/near-class values,
  RGB/RGBA, geometry mismatch, bounded header/file admission and cache isolation
  when one file is used as both matte and trimap.
- Use independent expected values for known constraints, unknown endpoint
  movement, strength zero/one/fractional, all-known/all-unknown and ordered
  overlaps. Jobs without trimaps must retain existing outputs.
- Verify final-native application in both memory paths, odd-sized spatial
  draft, Quick Key anchors and additional neural fallback, source offsets and
  directional ranges. Trimaps must remain absent from hint/carry/flow state.
- Verify bounded filter scratch, cancellation, guidance-reader ownership,
  source/output preservation and cache invalidation.
- Exercise actual Qt import → preview → save → reopen → attach → explicit
  shared-worker render and inspect exported pixels and owner shutdown.
- Complete independent review and final applicable CPU/static integration.
  Keep pretrained quality, GPU cost, Windows and Resolve-host acceptance
  explicitly unverified without their actual target evidence.


M46 evidence: corrected focused54tests passed2.477s. Independent tiled-vs-
monolithic float64 oracle max errors1.07e-7/2.09e-7; strict admission probes pass.
The actual Qt import/save/reopen/attach/explicit-worker render preserved sources,
assets, queued settings and old outputs, with native65×97 alpha error at most
5.960464477539063e-08 against a predeclared3e-6 oracle tolerance; all owners ended.
Final integration evidence is composite:965 cases passed in a966-case full run,
then the sole outdated close-test assertion was corrected and passed separately.
All production sources are unchanged between those runs. Full325file static
checks add no findings vs M45; retained519Ruff/684Pyright errors17warnings remain.
The full invocation's failure is retained, not labelled a passing full gate.
See [verification policy](code-health.md) and local
`benchmark-results/fresh-review-checkpoint-verification.json`.
