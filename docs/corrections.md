# Authored corrections

The shared CLI/session/batch engine accepts a local schema-1 correction document through `--corrections PATH` (API/batch: `settings.corrections_path`). This is optional; existing hint/output defaults remain unchanged. Corrections edit authored coverage or conditioning, not model weights. The optional [desktop editor](correction-editor.md) now authors binary native brush masks, imports soft masks and edits/saves existing core correction states. Trimap conditioning, tracked brushes and reverse neural inference remain unimplemented.

```sh
python infer.py --input /local/source --output_dir /local/corrected \
  --output_format exr --corrections /local/project/corrections.json \
  --diagnostics /local/corrected/job.json
```

## Document and binding

The envelope has exactly `schema_version: 1`, nonnegative `revision`, `source`, `shots` and ordered `entries`. `source` contains `sha256`, selected `start_frame`/`frame_count`, scalar `fps`, and native `width`/`height`. Frame indices and ranges are relative to that selected clip. The hash binds the complete source file, or the naturally ordered frame-directory names/content hashes using `identities.source_identity`. Scalar FPS validation does not prove variable-frame-rate timestamp alignment. Shots are contiguous half-open `[start,end)` ranges covering the complete selected clip.

Example structure (replace hashes/geometry with the actual local values):

```json
{
  "schema_version": 1,
  "revision": 1,
  "source": {
    "sha256": "<actual source SHA256>",
    "start_frame": 0,
    "frame_count": 100,
    "fps": 24,
    "width": 1920,
    "height": 1080
  },
  "shots": [[0, 100]],
  "entries": [
    {
      "id": "matte-24", "revision": 1, "frame": 24, "kind": "matte",
      "asset": {"path": "masks/matte-24.png", "sha256": "<actual asset SHA256>"},
      "strength": 1, "direction": "both", "range": [24, 25]
    },
    {"id": "reset-60", "revision": 1, "frame": 60, "kind": "reset"}
  ]
}
```

Asset paths resolve against the document directory and must be local filesystem paths. Use grayscale coverage or RGBA alpha in uint8/uint16 images or floating EXR. RGB luminance masks, implicit resize, mismatched native dimensions, invalid/NaN/out-of-range coverage and unsupported kinds/fields are rejected. Strength is finite in `[0,1]`; zero disables the operation and its forced anchor, while referenced assets remain validated. All rasters are native-resolution, with float32 constraint math. At most two decoded native masks are retained; encoded/decode/working buffers are additional allocations, not a total process budget.

Schema/source/asset hashes are checked during session preflight before model loading. Actual probe geometry, selected range/FPS and raster decoding are checked before pipeline stages/writers; a persistent session may already own a loaded model. A changed referenced file must have its hash/revision updated deliberately; changed assets are not silently reused. Decoding verifies the exact bytes being decoded. Normal jobs do not upload or copy footage. Put correction masks in the project asset directory, separate from source frame sequences.

To author fingerprints programmatically:

```python
from corridorkey.identities import file_sha256, source_identity
from corridorkey.corrections import CorrectionDocument, save_corrections

# Fill the envelope above with actual selected metadata from the source probe.
# Fill its hashes with these values; do not use placeholders at runtime.
source_hash = source_identity("/local/source")["sha256"]
mask_hash = file_sha256("/local/project/masks/matte-24.png")
document = CorrectionDocument.parse(data, "/local/project")
save_corrections("/local/project/corrections.json", document)
```

`data` is the completed envelope, with the actual hashes assigned before parsing. Saving is atomic and rewrites asset paths for relocation; identity excludes asset location strings and includes content hashes/order/revisions/source metadata. `load_corrections(path, source=...)` verifies and reopens the snapshot. `CorrectionHistory` stores immutable in-memory snapshots, returns invalidated shot ranges from apply/undo/redo, enforces increasing new document revisions, and supports discarding the redo branch after a new edit. Save the selected snapshot explicitly. The desktop editor publishes immutable recovery snapshots and offers bounded live undo/redo. Reopening a chosen snapshot works; reconstructing the complete undo/redo stack across restarts remains future work.

## Raster effects and direction/range

Entries are applied in their authored list order. For input alpha `a`, coverage raster `m` and strength `s`:

| Kind | Direct alpha result |
|---|---|
| `matte` | `(1-s)*a + s*m` |
| `keep` | `a + (1-a)*s*m` |
| `remove` | `a*(1-s*m)` |

Outside the declared range there is no direct raster operation. A forward range starts at its authored frame; a backward range ends at `frame+1`; both-direction ranges contain the authored frame. Every range remains inside its declared shot. A range repeats the **same static native raster**, not a motion-tracked brush or inferred backward matte. Regions may need frame-specific assets for moving subjects. Ranges crossing newly supplied/detected cuts are rejected so authors can revise their shot/range metadata or the heuristic cut policy.

Direction also controls which mechanisms may use raster values:

- `both`: at the authored frame, a full matte supplies the existing one-channel model hint. Its strength scales the existing hint-quality input; where a base hint exists, ordered raster math combines the authored hint with that base. Keep/remove can refine a known first/full/carried hint, but never invent unknown pixels of a partial hint. Both-direction constraints also refine predicted auxiliary alpha before hint carry and condition Quick Key propagation anchors.
- `forward` / `backward`: static raster constraints are applied to final alpha only. Their pixel values do not enter model hints, auxiliary carry or flow anchors. This explicitly does **not** claim forward/reverse neural propagation. These modes provide useful fixed-region protection/removal over an authored time range without pretending the model supports partial-mask presence channels.

Authored hints are applied after existing first/full/carried hints in every relevant native window. A missing base plus a partial keep/remove mask leaves the neural hint absent; output constraints still work. Conditioning follows the inherited bfloat16 hint representation; direct native output constraints use float32. Direct alpha editing does not solve reconstructed foreground colour again; foreground may still need visual review, and conditioning can change network RGB in ways requiring real-model validation.

Quick Key keeps raw neural alpha separate from both-direction flow conditioning and final output constraints. Neural/anchor frames get their complete ordered direct constraints exactly once. Directional static masks cannot be accidentally baked into flow state outside their range. Propagated non-anchor alpha can already contain earlier both-direction conditioning before current-frame constraints are applied. Every active authored/reset frame is a mandatory Quick Key anchor, including causal mode; source RGB remains current-frame draft RGB. Extra anchor/context work is counted. Changing anchor placement can affect other estimates even when a direct range is short.

## Resets and dependent invalidation

A reset entry has only `id`, `revision`, `frame` and `kind: reset`. It begins a fresh temporal segment: RVM state and predicted hint carry reset, and Stage 1 windows do not cross it. Declared shot starts also reset state, including in full-rate mode. Existing first-frame hints retain frame-zero meaning; a reset does not implicitly reapply them.

Correction/range/direction/strength/order/asset/reset changes can affect temporal estimates outside direct raster ranges. `invalidated_ranges(old,new)` conservatively invalidates the **entire affected declared shot**; source/timeline/shot-partition changes invalidate the complete old/new extent. Do not substitute a small neighbourhood for recurrence, carry, anchor or bidirectional dependencies. IDs are stable; the canonical document/asset identity participates in provenance and future cache keys. A complete inference cache key must also include model/assets, preprocessing/colour, precision/backend and temporal policy.

Normal CLI/session/desktop runs use job-local caches. Opt-in recovery can reuse verified completed/intermediate data only under identities binding correction manifest bytes, document/asset content and full runtime context; changed correction content cannot match an old identity. The editor does not automatically replay or relabel historical exports. The desktop editor now displays these invalidated ranges, and durable recovery identities bind correction content. This does not imply direct-seek/Stage2/writer resume or automatic corrected-output replay from the editor. Undo/redo returns the same conservative dependent ranges.

## Evidence and remaining validation

CPU tests verify immutable canonical snapshots and portable identities, save/reopen, changed-source/assets, strict schema, ordered soft operations, uint16/RGBA coverage, unknown-hint handling, exact correction outputs in both memory paths, RVM/carry resets, mandatory anchors, interval-1 corrected parity, preflight/session reuse, conflicting cuts, cancellation and keeping directional raster values out of flow state. Runtime neural modules are deterministic substitutes; no trained-model visual improvement or GPU speedup has been measured.

Job diagnostics include the actual revision/content identity, bound source metadata, ordered entries/asset references, hint/aux/propagation/refinement work and policy. `correction_preflight_seconds` includes schema/content hashing outside processing time but inside session elapsed/batch wall time. Benchmarks additionally fingerprint the manifest and actual assets before measurements; each measured job still performs its own validation. Real CUDA/weights/held-out footage, encoded output timing, visual comparison, interactive native UI and full persistent/direct-seek recovery acceptance remain open work.


Core JSON read/save admission is now4MiB with strict duplicate/finite/exponent validation; oversized atomic saves preserve existing destinations. `CorrectionHistory(max_snapshots=N)` optionally bounds live history (integer>=2); its default remains unlimited. The desktop selects128 live snapshots and retains immutable on-disk authored versions/assets. See `correction-editor.md` for exact bounds, UI workflow and verification limits.
