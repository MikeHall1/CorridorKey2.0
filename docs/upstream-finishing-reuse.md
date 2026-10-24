# Opt-in upstream reuse after global final coverage/composite edits

Current dispatch preserves the legacy **v2–v4 IID factory**. Enabled RVM/Hann
use a separate **temporal-only v5** parent under the same default-off flag, described
below. Earlier all-three-stage policy sections remain historical/direct-reader
contracts; they do not broaden IID numerical scope.

`reuse_upstream_finishing_edits` defaults to **false**. It permits explicitly enabled durable IID foundations, chronological RVM chunks and raw Hann Stage1 events to survive edits to exactly seven global final-only settings: `alpha_expand_radius`, `alpha_feather_radius`, `despill_strength`, `despill_screen`, `despill_space`, `despill_preserve_luma` and `despill_edges_only`. At least one of `durable_foundations`, `durable_rvm` or `durable_hann` must also be enabled. It does not enable any cache or backend implicitly.

The independent `reuse_stage2_finishing_edits` option remains Stage2-only and still requires `durable_stage2`. Enable both policies and all relevant durable stores to reuse every admitted neural stage. Unenabled stores compute normally. Cache quota, eviction, corruption and exact-call mismatches can require recomputation; no fixed speedup or resource reduction is promised.

## Use and inspect

Both recovery CLIs accept `--reuse-upstream-finishing-edits`:

```sh
python -m corridorkey.recovery run --batch /local/batch.json --workspace /local/cache \
  --max-cache-bytes 1073741824 --durable-foundations --durable-rvm --durable-hann \
  --reuse-upstream-finishing-edits --durable-stage2 --reuse-stage2-finishing-edits
```

`python -m corridorkey.worker run` accepts the same options with the usual required `--report`. The desktop recovery panel provides a separate default-off upstream checkbox. Saved queue requests preserve both opt-ins immutably; old requests missing the new switch remain disabled. Normal inference outside explicit recovery is unchanged.

Seed and edit runs must use matching recovery policies. Make a deliberate new request after editing these global final controls and run without `--resume` to create a fresh attempt. The previous attempt/output is preserved. `--resume` is still governed by the original immutable-request journal rule; changing policy alone does not authorize resuming a different request. Full-output lookup, writer-prefix identity and final output fences include the complete current job and enabled policy. Edited requests never restore the old final output or its writer prefix.

Reports record `reuse_upstream_finishing_edits`; each enabled adapter records `parent_kind=upstream_finishing_context` and its own hit/miss/evaluation counters. A disabled/absent RVM model is **not exercised**, not proof of RVM reuse. Restored historical output timings are not fresh inference measurements.

## Identity, compatibility and rollback

`recovery_identity.UpstreamFinishingContext` is a distinct versioned typed domain derived from an admitted full `ReplayIdentity`. Its upstream reuse scope is explicit; it retains source/content/order/range, hints, corrections, settings, schedule, models/assets, code, writer, profiling, environment/backend/device/precision and the complete enabled recovery policy. Conservative documents retain private context **version2**, removing exactly the seven listed global final settings; eligible pure render-region documents use version3 and retained-entry schema5 documents use version4 below. Version1 payloads remain readable with their original two-field policy and canonical keys, never reinterpreted as version2. Embedded shot-profile payloads and their bindings remain conservative dependencies; this is not general edit dependency analysis.

IID frames additionally bind exact frame/RGB bits, dtype, geometry and feature inventory. RVM chunks bind exact RGB and incoming recurrent-state bits, dtype and shot/chunk bounds. Hann events bind exact inputs, schedule and reconstructed incoming carry/overlap state, and reject active Stage1 recurrent outputs. Raw tensor storage and journal formats do not change. The new domain produces new cache keys; old entries/seals have no explicit rewriting, migration, forced deletion or salvage; ordinary managed quota/eviction still applies. The first new-policy run seeds its own entries safely.

Loaded model binding and adapter validation callbacks still check the **full current job**, not the reduced cache context. Existing ownership, checksums, quotas, bounded temporary stages and cancellation cleanup are reused. Disable the new checkbox/flag to return to full-job upstream cache identity; disabling it does not delete earlier outputs or user data.

## Verification and limits

`tests/test_upstream_finishing_recovery.py` exercises actual `run_recoverable` A → finishing-edit B → uncached clean B using deterministic CPU model fixtures. It verifies fresh attempts, old-A/source preservation, zero old-prefix restore, decoded/native EXR equality, seed evaluations, corresponding upstream hits/zero fixture model calls, and independent Stage2 policy. It covers each upstream flag combination plus low-memory, spatial, Quick Key and forced neural fallback paths. The RVM stage uses the established chronological prepass fixture, not an absent RVM model.

`test_upstream_finishing_identity.py`, `test_upstream_finishing_adapters.py` and `test_upstream_finishing_transport.py` cover conservative dependency binding, exact state, corruption, missing-middle recovery, zero quota, cancellation, asset/context drift, recurrent-output rejection and public/CLI/wire/queue/UI admission. Existing default and Stage2-only regression suites remain applicable.

Decode, hashing, transfers, recurrent/Hann state reconstruction, rendering and writing still run. IID partial misses can change GPU producer batch grouping and therefore roundoff, as already documented by the IID adapter. CPU fixture parity is **not** pretrained GPU numerical parity, representative-footage quality, Windows/Resolve acceptance or wall-time performance evidence.


The five despill controls affect only validation, output names, final-render
metadata/rendering and provenance; no IID/RVM/Hann inputs change. The checker
basename can change with strength. Full `ReplayIdentity`, finished writer-prefix,
output/journal/provenance bindings remain current and complete; no old-prefix
restore is permitted for the edited job. All model/assets/source/hints/corrections,
raw RGB/recurrent/carry state, dtype/device/autocast, policy, scheduling, writer,
environment and implementation dependencies remain unchanged. No correction
manifest/path/SHA/document projection is added.

Enabled shot profiles continue binding the base recipe, including global despill.
A stale profile is rejected; no rebinding is performed to skip validation. A new
valid embedded profile may conservatively miss because the entire profile payload
is retained. Shared opt-ins remain independent and default off.

The CPU recovery fixture exercises each of the five valid despill edits, combined
coverage/despill edits, no opt-ins, independent Stage2/upstream policies and all
upstream durable-flag combinations. It verifies exact clean-B exports, unchanged
alpha and signed/HDR raw foreground for despill-only edits, corresponding fixture
neural calls avoided only with their enabled reuse policies, and fresh outputs.
Changing `despill_space` alone does not change already-linear EXR rendering.
Real GPU numerical quality, footage benefits and performance remain NOT VERIFIED.

## Guarded pure render-region edits (private v3)

The existing default-off upstream policy also admits a **nonempty schema5**
document made exclusively of composite/despill and/or diagnostic/tracking-overlay
regions, including completely validated accepted region tracks. An explicit
private-v3 projection retains document presence, exact source mapping/geometry/FPS
and shots, while dropping only those final-only region details and correction path
from the optional neural reuse context. The complete original wrapper/document
identity is structurally checked before projection; unknown dependencies reject,
not disappear. This is not authentication or filesystem-integrity admission.

The full current job/path/manifest SHA/document/assets, writer-prefix, journals,
provenance, profile payload and validation remain unchanged. Current assets and
live decoder/cuts are still admitted; full context is checked at bind, each raw
entry retains owner/cancel and exact-input/state checks, and final publication
re-identifies the full current job. No skipped whole-job preflight or stale output
publication is authorized by a raw cache hit.

Source/shots/presence remain dependencies because pure region documents still
control shot cuts, Hann segments, Quick anchor plans and RVM resets. Empty/absent/
older-schema documents retain exact factory-v2 keys; pure-v3 behavior remains
unchanged and retained-entry schema5 documents use v4 below. Direct v1/v2/v3
formats remain unchanged; old seals have no explicit migration or forced deletion; existing managed quota/eviction still applies. No automatic
profile rebinding or first-document equivalence is implemented. Stage2 reuse remains independent. Actual CPU fixtures cover pure
mask/tracked-provenance/path edits and clean-export parity, plus conservative
mixed/shot/reset and exact-state/integrity negatives; GPU/quality/performance
acceptance remains external and NOT VERIFIED.


## Mixed and retained-entry schema5 documents (private v4)

The same independent default-off policy emits v4 for a strict schema5 document
with at least one retained nonfinal entry, including when no final region remains.
It retains **all ordered canonical nonfinal entries**, their IDs/entry revisions,
disabled controls, asset paths/hashes, reset/trimap/refinement semantics, complete
guidance/resources/bindings, source/shots and presence. Only proven composite and
diagnostic region entries, all accepted derived tracks, outer document revision
and the private path are removed after complete original wrapper/document/track
validation. The filtered document is strictly reparsed with revision0/tracks[];
no binding is rewritten to make it fit. Pure-v3/direct v1–v3 remain unchanged.

Derived ordinary masks are downstream-only in the current neural pipeline, but
retiring them can change final alpha. The editor retirement rule is **unchanged**.
Raw reuse therefore renders the current B recipe and must match clean B, never
restore A's finished output. Input-sensitive uncached state/motion/schedule/alias
characterization and actual cached-versus-clean CPU tests cover this distinction.
Stage2 still needs its independent flag; unenabled stages compute normally.

Full current assets/source/track/guidance admission, bind-time complete context,
per-get/put ownership/cancellation and final complete job re-identification remain
mandatory. Journals, writer prefix, outputs, provenance and profile recipes remain
full-current. First-add/last-remove reuse is within an already present schema5
retained-entry document, not a None/empty/older-schema equivalence. Retained path,
guidance, hint, state, scheduling or source/shot changes still invalidate/reject.
No general graph, new engine/cache/flag, old-seal migration or GPU/speed/quality
acceptance is introduced.


### Stage2-only v5 is a separate qualification

[Stage2 private v5](stage2-recovery.md#exact-input-qualified-stage2-context-private-v5)
can reuse whole matching RGB/baton calls across correction and first/full hint
edits. This does **not** broaden this upstream policy: upstream factory v2/v3/v4
payloads/keys remain unchanged, retaining original hints and all nonfinal authored,
source/shot/guidance/state dependencies. Exact per-entry input/state guards still
apply. Do not infer IID/RVM/Hann GPU producer-batch roundoff equivalence from a
matching Stage2 parent or CPU fixture. Full-current admission and publication,
fresh full-identity writer prefixes, independent flags and current rendering remain
mandatory for both policies.


## Temporal-only RVM/Hann qualification (private v5)

The existing `--reuse-upstream-finishing-edits` flag selects a new temporal parent
**only for enabled RVM/Hann**. IID continues using the unchanged
`upstream_finishing_context` v2/v3/v4 factory and its full hint/nonfinal authored
semantics. Only required enabled parents are constructed. No new flag, algorithm,
cache, child wire/seal format or generic cache base is introduced.

`UpstreamFinishingContext` validates exact version/scope pairs: versions1–4 retain
`iid_rvm_hann_raw_inputs`; version5 requires `rvm_hann_exact_inputs`. Both the IID
adapter constructor and direct IID identity builder reject the temporal scope
before use. The scope accessor is derived once during validation and reads in O(1),
without per-frame JSON parsing, new dataclass serialized fields or key/equality
changes. Stage2 direct v1–v5 and upstream direct v1–v4 payloads/keys remain exact.

After strict original correction-wrapper/document/all-track/source binding and
hint file/ordered-sequence structural/aggregate validation, only the private
correction record/path and known first/full hint slots are replaced by closed
versioned temporal tags. Unknown hint siblings/settings and all source/model,
assets/code/environment/backend/device/precision/writer/profile and other policies
remain bound. Projection performs no filesystem I/O; original full-current source,
hint, authored/derived/guidance assets and accepted binding validation still runs.

Each RVM chunk binds actual RGB, exact incoming state, shot/chunk geometry and dtype.
Each Hann event binds complete raw Stage1 inputs, global schedule, incoming carry,
accumulators/weights/pending frames and segment. Hits replay **raw** output only.
Current corrections run afterward and rebuild current aux carry: a late anchor with
an early active range can leave the first raw Hann event identical while changing
its outgoing corrected aux, so the next event must miss. Cached corrected carry is
never substituted. Only consumed raw aux alpha is sealed by the existing Hann
format; unused producer aux channels are not an extra replay contract. Active
Stage1 recurrent sites remain unsupported/rejected. Later reset/cut edits can
repartition earlier RVM chunks or alter the whole Hann schedule; conservative
prefix misses are correct, not grounds to relax schedule/state guards.

Full-original bind callbacks, per-get/put ownership/cancellation and final complete
job re-identification remain unchanged. Current writer-prefix keys, journals,
destinations and provenance stay full-identity/fresh-attempt; old finished prefixes
are not restored after edits. Current constraints and rendering run exactly once,
and earlier exports/source remain intact. No explicit old-format rewriting,
migration or forced deletion occurs; ordinary managed quota/eviction remains.

Generated input/state-sensitive CPU fixtures capture immutable real raw-call,
returned state, raw aux and current carry-transition digests. They establish the
uncached witness before namespace changes, then cached B equals clean B exactly
while B differs from A, across full/low/spatial and proportionate Quick/adaptive
fallback paths, independent stages/flag-off and current integrity/cancel guards.
These are correctness cases, not a GPU or throughput benchmark. Trusted loaded,
immutable eval models are required; arbitrary mutated/custom models are outside
this contract. IID GPU producer-batch rounding equivalence, real pretrained quality,
Windows/Resolve and hardware performance remain **NOT VERIFIED**.
