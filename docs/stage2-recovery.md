# Exact Stage 2 prediction replay

The current default-off finishing-reuse factory emits private **v5**, described below.
The v1–v4 sections record preserved direct-reader policies, not the current factory.
The legacy upstream IID factory remains conservative v2–v4; enabled RVM/Hann
now have separately scoped temporal v5 qualification under their existing flag.

The optional workflow is locally verified with actual Qt/worker/shared CPU orchestration and generated EXRs. It reuses verified prediction batches; it does not provide direct seeking, writer checkpoints or a measured GPU speedup.

Prepare a normal recoverable EXR batch with explicit scheduling. In **Recovery…**, choose the workspace and retained cache budget, then enable **Durable Stage 2 predictions** before **Run**. After interruption, inspect the actual failed report and explicitly choose **Verify/resume** with the same immutable batch and recovery flags. No automatic retry, submission or flag changes occur. Old requests load with this mode disabled.

The CLI flag is `--durable-stage2` on `corridorkey.recovery run` or `corridorkey.worker run` with a recovery workspace. It is an independent recovery policy, default false; it does not enable IID, RVM or Hann caches and is not an inference scheduling setting. Worker requests and saved queue history preserve the chosen policy.

Each entry corresponds to one whole original Stage 2 call. The identity binds the validated parent model/job/backend context, ordered native source frames, call policy, actual RGB and baton tensor bits, shapes and dtypes, and execution/autocast contract. A different subset, input bit, dtype, grouping or parent cannot reuse that batch. Entries do not split or regroup predictions frame by frame.

The cache stores the selected raw Stage 2 output before precision promotion, alpha refinement, authored corrections, crop/resize or rendering. It preserves the producer dtype and tensor bits. Replayed output passes through the same downstream processing exactly once. Unavailable, corrupt or mismatched entries recompute the whole batch. No pickle or executable model payload is loaded from a prediction entry.

All source decoding and chronological reconstruction still run. Upstream caches remain separate explicit policies. Renderers and writers create fresh output destinations; earlier exports remain intact. A verified Stage 2 hit avoids that model call only. Reports distinguish actual evaluated calls/frames, reused batches/frames and completed prediction frames. Historical timings from restored exports are not new inference FPS.

The existing managed sealed-entry quota is shared with other recovery caches. Zero retains no entries. Per-entry serialization admission is bounded before host-copy/hash/staging work; oversized entries skip optional caching without changing batch sizes, quality or precision. The retained cache quota does not bound staging RAM, output size, physical filesystem overhead or total process resources. The optional sampled RSS stop policy remains active during cache operations.

Cancellation or failure before sealing does not create a usable entry. Already sealed predictions can survive a later interruption; publication and integrity checks still determine whether each entry may be reused. CPU fixtures must cover exact output parity, failure-prefix replay, corruption, grouping/input changes, quota rejection and source/export preservation. Real pretrained numerical agreement, GPU performance, Windows and Resolve acceptance require target evidence.

Verification:701CPUtests/no skips46.295s and full static/compile/import/dependency/whitespace gate pass with0new findings. ActualQt/worker gracefulcancel andhardexit/lost→explicitreconnect/resume reused2batches/3frameinstances, evaluating6frames in3calls versus9frames in5uncachedcalls. All9 outputs match exactEXRbytes and independentfloat32/64x64/finite/.4pixel oracle, with originalsource/manifest/partialexports unchanged. Three screenshots inspected. This counts completed forwards (including invalid returned tuples), not partial GPU work in a forward that raises; failure stage timings remain recorded and successFPS null.

Entry admission caps tensorpayload256MiB, safetensorsheader64KiB, metadata4MiB and flatentryinventory4096files; larger optional work can be refused or skipped with explicit counters/reasons. Same-descriptor boundedmetadata reads and expectedpayload growth fences prevent later size growth from bypassing those admission limits. Limits apply to logical/staging scope described above, not a hostile-filesystem transaction. Exactsource/evidence including independentlyreviewed199–201: `benchmark-results/m37-verification-summary.json`, gatearchive `m37-final/`. RealGPU/pretrainedquality/performance and targethost acceptance remain unverified.

## Opt-in reuse after global final coverage/composite edits

Enable **Reuse Stage 2 across expansion / feather / despill edits** together with **Durable Stage 2 predictions**, or pass `--durable-stage2 --reuse-stage2-finishing-edits` to either recovery CLI. Both default off; old requests/projects remain off. The explicit policy is captured in worker requests, durable queue history and run reports. Start a new immutable edited recipe with **Run**, not resume of the old recipe. Existing queued requests are never mutated.

The previous private raw-Stage2 parent context used **version2** for conservative documents (eligible pure render-region documents use version3 and retained-entry schema5 documents use version4 below), excluding exactly seven global final-only settings: `alpha_expand_radius`, `alpha_feather_radius`, `despill_strength`, `despill_screen`, `despill_space`, `despill_preserve_luma` and `despill_edges_only`. Outside the explicit document projections below, every other full-job dependency remains conservative: source membership/frame mapping, hints/corrections/tracking, models/assets, implementation/dependencies/device/precision, temporal scheduling, draft settings, other refinement/foreground settings and output/writer policies. Actual ordered RGB/baton tensor bits, dtype/geometry/grouping and call policy still bind each batch. IID/RVM/Hann and whole-output/writer-prefix identities are **not** relaxed. Changed full output identities and request journals preserve the previous render and write a fresh destination. Live full-job dependency revalidation and loaded-model binding remain in force.

Upstream decoding, preprocessing, temporal chronology and model loading still run; compatible raw Stage2 hits avoid only those actual forwards. Finishing, constraints, compositing and writing run anew. This is not a general incremental graph, selective neural crop, direct seek, speed guarantee or memory cap. Missing/corrupt/evicted/oversize entries recompute safely; zero budget retains none. Version2 seeds its own namespace. Version1 private context payloads remain readable with their original canonical keys and two-field expansion/feather policy; they are not silently interpreted as version2. No explicit migration, rewriting, rekeying or forced deletion of old cache entries/seals is performed; existing managed quota/eviction still applies. Disable the option to roll back without deleting history or exports. Real pretrained GPU quality/performance remains NOT VERIFIED.

Upstream IID/RVM/Hann reuse across the same seven global final edits is a **separate** default-off policy; enabling Stage2 finishing reuse alone does not enable it. See [upstream finishing reuse](upstream-finishing-reuse.md) for dependencies, fresh-attempt semantics, limits and verification.


Enabled shot profiles retain the entire embedded payload and base-recipe binding.
Editing a global despill setting does not bypass stale-profile validation. A newly
analyzed/valid profile payload may conservatively miss the old cache. Full Replay correction manifest/path/SHA/document dependencies remain complete;
only the private v5 raw-call parent narrows authored conditioning identities.

All five despill controls are consumed only by validation, output layout,
final-render metadata/rendering and provenance, not neural inputs. Despill
strength can change the checker output basename. The full current `ReplayIdentity`,
writer-prefix identity, output destination, journal and provenance still include
every setting; edited recipes create fresh attempts, never restore old finished
prefixes, and retain earlier exports. The existing full-job callback validates at bind; each cache get/put
checks bound ownership and cancellation, and final output publication re-identifies
the full current job. Raw RGB/baton/state bits, frame/call policy,
model/device/autocast, precision, writer, scheduling and all non-final settings
remain dependencies.

CPU fixture verification covers each valid global despill edit and combined
finishing edits, exact A → edited B → clean B EXR bytes/native arrays, unchanged
alpha/raw foreground for despill-only edits (including signed/HDR foreground),
corresponding zero fixture Stage2 calls/hits, fresh attempts and zero old-prefix
restore. EXR despill is already linear, so changing `despill_space` alone need not
change EXR pixels. This is reuse correctness evidence, not GPU quality or speed.

## Historical v3 policy: pure composite/diagnostic documents: guarded private v3

With the same default-off finishing-reuse opt-in, a **nonempty schema5** correction
document containing exclusively `composite/despill` and/or
`diagnostic/tracking_overlay` regions may reuse raw Stage2 predictions after region
mask/strength/range, accepted-track/provenance or manifest relocation edits. The
factory first validates the original complete correction wrapper, strict document
and complete track/anchor bindings. It checks the document's content identity;
this is structural consistency, not authentication or a replacement for current
filesystem admission.

Only eligible documents produce private context **v3**. Its closed tagged
projection retains exact document presence, source mapping/geometry/FPS/hash and
shots; only the private raw context omits that document's final-only details and
`corrections_path`. Shots cannot be discarded: they affect Hann windows, Quick Key
cuts/plans and RVM reset starts even in a pure render-region document. Unknown
wrapper/document dependencies reject instead of disappearing. Unknown remaining
settings/full-context fields remain dependencies.

Under the previous v2–v4 factory, empty/absent/older documents used v2,
pure documents used v3, and retained-entry schema5 documents used v4 below.
Those direct policies remain unchanged; current Stage2 v5 qualification differs.
Direct v1/v2/v3 payloads and keys remain readable unchanged; newer policies create their own keys,
without explicit migration, forced deletion or reinterpretation of old seals;
ordinary managed quota/eviction remains unchanged. Independent upstream reuse
has the same eligibility guard and still needs its separate opt-in.

The full current job still includes the real correction path, original manifest
SHA and complete document/assets. Current admission verifies all authored/derived
assets, source, native mapping and live track/decode/cut context before reuse.
Full-context validation runs at bind; cache accesses retain owner/cancel checks;
full current re-identification still fences final output publication. Valid raw
intermediate entries can survive a later whole-job failure, but stale completed
output seals/old writer-prefix restoration are not permitted. Writers, journals,
provenance and shot-profile payloads retain the full current identity.

CPU fixtures exercise A → edited/relocated B → clean B for both purposes, static
and accepted tracks/provenance, independent/no opt-ins and low/spatial/Quick paths.
Composite edits change only applicable final composite RGB; diagnostic edits do
not alter exports, and captured diagnostic previews still use full document
identity. Alpha/raw foreground/HDR/association, exact-entry state checks and
current-integrity/cancellation refusal remain covered. Real GPU quality and
performance remain NOT VERIFIED.


## Historical v4 policy: retained authored dependencies

A strictly admitted schema5 document with **at least one retained nonfinal entry**
used private v4 in the previous factory, even when no composite/diagnostic region
currently remains. This supports first addition and last removal without a new
flag. Every canonical retained entry stays in exact relative order with its full
ID, entry revision, disabled strength, operation, range/direction, mask path/hash,
reset, trimap or refinement fields. The complete guidance/resources/binding list,
exact source/geometry/FPS/shots and document presence remain dependencies. Unknown
fields reject or remain bound, not silently disappear.

Only supported composite/despill and diagnostic/tracking-overlay entries, **all
accepted derived tracks**, outer document revision and the private correction path
are projected away. The original complete wrapper, identity, source and every
track/anchor binding are validated first. The filtered private document has
revision0 and tracks[] and passes the existing strict parser; no entry_order,
track-binding SHA or generated-guidance binding is fabricated/rebound. Pure-v3
and direct v1–v3 payloads/keys are unchanged; old seals are not rewritten.

**Ordinary track retirement can change final alpha.** The editor still drops
accepted derivatives on an authored edit. This raw-only policy does not preserve
A's old tracked output: B's current authored/static constraints and composites
rerender anew. Accepted uncached CPU characterization found identical actual
neural inputs/raw outputs, recurrent/Hann state, motion inputs and Quick planning
across that retirement, despite deliberately different final alpha. Cached B is
checked against clean B, not A. Input-sensitive fixture controls and storage-alias
checks are correctness evidence, not pretrained/GPU quality or performance proof.

Full-current admission, model binding, per-access owner/cancel checks and final
full-job re-identification remain unchanged. Writer-prefix, completed outputs,
journals, provenance and shot-profile recipe payloads still use the complete
current job. Resource/guidance/track corruption or late manifest changes cannot
be excused by a v4 hit. Retained asset path changes and legitimately renewed
profiles may conservatively miss. No Editor/runtime/adapter behavior, general
incremental graph, public document format, precision or default is changed.


## Exact-input-qualified Stage2 context (private v5)

With the same `--reuse-stage2-finishing-edits` flag, new Stage2 contexts exclude
those seven final global controls and, after strict original admission, replace
only the correction record and known first/full hint identity slots with closed
versioned exact-call tags. The private correction path is omitted. This includes
absent, empty, ordinary, mixed and tracked correction documents; it does **not**
change any public document or whole-job identity. Known hint file identities and
ordered sequence identities must have their original exact structure, hashes and
sequence aggregate. Projection performs structural validation, not file I/O;
existing full-current admission still reads and validates the actual resources.
Unknown hint siblings/settings and all source/model/assets/implementation/runtime,
backend/device/precision/writer/profile and policy dependencies remain bound.

Reuse is sound only at the existing trusted loaded eval-model boundary: Stage2
receives RGB and baton, with optional MAE mask left `None`. Each entry still binds
the complete ordered call, actual RGB/baton bits, dtype/shape, execution and policy.
No call is split or regrouped. An early complete call can hit after a late edit;
the first affected-input call misses, and current upstream chronology/carry still
runs. A first-hint edit is not assumed to affect every later packet: only actual
matching complete inputs may hit. An authored anchor, source/shot change or Quick
plan edit never implies unchanged scheduling merely because parent keys match.

Full source/hint/correction/track/resource admission, bind-time full-context
callback, per-get/put owner/cancel checks and final full-current re-identification
remain mandatory. Output destinations, journals, provenance and writer-prefix
keys use the full current identity; edited recipes start fresh outputs with no
old finished prefix. Current constraints/regions/rendering apply exactly once.
Direct v1–v4 canonical payloads/keys remain readable unchanged. No explicit old
format migration/rewriting/forced deletion is performed; ordinary managed cache
quota/eviction remains unchanged. Upstream factory v2–v4 output is unchanged;
producer-batch GPU rounding is not assumed interchangeable for upstream features.

Input/state-sensitive generated CPU fixtures check real late both-direction
conditioning and late full-hint edits beyond observed Hann lookahead, normal/low
VRAM/spatial/Quick paths, independent flags, actual complete-input misses, exact
cached-B versus clean-B EXR/native output, changed B versus A alpha, fresh attempts,
current resource refusal, cancellation and late publication mutation. Fixture
call counts are correctness evidence, not a GPU/throughput benchmark. Real GPU,
pretrained quality and target-host acceptance remain **NOT VERIFIED**.


### Independent temporal parent dispatch

The [temporal-only RVM/Hann policy](upstream-finishing-reuse.md#temporal-only-rvmhann-qualification-private-v5)
uses the existing upstream flag, not the Stage2 flag. Shared original correction
and hint validation now lives below Stage2 in `recovery_identity`; Stage2 v1–v5
canonical payload/key behavior is unchanged. IID retains the legacy v2–v4 parent
and rejects temporal-only scope. A raw Hann hit still passes through current aux
constraints and outgoing-carry reconstruction before the next event is qualified;
no old corrected carry or finished output is restored. Full-current admission,
publication and writer-prefix/journal semantics remain unchanged.
