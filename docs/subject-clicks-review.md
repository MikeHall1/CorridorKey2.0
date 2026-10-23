# Subject-click integration review

Reviewed frozen Python mapping `91a14a89130bb237d9002551b6b085061ad464a72e11ca8fadb0681853a17a28` in
`benchmark-results/ext04-connected-final/source-before.json`. Main integration
self-review is distinct from the independent process, editor/service and Qt reviews.
Both briefs, AGENTS, acceptance and the plan remain authoritative; this increment
is not full workflow/product acceptance.

Main review checked native coordinates/source offsets, explicit assumed-sRGB uint8
proxy conversion and native linear hash, preserved colour/precision/defaults,
one active owner, bounded framing/queues/logs, cancellation/deadlines, stale result
fences, diagnostic read-only roles, exact source restoration before painting,
failed checkpoint/staging cleanup and postcommit truth. Child cleanup uncertainty
retains ownership and blocks replacement; service done waits for retirement.
The source parent uses existing image decoding and atomic correction publication;
there is no new neural channel or fixture-backed production control.

Accepted Keep semantics are explicit: strength1 makes the coarse selection opaque,
outside unchanged; ordinary strength/range controls and tracking remain available.
Full provider provenance and fixture evidence persist. Accepted masks reopen without
provider files. Dataclass instance/class and heterogeneous-set serializer contracts
were explicitly inspected again; immutable frame/result instance checks and strict
primitive schema settings remain in force, with full serializer regressions retained.

Independent findings294–296 are fixed: invalid new prompts and predispatch
cancellation revoke old inspected masks; busy requests cannot clear active
cancellation; typed prompt/mask-format provenance must match even with matching
wire IDs. Actual malformed child tests reject/reap wrong coordinates, float/integer
coercion and wrong formats. Independent blocked-writer deadline test confirms the
child and transport threads retire. Main20process/editor PASS2.580s, independent
20PASS2.569s; independent24Qt PASS4.442s with warnings treated as errors.
Initial static fixture findings were corrected locally without suppression.

Final1232CPU tests PASS83.428s/no skips and372file full compile/import/dependency/
static/whitespace gate PASS;0new against baseline AND actual662707d. Existing
519Ruff/684Pyright errors17warnings remain. No checker/baseline, dependencies,
user environment or media changed. Prior unchanged-driver headless benchmark
3x1137CPU evidence is reused only for its applicable scope.

Actual SAM2, pretrained CUDA, representative quality/latency/memory footage,
frozen/native Windows packaging, GPU resource handoff and Resolve SDK/host
acceptance remain NOT VERIFIED. No speedup/quality/full-product claim.
