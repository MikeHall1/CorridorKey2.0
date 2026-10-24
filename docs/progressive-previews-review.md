# Progressive preview integration review

**Historical one-frame implementation review.** Current supported conditioned full-context semantics supersede the old hint/correction refusals and temporal restart below; see [current guide](conditioned-progressive-preview.md) and [checkpoint](checkpoint.md). Original results are retained, not rewritten.

Eighth extension batch, 2026-10-23. Main distinct review and independent
scheduler, execution and actual-UI reviews accept this documented local scope. Full product acceptance remains
incomplete, including unsupported preview contexts and external GPU/quality work.

Reviewed Python mapping SHA256:
`3f5fc42e94b3b4a42775e7fdea218c0767af6db6bab17781e1b644f858d65f53`.
Exact386-file mapping: `benchmark-results/ext06-connected-final/source-before.json`.
Parent checkpoint:f387223f7ad135d3d3ff898e5f1edb35bed98ec1. Focused source hashes
and initial failed evidence are retained in `benchmark-results/ext06-focused/`
and `benchmark-results/ext06-integration-initial/`.

Main distinct self-review inspected the scheduler priority ordering, sole pending
owner, context/content versus unique-attempt identities, cache/work admission,
focused-result retention, cooperative cancellation, stale rejection and immutable
requested/effective JSON. It reviewed all worker action predicates: preview shares
normal sequential model lifetime, capacity/seen/outstanding/lost/terminal handling;
recovery remains explicit; DesktopBridge never prepares durable preview records.
QueueStore ignores unknown preview request IDs before serialization/writes.

Execution review inspected strict original job validation, same-frame admission,
explicit restarted temporal context, no silent Quick Key/hint/correction removal,
colour/precision/quality preservation, deliberate draft scale, exact requested FPS,
source stat/ctime/content observations, bounded model metadata, output estimates
and owned-workspace cleanup. UI review followed capture → preparation → same-owner
inference → terminal → bounded decode → cleanup → cache admission. Export settings,
queues, original media and model selection remain unchanged. Normal render
submission disables/cancels preview admission. Worker identity changes fence reuse.

Findings306–314 were classified before remediation. Provisional API/optional-owner
and fixture typing/path expectations were corrected without suppression. Typed
preparation double-faults preserve both errors and retained workspace. Original
capture tokens fence Disable/Re-enable; invalid current JSON rejects late capture
without a timer exception. Source ctime enters cache context, including equal-size
pixel replacement with restored mtime. Cleanup retains exact root/empty-directory
proof after marker retirement, refuses replacement/new content and permits retry;
mandatory marker-size budgets refuse before creation. Context changes/seek hide
obsolete pixels and clear their tooltip. Artifacts retain selected actual execution
observations; CPU fixtures explicitly report neural_inference=false.

Dataclass contracts were explicitly checked: context/stage/limit/request/artifact
APIs require instances, original JobRequest classes refuse before I/O, immutable
JSON accepts bounded primitive objects and rejects classes/heterogeneous sets.
Configuration conversion retains shared diagnostics.json_value's deterministic
heterogeneous-set serialization and instance-only asdict; its8 existing regressions
remain required in the full suite. No baseline/checker/dependency changes occurred.

Focused evidence:16scheduler tests PASS0.003s plus independent review;19execution
tests PASS0.352s and independent19PASS0.304s;7new worker cases plus related tests
36PASS9.801s;8 actual desktop tests PASS5.272s and independently8PASS5.049s. Touched
modules/tests Ruff/Pyright clean. Actual native fixture EXRs are decoded/displayed,
requested stage scale0.5→1 is observed, source hashes/form/export recipes remain
unchanged, preview rows stay outside the queue, and cancellation/close/cleanup
retry guards exercise real threads/owned child processes. These are CPU fixtures,
not pretrained inference, speed or quality measurements.

Limits: initial PNG/EXR sequences,4Mi native pixels/64MiB encoded per frame;
no video/VFR/Quick Key/hints/corrections. Final is the captured preview recipe,
not full-range export equivalence. Metadata pins do not authenticate models or
prove compatibility; checks occur at explicit admission/completion/focus boundaries,
not continuous filesystem watching. Runtime path reads remain separate from
observed pins; hostile concurrent filesystem mutation is not guaranteed safe.
Cache budgets cover retained payloads and declared work estimates, not whole
process/GPU memory. Workspace admission/cleanup bounds are not disk allocation
caps; native blocking calls can delay cancellation. Failed cleanup remains an
owned blocker with explicit retry. No automatic installations/downloads, changes
to the user's environment, inference defaults or source data occurred.

Actual pretrained CUDA latency/VRAM, representative visual quality, target Windows
packaging and Resolve host execution remain NOT VERIFIED. Existing static005 and
nativeQt126 remain open; a passing narrow or full run does not resolve them.

Final frozen gate: **1323CPU tests PASS91.578s/no skips**.386Python files and
full compilation/import/requirements/VCS/dependency/whitespace checks PASS,0new
Ruff/Pyright fingerprints versus unchanged baseline AND actual precedingf387223.
Existing519Ruff/684Pyright errors17warnings remain, not static-clean. The source
mapping remained unchanged throughout verification. Exact logs/source evidence:
`benchmark-results/ext06-connected-final/`. Prior default headless benchmark
3x1137CPU is reused for unchanged-driver/orchestration scope only; this new shared
source has its own full suite, with no new inference-performance claim.
