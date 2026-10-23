# Architecture review

**Architectural status: BLOCK.** Independent architect lane reviewed HEAD `7d808e61d8a42d77e30792ded6911aabca96a08f` plus dirty worktree. Scope assumes a local operator, sequential jobs per worker, long sequences and optional recovery, not a hosted multi-tenant service. Final consolidated severity is in [findings-register.md](findings-register.md).

## Boundaries and dependencies

| Boundary | Inspected source |
|---|---|
| CLI/batch → immutable requests | `infer.py:234–266`, `batch.py:15–81` |
| Request → single-owner model/session | `session.py:32–86,98–135,148–306` |
| Worker → sequential shared engine | `worker_core.py:41–100,109–242` |
| Versioned IPC/bounded transport | `worker_protocol.py:13–166`, `worker_transport.py:24–104`, `worker_client.py:38–219` |
| Output admission → native writes | `output_paths.py:29–56,208–339`, `runtime.py:991–1009`, `video_io.py:919–962` |
| Durable queue/explicit recovery | `queue_store.py:211–424`, `recovery.py:124–368`, `recovery_store.py:202–511`, `recovery_journal.py:84–154` |
| Serialization/JSON boundary | `diagnostics.py:45–101`, `local_json.py:13–78` |

NumPy/OpenCV handle image/tensor bridging; PyTorch plus Transformers/timm and local foundation loaders own neural work; safetensors owns safe tensor storage; SciPy/einops/utils3d support numerical operators. Qt is optional and loaded inside the desktop entrypoint. Shared session calls the existing runtime; there is no second GUI inference engine. Local framed JSON IPC separates UI ownership from model execution. Native OFX consumes prepared caches rather than embedding the engine. The architecture is directionally sound; safety invariants fail at composition boundaries rather than requiring a wholesale rewrite.

## ARCH-001 — batch-wide preservation enforced only per job

**Independent lane severity P1; consolidated priority subject to final review.** `batch.py:71–79` rejects identical output-root strings but not planned writes against other jobs' dependencies; `batch.py:108–121` invokes individual sessions. `session.py:215–221` and `runtime.py:1001–1006` protect only the current job's source/hints/corrections. `output_paths.py:208–214` permits unrelated existing outputs to be replaced; `video_io.py:954–962` writes EXRs directly.

Executed proof used two64×64 PNG sources and the existing synthetic CPU neural fixture, with the actual shared runtime and EXR writer. Job A's planned alpha output was Job B's first-frame hint. Both output roots differed. The batch completed two jobs successfully, but the hint alpha changed from0.9 to0.4000000059604645; Job B logged loading the overwritten hint. No hostile filesystem race or real model was needed.

**Impact:** loss of an admitted input and silently changed downstream result. **Root cause:** job-local protection cannot enforce batch-wide immutability. **Fix:** preflight the union of all batch source/hint/correction/document/model dependencies against every planned writer before loading models or writing anything, including direct and worker batch entrypoints; reuse existing alias/layout checks. Do not infer intentional dependency chaining. **Effort M; confidence high.** Regression requires cross-job input/hint/correction and link aliases rejected before mutation, retaining ordinary noncolliding rerender behavior.

## ARCH-002 — growing report inventories can abort accepted renders

**P2, confirmed.** `output_paths.py:106–140,154–174` counts children during enumeration and again while protecting them against shared `MAX_SCAN_ENTRIES=100000` (`:17`). It inventories generated exports as well as inputs. `worker_client.py:396–411,449–454` repeats this work on progress publication, nominally every0.25s; validation errors propagate to worker close (`:469–495`).

Executed production-limit proof:17000 source entries → initial safe separate report accepted; add17000 alpha and17000 foreground output files → same report rejected with `report protection inventory entry limit exceeded`. These were empty local inventory fixtures, not neural/video measurements. Separate5000-file probe:10025 `_info` calls per admission, measured0.170/0.168/0.176s. This establishes scan cost and export-growth rejection, not production GPU slowdown.

**Fix:** retain admitted dependency protection and protect output subtrees directly, with cheap destination/identity revalidation and explicit refresh rules outside hot event draining. Do not simply raise bounds or remove alias validation. **Effort M; confidence high** for rejection and call cost. Test large expected export growth and repeated progress, maintaining source/hardlink/symlink guards.

## Contracts checked without new findings

- Dataclass **instances**, not classes, serialize (`diagnostics.py:51–52,85`); direct/nested class rejection tests exist.
- Heterogeneous supported sets sort by canonical JSON (`:65–74`) rather than incompatible Python comparisons; multiple hash seeds covered.
- Session concurrent run/close requests refuse and model-touching failure discards resident state (`session.py:155–156,270–299`).
- Worker waiting requests/history/message/event buffers have explicit limits.
- Recovery store mutation owns an exclusive workspace context and staged publication.

No dead-code or dependency-removal claim was substantiated. Avoid abstraction churn: repair batch invariants in the shared responsible layer. `runtime.py` and UI orchestration remain large integration surfaces; file size alone is not a defect. Historical ledger/static debt needs triage, not mass refactoring.

## Validation and limits

Architect executed `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /private/tmp/corridorkey-compatible-venv/bin/python -m unittest test_diagnostics test_session test_output_paths test_report_paths -q`: **50 passed,0.325s**. New failure cases are not covered by those passing tests. Existing Python3.12.15/NumPy1.26.4 environment used; user `.venv` preserved. Coordinator's full suite/static evidence is separate.

Desktop, model math and native host specifics were independently assigned. No Windows/GPU/Resolve acceptance follows from these architecture checks. Broad recovery/queue code was inspected, not every failure branch exhaustively proved. Unbounded journal reads were noted as further hardening investigation, not a confirmed exploitable finding in this register.
