# Current checkpoint — sequence tracking and source playback, 2026-10-23

Fourth connected increment locally accepted; all ten workflows and full product
acceptance remain incomplete. Authority: [extension brief](product-improvements-brief.md).

[Tracked corrections](tracked-corrections.md) now connect artist anchors and
forward/backward/both native motion to background Preview, read-only Inspect,
explicit Accept and history. Schema3 keeps authored assets and stores separate
float32 EXR final constraints, ordered once with source/anchor/decode/boundary
provenance and heuristic diagnostics. Undo/save/reopen/invalidation, cancellation,
failed publication, derived dependency guards and bounded cleanup are tested.

Source-only sequence Play prepares a bounded frozen natural-order path/stat
index on the existing decoder, independent of renders/models/queue. Reads use
exact paths without per-frame directory scans. Explicit refresh, source offsets,
loops, pause/resume, stale preparation and nonneutral EXR colour are tested.
Video/VFR, photo backgrounds and persistent export readers/cache remain pending.

Independent reviews accepted kernel, schema, editor/UI/runtime and playback.
Findings267–281 include typed fixtures, anchor/cleanup ownership, read-only canvas,
persisted observation and snapshot aliasing, source metadata union narrowing,
confirmed-root-exit sampling, report-envelope overflow and failed-operation cleanup
visibility. Root-exit RSS remains unavailable; PIDreuse/denial/unknown discovery
and final descendant/group verification remain failclosed. Near-limit schema1/3
manifests now fit terminal records through bounded summaries referencing full local
provenance. No baseline/checker/environment/venv changes or model/media downloads.

Final headless benchmark **3×1137CPU tests PASS/no skips**:
77.876/75.657/75.751s unittest durations; median full CLI process76.496377s.
These are CPU test orchestration timings, not inference performance. The initial
sandbox run failed15localhost permission cases per repetition and spanned test
additions; it is not an accepted frozen baseline. The first fullgate exposed the
root-exit race and5new type findings; raw failures remain recorded. Final rerun
uses unchanged benchmark defaults and the required local-fixture execution context.

355Python-file compile/import/dependency/static/whitespace checks PASS,0new vs
baseline and actual prior checkpoint. Existing519Ruff/684Pyright errors17warnings
remain; static005/nativeQt126 remain open. Exact reviewed frozen source identity:
`bfab1efbd403904f270f05adcfcaa46cd1a0885b031cd9827ae1380994d9d4a4`.
Raw source/full verification: `benchmark-results/ext02-tracking-ext05-final/`.
The three benchmark full suites cover the unchanged final source, so no redundant
fourth complete suite was run. Main distinct self-review covers both briefs,
ownership, serialization, compatibility, source/time/colour and quality contracts.

Actual pretrained CUDA, representative motion/occlusion/quality footage, optional
segmentation model, native Windows and Resolve SDK/viewer/export remain
NOT VERIFIED. No full acceptance or speed/quality improvement is claimed.

Prior ae381bb9668e266fa2bf8078ad9ddca8f48af0ed normally pushed and confirmed to
MikeHall1/CorridorKey2.0 branch review/corridorkey-audit-20261023. This checkpoint's
exact publication identity follows in local ignored receipt
`benchmark-results/ext02-tracking-ext05-publication.json`.

Next: optional clean-plate assistance and strict generated-guidance reference
provenance/invalidation; continue other incomplete features. Next-batch work stayed
outside the repository during verification. Five cycles remains a checkpoint
interval; no competing supervisor was launched.
