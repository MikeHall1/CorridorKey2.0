# Portable projects

The desktop **Projects…** menu groups the existing Open/Save desktop snapshot
actions and the new portable-project workflow. Legacy schema1 snapshots retain
their format and CLI/API behavior. Saving them now requires readable bounded
attached correction metadata so every referenced asset, including tracked masks
and generated-guidance resources, can be protected from accidental overwrite.
Existing hardlink aliases are protected too.

Portable schema2 captures the complete original job and selected preview frame,
dependency identities and model references. Inventory is explicit and reads
metadata only. It lists source, first/full hints, correction documents and every
authored, tracked and guidance resource. Source sequences include all direct
supported frames, preserving their original names, order and selected offsets;
collection does not trim to the selected render range. Unrelated sidecars,
environments, outputs and caches are excluded.

Select dependencies to collect, or leave all unchecked for reference-only export.
Verification and copying have explicit byte budgets. Verify produces an immutable
reviewed plan; Publish creates a fresh exclusive package and writes completion
metadata and an exclusive empty completion marker last. Metadata sync and final
fences precede marker creation; failed precommit packages cannot open even when
cleanup is blocked. Directory-entry durability after power loss and hostile
concurrent filesystem writers are not guaranteed. A changed dependency refuses publication. No large media or model
is copied automatically. Initial model/foundation references are metadata only;
they are neither copied nor declared tensor/runtime compatible. Accepted guidance
provider weight hashes remain generation provenance, not render dependencies.

Open a package to preview missing references and collection status. Deliberate
relinks select replacements for dependency IDs; Verify requires the exact stored
file list/content identity before readiness. Changed content with the same name
does not match. Relative collection paths stay inside the package; unsafe names,
traversal, aliases and filename collisions refuse. Initial export requires all
render dependencies available and readable within the verification budget. A
reference-only package can later detect and relink missing media after moving.

Apply to draft requires explicit output and scratch destinations. Inspection is
read-only; application revalidates inputs and creates a fresh relocated correction
snapshot in the selected scratch area. Only typed dependency paths and explicit
destinations change. Source identity, correction semantic identity, settings,
selected frame offsets and original requested job remain recorded. The package,
source files and existing correction files remain unchanged. The main workspace
acknowledges whether draft application succeeded; a changed correction draft or
invalid payload refuses without partially changing fields. Generated correction
snapshots remain available for review and are not removed automatically after a
possible application.

Applying loads draft settings only. Existing queued jobs and the current model
owner are preserved. Review model references in Local assets and deliberately
select a compatible local environment before rendering; metadata presence/hashes
do not establish neural execution or target-platform compatibility. Background
inventory/hash/copy/relink work has one owned operation, cooperative cancellation
and a close guard that retains ownership until it retires. GUI summaries are
bounded, paged off-thread and retain collection/relink choices; complete provenance
remains local in the manifest. The desktop represents preview frames through1e9;
larger int64 portable frames refuse before application without changing the stored
value. A late committed correction snapshot is retained for review when Cancel or
Close prevents draft application; no late job is emitted or automatically rendered.

Initial portability supports the same native path flavour only. Windows↔POSIX
manifests cannot currently be opened/relinked, even when all media was collected;
this is an unsupported capability, not a passed test or just missing hardware.
Model roots/weights remain external references; selecting/relinking a compatible
model environment is deliberate through Local assets. The initial dialog admits
0–65536MiB verification/copy budgets (defaults256MiB read/0copy); the core accepts
explicit integer byte budgets below2^63. Read budgets cover admitted payload
verification per pass, including reauthentication at Publish/Apply; these are
accounted extents, not measured filesystem I/O. Metadata has separate4MiB
correction/32MiB project bounds,100000files and32MiB path bytes. Unsafe/reserved
basenames and casefold/NFC collisions refuse, including reference-only export.

This connected increment is locally verified:1273CPU tests PASS83.370s/no skips,
379Python files and full compile/import/dependency/static-regression/whitespace
gate PASS,0new diagnostics. Existing519Ruff/684Pyright errors17warnings remain.
Evidence: `benchmark-results/ext09-connected-final/`; independent and main
[review](portable-projects-review.md). Reproduction commands in the existing
compatible environment are:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen python -W error -m unittest test_portable_project test_project_collection test_desktop_portable_project test_desktop_window_portable test_desktop_state -v
python -m unittest discover -s tests -v
```

Real moved-package fixtures do not establish native Windows packaging, model
compatibility or pretrained/GPU/Resolve acceptance. Those remain NOT VERIFIED.
