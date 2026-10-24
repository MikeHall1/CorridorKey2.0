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

Schema2 now supports foreign Windows/POSIX archives when every render dependency
is collected. Paths are validated lexically in their producing flavour, then
collected files and correction resources become native only after verification
and explicit Apply destinations. Original wire/model paths remain archival
provenance and are never probed as native paths. Partial/reference-only foreign
packages refuse; general foreign missing-reference relinking remains unsupported.
Direct foreign publication refuses: Apply to an explicit native draft and create
a fresh verified native plan. This is a reader extension, with no changed writer
layout or implicit manifest migration. [verification policy](code-health.md).
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
[verification policy](code-health.md). Reproduction commands in the existing
compatible environment are:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen python -W error -m unittest test_portable_project test_project_collection test_desktop_portable_project test_desktop_window_portable test_desktop_state -v
python -m unittest discover -s tests -v
```

Real moved-package fixtures do not establish native Windows packaging, model
compatibility or pretrained/GPU/Resolve acceptance. Those remain NOT VERIFIED.


Foreign Apply requires explicit absolute native output/scratch paths; native
package destination behavior remains compatible. Video jobs with a foreign
absolute encoder refuse without replacement. EXR preserves that unused reference.
Bare/relative executables remain unverified environment references; models are
never automatically activated. Single-origin lexical paths must be unambiguous;
device namespaces, relative drives, dot segments, controls and unsafe collection
names produce explicit compatibility errors. POSIX literal backslashes/colons in
parent names remain valid. Public plan/inspection/file-pin records are checked
for exact native associations before I/O; Apply also rereads bounded pinned
manifest metadata to preserve actual package settings provenance.

**1408CPU tests PASS103.288s/no skips**;401Python files/full compilation/import/dependency/static regression/whitespace gate PASS,0new vsbaseline AND actualc357cad. Windows-flavour moved
package/offset/hint/schema4 raster/Qt cases run on POSIX payloads, not native
Windows. Native Windows filesystem/packaging/runtime acceptance remains NOT VERIFIED.
