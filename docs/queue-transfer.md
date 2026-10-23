# Portable archived queue history

In **Queue history…**, stop the writer and explicitly **Archive stopped history** first. Open **Export / import history…**, select the archived workspace and an existing destination parent, then **Export to new container**. To import, switch the operation, select the exported container and an existing parent, then **Import to fresh history**. Each operation creates a unique child without overwriting existing history.

**Use imported history** opens the result for reading. It remains archived: **Use for new jobs** stays disabled and core writer admission refuses it. Loading render or recovery settings only prepares the corresponding form. Create or activate a separate fresh queue and explicitly Run/Verify to submit work. Unresolved observations stay unresolved; old PIDs and saved results never establish live-process state or current output validity.

## What travels

The original SQLite history, owner/archive markers and managed recovery batch snapshots travel as exact bytes. Original request/event envelopes and hashes are never rewritten. A separate bounded checksummed `queue-transfer.json` records each original historical batch path and its copied `batches/<request_id>.json` location. Read-only detail inspection validates the original request/event hashes, original mapping identity and local snapshot checksum. It returns a separate transient `load_payload` for explicit recovery loading; displayed `request` remains historical. It never falls back to the old snapshot path.

The imported workspace has a fresh zero-byte SQLite ownership lock. Archive and relocation metadata are written before the original owner marker is atomically published last. A partial unmarked destination cannot be admitted as history or a writable queue. The relocation marker requires an archive and independently prevents writer admission if the archive is missing. Re-exporting imported history preserves original historical paths and can be imported again without either previous queue.

Footage, hints, correction dependencies, models, recovery caches and rendered outputs remain at their original paths and are not copied or verified by transfer. Recovery still verifies current dependencies and exports through the existing controller. Native path flavour must match: POSIX-to-POSIX or Windows-to-Windows. Cross-platform dependency conversion and automatic rebinding are unfinished; incompatible history is refused rather than rewritten.

## Headless commands

```sh
python -m corridorkey.queue_transfer export --workspace /existing/archived-queue --output-parent /existing/parent
python -m corridorkey.queue_transfer inspect --archive /existing/queue-export-CHILD
python -m corridorkey.queue_transfer import --archive /existing/queue-export-CHILD --output-parent /existing/parent
```

The packaged dispatcher uses `--corridorkey-module corridorkey.queue_transfer` in place of `python -m corridorkey.queue_transfer`. Terminal JSON reports `status`, actual `destination`, `usable`, copied byte/file counts, request count, error/warning and scope. Exit zero means complete. Partial destinations remain visible; cancellation after a verified committed publication reports complete with a warning. Unknown accounting is null. No automatic deletion, retry, activation or submission occurs.

## Admission and limits

Export takes actual read-only exclusive cooperating ownership of the existing queue lock, accepting a legitimate zero-byte SQLite lock and refusing busy ownership or WAL. It validates every request, session association and batch snapshot, not just the first history page. It refuses unexpected/orphan files, links/reparse points/hardlinks, changed snapshots, invalid hashes and malformed managed paths. The flat numbered-payload container avoids general archive extraction. Destination parents within the source or recorded media/assets/workspaces/exports are refused before creation; a fresh sibling under a common ancestor is allowed.

The database is bounded to256 MiB, individual observations and snapshots to4 MiB, requests to4096, transfer manifest to8 MiB and relocation envelope to4 MiB. Copies/hashes stream within admitted lengths with an extra-byte growth check before any over-budget write. These admission bounds are not a physical disk quota, free-space reservation or total RAM/VRAM budget. Failed copies may leave partial children. Cooperating ownership and atomic marker publication do not prove hostile-editor transactions, network-filesystem support or power-loss durability. Windows runtime acceptance remains unverified.

Desktop transfers run in an owned background actor with immutable captured inputs, cancellation and bounded terminal delivery. History I/O and transfers exclude each other. Parent/main close waits for actual owned work, including pending terminal results. Actions require an actually highlighted history row; clearing selection cannot load a stale focused row.

## Verification

The actual local Qt/shared-worker/native EXR workflow is recorded in `benchmark-results/m32-generated-summary.json`: render and recover, archive/export, make the original queue unavailable, import/read/load, then explicitly resume in a fresh queue. Imported database and historical requests, moved original history, export container and six original media/configuration files stay unchanged. The explicit resume verifies existing exports, and the fresh queue creates its own independent snapshot. The worker uses an explicit CPU neural substitute with a separate local asset directory. An initial fixture placed assets inside the owned batch directory and correctly triggered unexpected-file refusal; the private demo fixture was corrected without relaxing product admission (`m32-demo-fixture-admission.txt`).

Focused tests cover repeated export/import with old roots removed,130-row paging/detail traversal, immutable history, fresh snapshots, ownership, corrupt/extra/linked/traversal inputs, destination preservation, cancellation, disk failure and late publication. This is local correctness evidence; GPU quality/performance, actual Windows packaging/filesystems and Resolve acceptance remain separate.

Final integrated verification:599 tests/no skips pass in44.396s;244Pythonfiles, full applicable CPU/import/dependency/compile/whitespace/static gate passes with zero new findings. Existing static backlog retained. Independent review and source/config/environment identities: `benchmark-results/m32-verification-summary.json`; logs `m32-final/`.
