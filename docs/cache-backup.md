# Recovery workspace backup and restore

Open **Manage recovery cache… → Backup / restore…**. Select an existing owned recovery workspace and an existing destination parent. **Create new backup** creates a unique child directory. To restore, select that backup, switch to **Restore backup**, and choose an existing destination parent. Restoration creates another unique child; it never overwrites a workspace. **Use restored workspace for inventory** selects the result in cache management. Recovery and rendering require a separate explicit request.

This transfers owned sealed entries, staging, journals, reports and workspace metadata. It does not copy original footage, exports, weights or correction dependencies. Historical absolute paths remain unchanged. Transfer integrity does not establish compatibility on another machine: explicit recovery still verifies source/settings/assets/environment/implementation and current exports through the shared controller. Queue history portability is separate.

## Headless commands

Use the existing compatible Python environment, or the packaged dispatcher with `--corridorkey-module corridorkey.cache_backup` in place of `python -m corridorkey.cache_backup`:

```sh
python -m corridorkey.cache_backup backup --workspace /existing/recovery --output-parent /existing/parent
python -m corridorkey.cache_backup inspect --backup /existing/recovery-backup-CHILD
python -m corridorkey.cache_backup restore --backup /existing/recovery-backup-CHILD --output-parent /existing/parent
```

Terminal JSON includes `status`, actual `destination`, `usable`, copied byte/file counts, sealed entry/journal/report counts, `error`, `warning` and scope. Exit zero means complete. Failure/cancellation can leave a partial destination; its path is retained and reported. Counts become null if actual partial accounting cannot be established. Cancellation after a verified committed publication reports complete with a warning. There is no automatic retry, cleanup, inference or recovery.

## Preservation and ownership

Backup holds the existing rollback SQLite cooperating-owner lock read-only throughout a bounded snapshot and streamed copy. It refuses a live writer, WAL mode, links/reparse points, unexpected files, changed snapshots and invalid ownership. The backup is a plain directory of numbered payload blobs with a bounded SHA256 manifest, not a general archive extractor. Each copy/hash reads only the admitted size plus a one-byte growth check; growing sources fail before additional bytes are written. The manifest is published last.

Restore checks the exact container inventory, hashes and managed path grammar. It preserves copied bytes, historical identity and entry recency, while creating a fresh SQLite lock and holding ownership through publication. Fully written/fsynced private completion bytes are renamed into place last. An incomplete unmarked root cannot be admitted as a recovery workspace. A post-publication error requires verification of actual restored metadata and sealed payloads before reporting the destination usable.

Destination parents inside the workspace/backup or recorded native absolute input, asset and export directories are refused before creation. A common ancestor is allowed for a fresh sibling. Undecodable historical journals/reports are preserved with an explicit provenance warning; unrecorded references cannot be protected. Preserve lexical paths when submitting requests so root links reach the core validation unchanged.

Bounds inherit cache maintenance: 4096 entries/stages, 32768 files, 4 MiB per metadata record and 16 MiB aggregate metadata, bounded names/paths; the transfer manifest is at most16 MiB. Payloads stream in at most1 MiB blocks. These are admission limits, not free-space reservation, a persistent retention policy or total process/storage budgets. Destination disk exhaustion may leave a partial child. Cooperating ownership does not provide transactions against hostile filesystem editors; Windows/network-filesystem and power-loss durability remain unverified.

The desktop runs copies, hashes and validation in an owned background actor. Operation inputs are immutable until terminal delivery. Close requests cancel and wait for the actual owner; child and parent maintenance operations exclude each other. Main-window dirty correction drafts still prevent shutdown before stopping owners.

## Evidence

Milestone31 final:580 tests pass in43.420s,240 Python files, complete applicable CPU/import/dependency/compile/static/whitespace gate passes with zero new static findings. Existing static findings remain. See `benchmark-results/m31-verification-summary.json` for source identities, review and scope.

`benchmark-results/m31-generated-summary.json` records the actual shared recovery controller/native EXR workflow with explicit CPU neural substitutes, followed by actual Qt backup, fresh restore, inventory selection and explicit resume. All three jobs verify existing exports with one original model load/two original renders and no additional inference. Two bundles are restored;11 source/export/journal/report files and the source workspace/backup remain unchanged. Screenshots were inspected. This proves local workflow correctness, not pretrained quality, GPU speed, Windows packaging or Resolve compatibility.
