# Managed recovery cache maintenance

Open **Manage recovery cache…** in the desktop queue toolbar. Select an existing owned recovery workspace, then **Inspect inventory**. Inspection uses an existing read-only SQLite exclusive transaction; it does not construct a BundleStore, create a workspace, prune staging, refresh entry timestamps or apply a budget. An active controller/maintenance owner refuses. A normal rollback-journal lock database must already exist; WAL and unsupported/malformed ownership are refused.

Set **Retain at most bytes** to an exact decimal integer (0 through9223372036854775807), then **Preview trim**. Zero selects every sealed bundle. Preview lists least-recently-used bundle candidates, apparent bytes before/after and the exact proposed removals. All candidates are accessible through100-row pages. Changing workspace/target invalidates the old preview. **Apply preview** explicitly deletes that reviewed plan; it does not submit renders or change future recovery budgets. SQLite ownership and the full bounded metadata/stat snapshot are revalidated before the first removal; busy, changed or stale previews refuse without deletion.

Only recognized complete tool-owned sealed entries are eligible. Sources, render exports, journals, reports and staging are preserved. Metadata/ownership/lengths are checked; payload checksums are not read by inventory, so this is not restore/inference validation. Accounted bytes are apparent file lengths, not filesystem allocation, reserved space or guaranteed reclaimed free space. Total process RAM/VRAM/disk budgets remain separate requirements.

All filesystem/SQLite work runs on one owned background thread. Cancel is checked between entries; deletion of a bounded individual entry finishes before cancellation is observed. Actual already-removed bundles remain removed. Failures and interrupts preserve an authoritative terminal result with committed removal counts and a partial-entry identifier when relevant. If an exceptional filesystem denial prevents reconciliation, byte totals are explicitly unknown. A partial entry remains in place for deliberate repair, and later inventory refuses its incomplete metadata; it does not guess how to delete unowned/incomplete content. Window close waits for actual owned I/O and pending terminal delivery. No GUI thread performs hashing/SQLite/directory scans.

Bounds:4096 sealed/staging entries,32768 files total,4MiB per metadata record,16MiB total metadata,4096-character canonical workspace path. Unexpected files, links, Windows reparse points, multiple hardlinks or names refuse rather than expanding a deletion boundary. SQLite excludes cooperating writers; this is not a security boundary or a transactional guarantee against hostile external edits or abrupt process termination. Actual Windows filesystem/packaged application behavior remains unverified.

Headless use with the existing authorized environment:

```sh
python -m corridorkey.cache_maintenance inspect --workspace /local/cache
python -m corridorkey.cache_maintenance plan --workspace /local/cache --target-bytes 0 > /local/trim-plan.json
python -m corridorkey.cache_maintenance apply --workspace /local/cache --plan /local/trim-plan.json
```

The shell redirection is an explicit operator choice; the module prints bounded JSON and does not create a hidden plan file. Apply validates the plan's exact immutable token/snapshot. Choose a new plan destination to preserve earlier files. The explicit packaged-app dispatch route admits this command, but actual Windows bundle execution remains unverified.

Local evidence: `benchmark-results/m30-generated-summary.json` records a real recoverable controller with explicit CPU codec/neural substitutes, actual native EXR exports, and desktop inventory→preview→apply. Two sealed bundles totaling16080 apparent bytes were removed after preview;11 source/export/journal/report files were unchanged, no maintenance renders started, and screenshots were inspected. A first layout artifact was recaptured after Qt processed its deferred relayout; no product source change was required. This is correctness evidence, not pretrained quality or GPU throughput. Independent review/fault probes and final554-test/full-static scope are recorded in `m30-verification-summary.json`.

Portable recovery-workspace backup/restore and queue relocation are subsequent requirements. A directory copy alone does not safely relocate absolute dependency or queued-batch identities.
