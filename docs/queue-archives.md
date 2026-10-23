# Sealed queue history and fresh workspaces

Queue history now supports **Archive stopped history**, **Archive + fresh queue** and **Open last archive**. These controls preserve old requests, batch snapshots, source paths and outputs. They never resubmit work or delete records/media. A new workspace resets the4096-request admission count without pretending total disk usage is bounded.

## Workflow

Stop the active worker and wait for its owned connection/journal to close before archiving that queue. Stopping an active render follows existing cancellation/partial-output behavior; archiving does not stop or cancel it for you. A live journal owner causes archive refusal without changing its state. Old PIDs do not establish whether any process is alive.

**Archive stopped history** seals the selected queue against future managed writes. It stays in its original directory, preserving absolute batch/snapshot references. Read-only paging, selected detail checksums and settings loading remain available. **Use for new jobs** is disabled and the core refuses writer ownership, including explicit New worker against that archived root. A separate fresh queue is required.

**Archive + fresh queue** seals the stopped selected history, creates a private sibling workspace and requests normal activation through the existing clean processing/asset-path barriers. If the selected history is already archived, **Fresh queue beside archive** only creates the new workspace. No render or recovery request is submitted. The worker factory retains deliberately selected runtime/assets. **Open last archive** reopens the preserved path as read-only history while the fresh workspace can remain active.

Sealing and fresh creation are separate steps. If fresh creation/inspection fails after sealing, the UI re-reads actual archive state, reports the committed seal, keeps new activation disabled and preserves the old path. Refresh inspects current state; an error does not prove that a seal was rolled back. No automatic retry or unsealing occurs. Creating a fresh workspace may leave its newly owned incomplete directory if I/O fails; it does not delete user files to hide that failure.

Saved results remain observations, and unresolved records remain unresolved. Archiving does not declare completion, validate current outputs or prove global process death. Loading normal settings prepares the form only; a new explicit render gets a fresh output destination. Loading recovery settings into a fresh queue explicitly snapshots the same original managed batch; actual resume/output/cache/context verification stays in the existing controller. No neural/GPU quality or performance claim follows from loading history.

## Core contract

`QueueStore.seal(Path)` obtains the same exclusive writer lock used by journal ownership, publishes a small schema1/queue-ID-bound `queue-archive.json` and closes ownership. Every future `QueueStore` writer checks archive state **after acquiring the lock**, preventing an acquisition race. `QueueStore.inspect` remains read-only and includes archive metadata for pages/details.

The marker is bounded4096 bytes/depth8 with strict finite/duplicate/schema/type/owner validation. Invalid markers or symlinks fail closed. Publication writes and fsyncs a new private stage file, then uses an exclusive hard link to expose complete bytes without replacing any existing marker. Only that attempt's stage alias is removed normally. A process crash after publication can leave a small private stage alias; the published archive remains readable and refuses managed writes. No automatic cleanup of old outputs or unknown files is performed.

Current-format database bytes are not rewritten by sealing. Opening a legacy queue can still apply the existing optional-column compatibility migration before sealing. This is a managed-write prohibition, not OS-level immutability or an authenticated/portable backup. External filesystem edits can invalidate content; selected detail checks continue to verify checksums. Media, weights and correction files remain referenced in their original locations and are not copied.

Hard-link/SQLite/filesystem behavior is tested on the local macOS filesystem. Unsupported publication fails explicitly; Windows, network filesystems and power-loss durability remain NOT VERIFIED. File fsync does not prove directory-entry survival after power loss. Explicit portable export/import of already sealed histories is now implemented separately; see [queue transfer](queue-transfer.md). Physical storage quotas, retention deletion and archive search/cataloguing remain unfinished. Existing128-header pages and4096 admission limit remain.

## Evidence

Milestone24 tests cover live ownership refusal, unchanged database/unresolved states, publication failure, interleaved writer acquisition, marker tampering, existing/concurrently created files never overwritten, process exit immediately after publication and explicit re-snapshotting of a sealed recovery request. Qt tests cover inactive controls, clean ownership, no automatic submissions, committed archive plus fresh-creation failure and asset-path separation before stopping the current owner.

`benchmark-results/m24-generated-summary.json` records actual Qt/correction editing/worker/SQLite/native EXR operations with an explicit constant-alpha CPU fixture. A dirty parent close was refused without stopping its workers, then a committed correction was saved/rendered. Archive/rotation preserved database bytes, original native outputs and the immutable correction snapshot, submitted no work automatically, and allowed a later explicit render loaded from old history to a fresh destination. Captures were inspected. These are correctness observations, not pretrained neural quality or GPU throughput evidence.
