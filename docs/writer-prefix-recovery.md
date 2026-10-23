# Durable EXR writer-prefix recovery

This optional recovery policy saves complete, contiguous groups of EXR output
frames in the existing recovery workspace. It defaults to off. Enable it in the
desktop recovery panel, or add `--durable-writer-prefix` to the recovery or worker
batch command with an explicit workspace and cache byte budget. Reuse requires
an explicit resume request; starting a worker does not submit or resume a job.

Each saved frame includes every selected output role. A writer queue barrier
waits for earlier native EXR writes to return. The owner then checks regular
files, calls file/directory fsync, hashes their bytes and publishes a checked
all-output bundle. Observation-only profiling callbacks do not establish this
receipt. Fsync calls do not prove power-loss durability on every filesystem.

Resume copies only the longest verified contiguous prefix into a fresh attempt
directory. Missing, corrupt or incompatible entries stop prefix restoration.
Original attempts remain separate. Source, model, settings, correction, colour
and output contracts participate in reuse identity. Copied outputs are checked
again before success and recovery publication so ordinary local changes cannot
be accepted as the verified prefix.

All chronological source processing, recurrent state reconstruction, neural
processing, refinement and correction work still runs. The policy suppresses
only the native write submissions for the copied prefix. It supports EXR output;
other formats refuse before model loading. It does not provide inference seeking
or a promised processing speedup.

Logical frame counts cover the full selected timeline. Recovery diagnostics
separately report copied files/bytes, sealed frame groups and newly submitted and
confirmed native writes. Counts after failures may be confirmed lower bounds.
Render-ready profiling covers the full replay; queue and native timing
distributions cover only newly written frames. Copied frames have no fabricated
timestamps. An entirely copied output has no new-write percentiles.

Cache byte quotas and shared workspace entry/file/metadata limits bound capture.
Retained copy-verification receipts are separately capped at 32,768 files and
16 MiB of encoded receipt metadata; reaching either limit stops restoration at
a complete frame boundary, and the remaining frames use ordinary native writes.
When capacity is exhausted, further prefix capture stops while rendering
continues. An already full or unmanageable workspace can refuse before starting
a new attempt; select a fresh workspace or use the existing maintenance tools.
Global LRU eviction can remove earlier groups, so cumulative sealed counts do
not promise that the same contiguous prefix remains available later.

Local acceptance uses deterministic CPU models, real EXR files and an actual
offscreen Qt worker interruption/resume workflow. Production GPU performance,
pretrained output quality, Windows filesystem durability and Resolve host
behaviour require their own acceptance evidence.
