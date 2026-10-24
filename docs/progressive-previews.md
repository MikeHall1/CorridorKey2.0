# Progressive inference previews

Current full-context sequence behavior and evidence: [conditioned preview guide](conditioned-progressive-preview.md), [checkpoint](checkpoint.md). The original1323-case one-frame review is [historical](progressive-previews-review.md), not current inference semantics. The desktop **Progressive preview…** panel is disabled
by default. Opening it does no source, model or GPU work. Choose a separate
existing scratch parent, review the current workspace settings and display mode,
enable the feature and explicitly **Capture settings and start**.

The workflow supports PNG/EXR image sequences with explicit declared FPS and
captured In/Out, including initial/full hints and existing corrections. Video/VFR,
Quick Key and derived shot profiles remain explicitly unsupported. Required
Alpha/Foreground display exports must already be selected; no output is added
silently. Jobs use fresh owned EXR/scratch destinations without changing queued
requests or normal exports. Each selected request processes the **complete captured
timeline**, preserving recurrent/carry/Hann/whole Stage2 context, then captures only
that frame. No early-stop/latency improvement is claimed. Final means the last
configured preview stage, never automatic final-export promotion. CPU sensitive
fixtures prove selected clean-full parity for supported settings, not pretrained
quality or GPU responsiveness.

An optional explicitly selected draft scale must be below the captured processing
scale. The current frame gets its configured draft/final ladder first, followed
by a surrounding two-frame radius and up to four upcoming playback frames.
Surrounding upgrades run after one second idle when enabled. Optional **selected
absolute source range** (inclusive In, exclusive Out) and **remaining captured
requested range** controls add final-preview-recipe coverage after nearby work.
Both are off by default, captured explicitly with settings, and operate only
after one second idle with the background-work checkbox enabled. Selected range
coverage precedes remaining range coverage. Bounds must lie within the captured
job's source-global requested In/Out, not export-local thumbnail indices.
Priority refers to
admission after the active request retires; native inference cannot be forcibly
preempted. Normal exports disable/cancel previews and take priority on the same
sequential engine owner. No second GPU model owner or durable preview queue is
created. Preview requests consume the existing worker generation's bounded
request-history capacity; use the normal Stop/New worker workflow if exhausted.

Range admission is deliberately limited to256 distinct source frames per
capture (the entire captured requested range when Remaining is enabled), and
1024 range attempts including superseded/cancelled work. The core permits
explicit limits up to1024 frames/1024 attempts; the desktop uses the defaults.
Larger coverage refuses with an actionable smaller-range error before model
work. The worker generation still has its existing4096-request history shared
with other jobs: this is not reserved capacity, and exhaustion fails explicitly;
the panel never restarts a model/worker automatically. Visible current/nearby
work keeps its existing queue/work admission limits.

Two lazy cursors and bounded final-frame terminal acknowledgements track range
coverage, not a whole-range queue or retained job/result history. Accepted or
failed background frames are not rendered repeatedly after cache eviction or
seek; deliberate focus may render an evicted visible frame again. Failed range
frames are counted separately and are not retried implicitly. The panel shows
completed/total, failed and admitted-attempt limits; request-budget exhaustion
requires explicit recapture. Changing scope/settings/source/worker requires a
new capture and rejects stale completion. Cancel/disable stops range work while
retaining the active owner until cleanup. Range completion records coverage,
not permanent thumbnail retention, temporal export equivalence or automatic
draft promotion. Range results remain transient preview-only artifacts.

Source/frame, captured requested settings, explicit effective preview settings,
model-reference metadata and temporal limitations are visible in the panel.
Pending, draft, final, stale and failed states are distinct. Invalidating context
or changing the playhead hides obsolete pixels until validated current work is
available. Late inference, decode and capture results cannot override newer
requests, including Disable followed by Re-enable during a pending capture.
Re-enabling alone does not start work; capture again explicitly.

The scheduler is independent of Qt and inference. It keeps one owned request
through terminal delivery and thumbnail decoding, with at most64queued tasks,
16retained artifacts/32MiB payload cache,8MiB per artifact and16KiB per artifact
metadata. Actual displayed thumbnails are at most512pixels on their longest
side. The default64MiB declared work reservation is a native float-buffer
estimate, not a measured or enforced Torch/RAM/VRAM ceiling. The source-admission
limit is4M native pixels and64MiB encoded bytes per frame; sequences allow100000
direct entries/frames under the existing bounded index path limits.

Model references use bounded metadata pins (up to2048entries/512KiB paths), not
content authentication or compatibility proof. All source/context and hint/correction/asset content hashes plus pinned metadata
fence admission and completion; total dependency reads are bounded to512MiB per pass.
These checks run off the UI thread and can incur substantial IO. Runtime opens its
usual paths; these observations do not pin source bytes through every native
read or defend against hostile concurrent filesystem owners. Model-cache writes
may conservatively require a new capture. Worker replacement/settings/source/
model-selection changes require recapture. Cache reuse requires exact captured
context, stage and source frame; unique attempt identities fence late artifacts.

Each request creates an exclusive private workspace. Native-output estimates and
filesystem free space are checked before submission; cleanup admission allows
256entries/512MiB and bounded nesting. These are admission/cleanup limits, not a
hard filesystem allocation cap. A blocked native call can delay cancellation.
Cleanup failures retain the exact workspace and stop scheduling; **Retry retained
cleanup** acts on that same verified owner. Parent shutdown waits for owned work
and does not report unverified cleanup as complete. Original media, models,
exports and unrelated files are never cleanup targets.

Focused reproduction in the existing compatible environment:

```sh
PYTHONPATH=tests:. QT_QPA_PLATFORM=offscreen python -W error -m unittest tests.test_conditioned_preview tests.test_progressive_preview tests.test_preview_execution tests.test_worker_preview tests.test_desktop_progressive -v
```

CPU fixtures prove portable scheduling/ownership/display and selected clean-full
contract parity with sensitive neural substitutes, not pretrained visual quality.
Actual pretrained CUDA draft/refinement latency, visual quality, memory behaviour,
Windows packaging and full product acceptance remain NOT VERIFIED.
