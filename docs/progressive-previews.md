# Progressive inference previews

This connected increment is locally verified:1323CPU tests PASS91.578s/no skips,
386Python files and full compilation/import/dependency/static regression/whitespace
gate PASS,0new diagnostics. Existing519Ruff/684Pyright errors17warnings remain.
[Review and exact evidence](progressive-previews-review.md). The desktop **Progressive preview…** panel is disabled
by default. Opening it does no source, model or GPU work. Choose a separate
existing scratch parent, review the current workspace settings and display mode,
enable the feature and explicitly **Capture settings and start**.

The initial connected workflow supports PNG/EXR image sequences with the job's
explicit declared FPS and requested In/Out. Video/VFR, Quick Key, hints and
corrections refuse because this single-frame preview cannot preserve their
context semantics. Required Alpha/Foreground display exports must already be
selected in the captured job; the panel never adds outputs silently. Preview
jobs use fresh EXR/scratch destinations, leaving the original export recipe and
queued jobs untouched. All other processing settings remain captured. Each frame
restarts temporal context; **final** means the final configured preview stage,
not equivalence to a full-range export or a full-quality claim.

An optional explicitly selected draft scale must be below the captured processing
scale. The current frame gets its configured draft/final ladder first, followed
by a surrounding two-frame radius and up to four upcoming playback frames.
Surrounding upgrades run after one second idle when enabled. Priority refers to
admission after the active request retires; native inference cannot be forcibly
preempted. Normal exports disable/cancel previews and take priority on the same
sequential engine owner. No second GPU model owner or durable preview queue is
created. Preview requests consume the existing worker generation's bounded
request-history capacity; use the normal Stop/New worker workflow if exhausted.

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
content authentication or compatibility proof. Source stat/ctime context and
selected-frame content checks fence admission and completion. Runtime opens its
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
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen python -W error -m unittest test_progressive_preview test_preview_execution test_worker_preview test_desktop_progressive -v
```

CPU fixture outputs prove scheduling, ownership and display mechanics only.
Actual pretrained CUDA draft/refinement latency, visual quality, memory behaviour,
Windows packaging and full product acceptance remain NOT VERIFIED.
