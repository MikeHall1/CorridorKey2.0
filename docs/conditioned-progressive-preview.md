# Full-context selected progressive previews

The opt-in desktop progressive scheduler now preserves the captured sequence job's
original Source In/count, Hann windows, recurrent state, carry, complete Stage 2
groups, hints, corrections and finishing. Each request processes the **entire
captured timeline**, but writes only the selected frame's EXR outputs into its
owned temporary workspace. There is no early stop or inference-latency improvement
claim. Long timelines can remain expensive even for a single playhead frame.

The optional draft stage changes only the explicitly selected model scale; the
final stage restores the captured scale. Both keep the original timeline. Video
sources, Quick Key and derived shot profiles remain explicitly unsupported here.
Initial hints, full hints and existing authored/derived correction assets are
captured and checked; missing or changed dependencies require a new capture.

Capture hashes all source and guidance dependency files using existing pinned-file
admission. The existing sequence/count/per-file bounds apply, with a 512 MiB total
dependency-read bound per pass. This is a deliberately bounded initial workflow,
not an implicit large-footage download or collection operation. Model references
retain the existing bounded metadata pins, not authenticity or load verification.
Repeated checks can involve substantial disk reads, performed off the UI thread.

Existing workspace cleanup bounds are unchanged. Full-context inference can use
more temporary disk and memory than the previous one-frame restart; the captured
job's `temp_map_budget_bytes` and RSS-stop policy remain effective. Cleanup
admission budgets are not allocation hard caps. A failure or cancellation never
becomes a successful sparse preview, even if the selected frame was already
written; stale results still pass the existing scheduler/generation fences.

The worker's existing `preview` action optionally accepts an absolute
`selected_frame`; ordinary preview/render requests retain their previous contract.
The shared session uses `preview_frame` as execution control. Only the built-in
runtime supports selected capture; injected custom runners fail closed. A successful
selected result has `status: preview_complete`, one submitted/captured output
frame, separate full-context processed coverage, and no export FPS. It cannot pass
ordinary completed-export admission. Output index zero maps explicitly to the
requested absolute source frame. Output telemetry counts only that captured frame.

CPU validation uses the real runtime/readers/writers with sensitive deterministic
neural substitutes, not pretrained inference:

```sh
PYTHONPATH=tests:. QT_QPA_PLATFORM=offscreen python -m unittest \
  tests.test_conditioned_preview tests.test_preview_execution \
  tests.test_worker_preview tests.test_desktop_progressive \
  tests.test_progressive_preview tests.test_progressive_ranges -v
```

Tests compare selected float32 signed/HDR outputs and complete neural-call traces
against fresh full-context renders in streaming and low-memory modes. Mutation
controls exercise hints, carry, recurrent state, corrections, future/window
context, grouping, resets and one-frame restart divergence. Actual CUDA/model
quality, interactive latency and representative-footage acceptance remain
**NOT VERIFIED**.
