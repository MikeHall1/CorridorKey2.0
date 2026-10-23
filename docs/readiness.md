# Readiness and guided local render

The desktop's first launch opens **Readiness / first render…** without running
checks, loading models or downloading assets. The introduction remembers that it
was shown using Qt application preferences (`MikeHall1` / `CorridorKey2`, key
`readiness/introduction_seen`). `python -m corridorkey.desktop --readiness` opens it
explicitly; `--no-first-run` suppresses it for one invocation. These flags are
mutually exclusive. On-demand access always remains in the workspace toolbar.

**Check readiness** performs off-thread local metadata checks. Dependency versions
are not proof of native imports; asset presence is not structural compatibility
or authentication. Existing Local assets presence/probe/audit controls remain
available. Output path/permission checks are advisory and reserve no space.

**Probe backend / encoders** explicitly runs owned subprocess metadata queries for
a captured job. Backend availability does not load models or imply inference.
Video checks configured encoder help and pixel-format lists, including the
existing cutout override `prores_ks` / `yuva444p10le`. Codec-descriptor aliases can
list multiple canonical encoders; selection remains NOT VERIFIED, with candidates
reported separately. Their pixel-format lists never mix. Encoder help is not a
successful encode, hardware initialization or validation of preset/CRF/bitrate/
muxing. EXR skips FFmpeg. Requested settings are never rewritten.

Probe output is bounded to256KiB before parsing and the deadline is20seconds per
subprocess. Cancellation checks every20ms and supervises the direct child with
terminate/kill/reap. Independent children from executable wrappers are outside
this contract; use the actual configured binary. Native Windows execution is
NOT VERIFIED. CPU child tests include timeout, overflow, cancellation and a
SIGTERM-resistant process. No repeated unavailable CUDA prerequisite probe was
performed as acceptance evidence.

**Prepare short render** captures absolute source In and1–120frames (default8),
unchanged requested quality/precision/writer settings and a fresh output child.
Supplied cuts/resets rebase; moved first hints and correction subranges refuse
silent reinterpretation. Preparation starts no work. Review the displayed
snapshot, then **Run displayed snapshot** submits through the existing shared
worker/queue. Existing queued jobs and form settings remain unchanged.

A matching completed record must agree with submitted source, settings, writer,
hints, range and planned output paths before inspection enables. The capture is
independent of subsequent form edits and limited by the4MiB worker-frame bound.
Requested and observed frame counts are separate. If fewer frames are reported,
the cause/source-end truncation remains NOT VERIFIED without supporting metadata.

**Inspect completed frame** explicitly decodes one selected source-coordinate
frame off-thread into a thumbnail capped at512pixels per side. Alpha, Foreground
and checker composite use existing render view/colour adapters. Cutout-only or
checker-only exports currently explain that Alpha or Foreground must be requested;
no automatic job edit/rerender occurs. Inspection replaces displayed details, so
any newly prepared render is invalidated and must be prepared/reviewed again.

Reports distinguish engine-reported requested/effective settings, model load/reuse,
work counters and CUDA device observations from actual local output decoding and
Qt thumbnail assignment. CUDA inference is labelled ENGINE REPORTED only when
GPU/device/CUDA-version plus positive Stage2 counters exist. Provider-interface
fixtures or missing observations remain NOT VERIFIED. This does not authenticate
pretrained weights, establish quality or replace actual hardware acceptance.
Decode PASS survives a simulated Qt presentation failure; display FAIL is separate.
One inspected frame does not establish whole-clip decode/display or human/native
display acceptance.

Cancel/close/Escape/new submissions fence late inspections; native decoding may
finish before cancellation can be observed. A single background operation and
thumbnail are retained. Export saves an explicit fresh local JSON file without
pixels, overwrites or writes into protected inputs/assets. No model/media upload,
automatic installation, precision/backend fallback or changed inference default.

Validation: `python -m unittest tests.test_readiness tests.test_readiness_capabilities
tests.test_desktop_first_run tests.test_desktop_readiness -v` (one shell command).
CPU fixtures/metadata probes are not pretrained inference, real FFmpeg encoding,
Windows packaging or Resolve acceptance. Those require target hardware/assets/
local footage and remain explicitly unverified.
