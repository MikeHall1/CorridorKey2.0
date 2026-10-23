# Bounded decoded-frame reuse investigation

This implements the investigation required by development brief §6 item 4
(CK-P06-04). It is a generated CPU reader experiment, locally verified in M44. It adds no processing cache option and loads no neural
model. A negative result is useful evidence.

Use the existing CLI and developer benchmark panel with the `decode_reuse`
manifest kind. An example is `benchmarks/suites/decode-reuse.json`. Choose an
existing artifact parent before running; the runner creates a fresh private
experiment directory. The report remains local and can be inspected/exported
through the existing benchmark tools.

The comparison uses the actual sequence readers and preprocessing functions:

- Overlapping full-rate windows use `SequentialFrameDirWindowReader` and
  `window_starts`. The reference already retains overlapping frames; those
  retained reads cannot be counted as a new cache saving.
- Separate chronological prepasses use the existing chunk iterator followed by
  the window reader. A cache smaller than the clip may thrash and produce no
  useful cross-pass hits. This must remain a measured negative result.
- Quick Key current/anchor reads use `DraftSourceReader` and the actual offline
  frame assembler with a constant diagnostic matte. This exercises reader,
  two-anchor retention and CPU motion work; it is not pretrained inference or
  measured keying quality.

Each schedule runs on generated uint16 display-coded PNGs, signed HDR float EXRs
in declared linear BT.709, and premultiplied RGBA float EXRs in declared linear
BT.2020. Existing unpremultiplication, primaries conversion, transfer functions,
padding and model/foundation/review preprocessing remain the reference.

The research candidate caches normalized native float32 RGB after existing
decode/unpremultiplication and before transfer/primary conversion. Derived views
are recomputed by the existing functions. Cache hits do not imply that model or
foundation preprocessing was avoided. The adapter is confined to generated
fixture paths and the measurement thread, restores its hooks on exit and
provides defensive ownership. It is not installed in the processing runtime.

The predeclared acceptance conditions are:

- At least three matched alternating reference/candidate pairs for every
  case/schedule, after a declared unmeasured warmup.
- Exact dtype, shape and element-bit digest agreement for all emitted
  representations. Validation uses separate untimed streaming passes with fresh
  caches; it does not retain an entire clip for later comparison.
- Separate actual decode and preprocessing call counts and inclusive component
  times. Preprocessing time contains decode time; do not add them together.
  Workload elapsed time includes the schedule's actual CPU work, including
  motion where applicable, and excludes identity/parity hashing.
- Explicit cache byte accounting, zero-budget bypass, eviction and oversized
  entries, caller mutation isolation, unchanged sources, cancellation/failure
  cleanup and restoration of the original reader functions.
- Individual samples and per-case summaries. No averaging across incompatible
  formats or schedules to create a single speedup claim.

The OS filesystem cache is uncontrolled and may be warm. No cache drops or cold
storage measurements are performed. CPU source-stage timings do not measure
end-to-end GPU inference, presentation latency or production footage quality.
No production control should be promoted from decode-count reduction alone;
useful reproducible net benefit and target-pipeline acceptance are still needed.

Research manifest bounds are 3–20 pairs, 8–128 frames, dimensions 16–512, and
`frame_count * height * width <= 2**20`. The fixed aggregate bound limits the
generated source payload to roughly 34 MiB before file headers/compression.
The candidate cache budget is 0–256 MiB. Retained logical array payload excludes
returned arrays/stacks, Python metadata, decoder/flow allocations, allocator
retention and the OS cache; it is not a total RAM limit. Native decode calls can
remain uninterruptible until the codec returns.

## Recorded local result

M44 used the actual shared CLI in an owned child: 12 frames at 64×64,
262144-byte candidate budget, three alternating matched pairs per row.
All representations matched exactly in separate untimed validation arms.
Peak retained cache payload was 245760 bytes. Generated inputs and frozen
source files remained unchanged; all owned processes/threads terminated.

| Format | Schedule | Decodes reference → candidate | Median reference / candidate (ms) | Median paired candidate/reference |
|---|---|---|---|---|
| uint16_png | overlap | 12 → 12 | 5.017 / 4.841 | 0.9648 |
| uint16_png | sequential_prepasses | 36 → 36 | 13.300 / 13.041 | 0.9916 |
| uint16_png | quick_current_anchor | 17 → 12 | 18.819 / 18.009 | 0.9570 |
| signed_hdr_exr | overlap | 12 → 12 | 5.643 / 5.666 | 1.0042 |
| signed_hdr_exr | sequential_prepasses | 36 → 36 | 15.719 / 15.785 | 0.9982 |
| signed_hdr_exr | quick_current_anchor | 17 → 12 | 19.884 / 18.941 | 0.9507 |
| premultiplied_bt2020_exr | overlap | 12 → 12 | 7.513 / 7.577 | 1.0086 |
| premultiplied_bt2020_exr | sequential_prepasses | 36 → 36 | 20.960 / 21.189 | 1.0148 |
| premultiplied_bt2020_exr | quick_current_anchor | 17 → 12 | 22.399 / 20.752 | 0.9307 |

The last column is the median of individual paired ratios, not the ratio
of the two median times. Overlap already avoids repeat decoding; independent
chronological passes thrash this five-frame cache. Quick Key saves five decode
calls, but these short synthetic samples do not establish a production benefit.
No production cache control was added.

Evidence: `benchmark-results/m44-actual-decode-summary.json` and its exact
referenced report retain raw samples, provenance, inventory and validation.
The actual Qt developer panel inspected/exported that report byte-for-byte
without a rerun (`m44-benchmark-panel-summary.json`). Final integrated checks:
897 CPU tests/no skips, compile/import/dependency/whitespace checks, 310 Python
files, no new static findings against the baseline or M43. Existing 519 Ruff
and 689 Pyright errors/17 warnings remain. Independent scoped review and main
integration review are recorded in `m44-review-summary.json` and
`m44-verification-summary.json`.
