# Diagnostic ranges and standalone rerendering

Select a completed, failed or cancelled normal render with recorded source metadata
in the desktop queue, then **Diagnostic
ranges…**. Inspection reads the Quick Key and shot trace paths recorded by that
job. It also includes recorded fallback or extra-inference events that may have
no trace row because processing stopped before yielding that frame. Recovery
batch rows require per-job report inspection and cannot be selected here.

The dialog shows suggested ranges or individual diagnostic events, observed
coverage and summary-only events. Select a row to stage absolute **In / Out
(exclusive)**, or edit those values within the recorded selection. **Preview
source In** revalidates the captured inputs in the background, then uses the
existing preview decoder with the recorded source interpretation. Trace events
are selected-clip-relative: a job starting at source frame 2 maps event frame 3
to source frame 5. No offset is inferred from filenames or guessed for old
middle/random selections without observed metadata.

Choose an existing output parent and select **Prepare range settings**. Preparation reserves a fresh `range-<uuid>` directory and loads the
new settings into the main editor. It submits nothing. Use the existing **Add
render to queue** action explicitly; that action creates its normal fresh render
child. Temporary files use a separate path under the reserved range directory.
Sources, trace files, earlier requests and existing exports are not rewritten.

**Full-rate neural processing** defaults to checked. It uses the existing full
processing mode and clears incompatible Quick Key controls. Unchecking it
explicitly preserves the parent's mode, rebasing and filtering selected-relative
cut/reset indices into the new selection. Source colour, writer settings,
profiling and supported refinement settings remain captured from the parent.
The normal full-quality and Quick Key actions remain available in the main
editor after preparation.

Changing the selected range can invalidate conditioning:

- Attached correction documents bind the original source range, geometry, FPS
  and content identity. Preparation rejects an incompatible range by default.
  Choose **Drop** explicitly to detach corrections from the new job; the original
  document remains unchanged. Rebinding or authoring replacement corrections
  uses the existing editor workflow.
- A first-frame hint belongs to the original In. Moving In rejects that hint by
  default; choose **Drop** explicitly to prepare without it. A replacement hint
  can then be selected for the new job.
- Full hint sequences retain their existing source-aligned interpretation.

A standalone subrange starts temporal/recurrent processing afresh at its In. It
is not a full-clip-equivalent patch, a continuation of the old recurrent state,
or a direct seek into prior inference. Heuristic cuts, uncertainty and fallback
events are diagnostics, not calibrated quality confidence. Full-rate processing
does not itself establish that keying quality improved.

Inspection and preparation run in one owned background task. Changing the
selected job, relevant protected output/assets context or dialog inputs revokes
stale actions, including an A → B → A selection round trip. Changes to future
editor settings during preparation prevent the result from replacing those
edits. Cancel and close wait for actual terminal delivery. If a late preparation
has already reserved an empty directory, the dialog reports that path and does
not claim the reservation never happened.

The inspector accepts at most 100,000 selected frames, two traces of at most
16 MiB each, 64 KiB per complete JSONL line, JSON nesting depth 16 and 8,192
flagged event frames. Captured snapshots are capped at 4 MiB; current dependency
directory observations are bounded at 100,000 entries and 16 MiB of metadata.
Malformed, nonfinite, duplicate, out-of-order or out-of-range trace rows refuse.
Trace files must be regular, single-link files beneath the recorded output
directory, without symbolic-link/reparse aliases. Missing or changed traces and
dependencies require inspection again before an action is authorized.

Trace hashes and current filesystem observations bind actions to the inspection.
They cannot authenticate the source bytes used by an older render when its
record did not capture that identity. Source-index alignment is not variable-rate
timestamp alignment or a colour-managed production viewer.

Local verification uses generated PNGs, real native EXRs, the shared runtime
with an explicit deterministic CPU model, actual Qt widgets and owned workers.
Pretrained quality, GPU performance, target packaging and Resolve remain separate
acceptance requirements.
