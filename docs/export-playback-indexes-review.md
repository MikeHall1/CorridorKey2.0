# Export playback index review

This EXT-05 playback-only increment is locally accepted; full product and external acceptance remain incomplete.
It captures immutable view/mode requests without I/O, prepares deduplicated bounded
sequence indexes on the existing loader, and supplies source/foreground and separate
alpha indexes to unchanged native decoding. Manual still reading remains unchanged.

Core10actualPNG/EXR/4K cases PASS0.867s/-Werror. Pipeline6new+25existing cases
PASS0.447s/-Werror. Main expanded33actualQt cases PASS5.444s/-Werror; focused
Ruff/Pyright0. Independent16core/pipeline cases PASS0.806s and6actualQt PASS1.495s.
No speed/RSS/GPU/quality claim follows from these test durations. Evidence:
`benchmark-results/ext05-export-focused/`.

Classified findings:

- 351: request identity used asdict before its init=False derived field existed;
  explicit validated payload now computes the cached identity without recursion.
- 352/353: narrow import grouping fixes in new integration/fixtures.
- 354: explicit Any boundary represents deliberately invalid limits classes in
  negative tests while preserving runtime class/instance refusal and typed APIs.
- 355: two new tests unpacked the pair wrapper incorrectly; loader behavior was
  already correct. An independent run overlapped these test-only edits and is
  retained as invalid frozen evidence; final explicitly frozen runs supersede it.
- 356: actual added-file Refresh correctly refused frame coverage drift; the new
  Qt assertion expected an unpromised word count. It now checks the actual coverage
  refusal while retaining frozen-before-Refresh and empty-cache assertions.
- 357: three old native-pixel fixtures blocked the GUI before asynchronous index
  preparation could finish. Existing event-driven until waits now reach the real
  gated decode, preserving all ownership/stale/pixel/source assertions. The current
  decode-error test also waits for actual error after preparation, avoiding a vacuous
  pending_serial=None check. Independent fixture diff review found no weakening.

Independent review queried forged same-count swapped stream bindings and retargeted
root aliases. Actual operational previews already refuse them through the original
path/SequenceIndex source association guard. That proposed defect was retracted;
no unnecessary source change was made.

Main distinct self-review against AGENTS.md and both briefs checked complete request/
index/binding admission, dataclass classes versus instances, heterogeneous sets,
mutable nested fields and canonical serialization. Request and bundle identities
are cached; GUI binding never hashes the full path table. Aggregate limits and
ctime observations are scoped as metadata estimates/ordinary mutation fences,
not hard RAM or content-authentication guarantees. Source/output offsets and alpha
indexing remain distinct; the actual image/colour/geometry path is unchanged.
Paused kernels retain old native decode ownership until terminal acknowledgement;
source/export/photo preparation share one bounded queue and superseded terminal
routing. Refresh is explicit and no model/queue/export setting changes occur.

Initial reviewed11-file mapping:
`c04e430c0cde68f96cd9ab515dd68b2feeae7be2aebaeb38f2db63a07311f94d`.
Independent stable source and fixture review finds no unresolved local blocker.
358: the initial full gate ran1456 tests in104.244s with one existing hard-crash
restart fixture error. Lazy descendant discovery raced os._exit; production correctly
refused restart with unverified cleanup. The test now establishes actual complete
discovery under the existing lock while its ready child is idle, then retains the
real crash/lost/restart/new-generation assertions. No production cleanup changed.
Existing no-discovery/unknown-group refusal tests remain; focused worker/ownership
checks pass. A temporary redundant negative test was removed after confirming
existing equivalent coverage; no production or checker change. Independent review
accepted the exact ten-line fixture change. Expanded12-file reviewed mapping:
`57e15ae8454229b89ac134cc3e5d53e333bad184d392ffcdf8a370ef23d3ae25`.

Final frozen **1456 CPU tests PASS in 105.303s, no skips**; 410 Python files pass compilation, imports, dependency consistency, static regression and whitespace checks, with zero new diagnostics against the baseline and actual preceding 03d9e9a checkpoint. Source identity `843028c3e418f7bd427640cbaaecb502480698d3ab03e4965e2de1c5becca53a` is unchanged through the gate.
Failed first gate remains in `benchmark-results/ext05-export-connected-final/`;
accepted final evidence is `benchmark-results/ext05-export-connected-final-fixed/`.
Prior unchanged-driver3x1137CPU headless benchmark reused solely for applicable
orchestration scope; no inference/decoder FPS or quality claim. Existing static005/nativeQt126
remain open; actual CUDA/quality/Windows/Resolve remain NOT VERIFIED. No dependency/
environment/default/precision/colour change, download or asset publication.
