# Reusable export playback indexes

Play sequence now prepares bounded source/foreground/alpha path indexes off the
GUI thread, on the existing preview loader, before decoding export playback.
Each distinct resolved directory is indexed once per preparation. Foreground and
matte have independent indexes, and synchronized A/B preparation is atomic. The
existing source-global to output-local frame mapping, colour interpretation,
photo/solid/checkerboard compositing, loop/speed and visible-frame captions remain.

Subsequent frames reuse natural filename ordering and frozen membership; they do
not scan the directory again. Ordinary seek/pause/resume and compatible requests
reuse the prepared bundle. Refresh sequence indexes explicitly drops source and
export indexes; Play inspects membership again. New independent frame files are
ignored until Refresh. If refreshed output count differs from the captured
completed-render record, playback refuses rather than inventing a new range.

Each used file and directory is checked around reading with sampled filesystem
identity, size, mtime and ctime where provided by the OS. Changed/missing files or
replaced directories stop playback and request Refresh. There is no automatic
reindex or hidden content/quality fallback. Existing read-only symlink aliases and
hardlinks remain supported; retargeted roots refuse through the existing original
path-to-prepared-source association check. Stat fences detect ordinary mutation,
not cryptographic content authenticity, hostile metadata manipulation or portable
ctime semantics. Native Windows filesystems still require actual validation.

Preparation admits at most six unique directories, 100,000 inspected entries and
frames in total, 32 MiB path bytes, and a conservative 64 MiB metadata estimate.
These limits apply together across A/B and all streams, including unsupported
entries inspected by the scanner. Combined accounting and conservative initial
reservation can admit fewer frames/path bytes than each individual ceiling.
Request metadata is limited to64 KiB. The metadata estimate is not a hard process
RSS limit. One cached export bundle is retained; an active older request or a
source-only index may retain bounded references until retirement. No pixel cache,
persistent codec reader, new thread or neural owner is introduced.

Native image decoding retains its previous geometry/memory scope; there is no
new4M-pixel cap or resize for export indexing. An actual4K EXR alpha fixture is
read successfully, establishing local decoder mechanics rather than4K throughput.
Photo-background admission remains a separate explicit limit and still does not
support native4K. Video/VFR export playback remains unsupported. Manual still
previews retain their existing read path; this increment indexes playback only.

Play creates a paused timeline before asynchronous preparation so frame/range and
clock state remain explicit. It starts pixel decoding only after accepting a
matching prepared generation. Pause, selection/mode changes, Refresh and Close
fence old requests and retain the one sequential loader through terminal delivery.
A later unrelated photo/still terminal can prove an obsolete queued preparation
retired without swallowing that terminal. Closing waits for the existing loader.

Focused validation in the existing compatible CPU environment:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen /private/tmp/corridorkey-compatible-venv/bin/python -W error -m unittest test_preview_exports test_export_playback_pipeline test_desktop_export_playback test_desktop_playback test_desktop_source_playback test_desktop_photo_background -v
```

Core10actual-image/4K cases,31new/existing pipeline cases and33expanded actualQt
cases pass. Instrumented repeated reads show zero full directory rescans after
preparation; this is an operation count, not measured inference or decoder FPS.
Final frozen **1456 CPU tests PASS in 105.303s, no skips**; 410 Python files pass compilation, imports, dependency consistency, static regression and whitespace checks, with zero new diagnostics against the baseline and actual preceding 03d9e9a checkpoint. See [review](export-playback-indexes-review.md). Actual
GPU, representative quality, native Windows and Resolve acceptance remain NOT VERIFIED.
