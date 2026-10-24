# Native-pixel edge inspection

The existing desktop viewer defaults to its unchanged **1280-pixel thumbnail**.
Select **Native pixels · still inspection** to inspect one full image frame
without thumbnail resizing. Use **1:1 decoded pixels**, wheel zoom and drag pan;
A/B share the transform. The native label means one source-image pixel per
**logical Qt canvas pixel**, not a calibrated physical monitor pixel. Native
magnification uses nearest pixels, without smooth pixmap interpolation. Fit can
still reduce the displayed image to the available canvas.

This is view-only: no engine settings, queued jobs, render outputs, project
schemas or persisted configuration change. Switching modes pauses playback,
fences stale requests and clears incompatible pinned display snapshots. An
existing synchronized comparison is decoded again with the same mode on both
sides. Native autoplay is explicitly refused; choose still frames manually or
disable Native pixels before Play. Video has no native-mode fallback.

Native mode supports local PNG/JPEG and admitted scanline HALF/FLOAT EXR stills
and image sequences. Before decoding each selected source/foreground/alpha image,
the existing native admission helper captures encoded bytes, checks its header,
and enforces **16 Mi-pixels /256 MiB encoded per image**. Decoding uses those
captured bytes, then checks actual geometry/channels and file fingerprints.
Prepared sequence indexes retain their before/after selected-file stat checks;
unprepared directories use the existing bounded sequence scan. Unsupported
headers, aliases, changed files and over-budget frames fail rather than resize.
These limits are not a RAM cap: full-frame decoding, display/composite arrays,
Qt copies and A/B can require substantially more memory than encoded files.

Alpha, checker phase, solid backgrounds and prepared photographic backgrounds
use the full existing geometry and colour/compositing rules. Photographic
background preparation retains its own existing smaller pixel/work limits;
native mode does not enlarge those bounds. Signed/HDR values
clip only in the final display conversion; input/export files are not changed.
The viewer does not provide selective/ROI decoding, HDR monitoring, content
provenance authentication or GPU/visual-quality certification.

Focused tests include actual one-pixel patterns wider than1280, native UHD
3840×2160 PNG/EXR, immutable file hashes, pre-decode admission, file mutation,
limits, coverage/checker alignment, mixed-mode refusal and offscreen Qt
publication/thread/zoom/pan/A/B/playback fencing. Test timings are not performance
measurements, and no reduced-memory or GPU-quality claim is made.
