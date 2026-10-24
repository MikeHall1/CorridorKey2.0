# CFR timestamp-provider decision — 2026-10-24

**Decision:** keep video synchronized-playback admission fail-closed. A same-provider PyAV decoder is a feasible dependency-gated experiment, not yet an accepted implementation or an intrinsically impossible feature. No package, source manifest or active environment was changed.

## Evidence and reuse boundary

Existing OpenCV playback/loader interfaces are reusable. OpenCV 4.11's FFmpeg backend substitutes missing/zero presentation timestamps and quantizes them; its getters cannot establish raw PTS provenance. The generated seek probe proves internal decoder consistency only. [Pinned OpenCV implementation](https://github.com/opencv/opencv/blob/4.11.0/modules/videoio/src/cap_ffmpeg_impl.hpp#L1538-L1559).

PyAV can return decoded planes and `Frame.pts`/timebase together, avoiding a separate probe-to-preview decoder correspondence assumption. Keep the existing player, scheduler, loader and ownership. Merely probing with PyAV then displaying OpenCV frames does not solve that boundary. [Pinned frame implementation](https://github.com/PyAV-Org/PyAV/blob/v19.0.2/av/frame.py), [codec context](https://raw.githubusercontent.com/PyAV-Org/PyAV/v19.0.2/av/codec/context.py).

OpenCV inference and EXR source-index/colour correspondence remain separate proof obligations. An initially conservative route compares complete ordered canonical decoded pixels and ranged-start inputs across providers, rejects mismatches and binds actual output provenance; nominal FPS/count alone is insufficient. This route is proposed, **not verified**. Use no automatic colour transform, tone mapping or precision downgrade.

## Availability and version choice

PyAV is absent from the compatible environment. Bounded named-file cache searches found no candidate wheel; opaque HTTP-cache contents were not exhaustively inspected. FFmpeg/ffprobe/GStreamer commands are absent from PATH. Installed PySide6-Essentials has QtMultimedia stubs, not its extension; documented Qt frame microseconds are not original integer PTS/timebase. [Qt frame API](https://doc.qt.io/qtforpython-6.10/PySide6/QtMultimedia/QVideoFrame.html#PySide6.QtMultimedia.QVideoFrame.startTime).

PyAV19.0.2 is suitable only for an isolated current-target experiment: Python>=3.12 matches CI/packaging's 3.12 target and the compatible interpreter. It must not silently impose that minimum on all headless users. PyAV18.1 supports>=3.11;16.1 supports>=3.10, but older versions require independent review of fixes rather than automatic selection.19 uses `AVRational`, not `fractions.Fraction`; convert numerator/denominator without float loss and reject missing/falsy timebases. [19 registry/files](https://pypi.org/project/av/19.0.2/), [18 registry](https://pypi.org/project/av/18.1.0/), [16 registry](https://pypi.org/project/av/16.1.0/), [19 changelog](https://raw.githubusercontent.com/PyAV-Org/PyAV/v19.0.2/CHANGELOG.rst).

## Acceptance before admission

- Full presentation-order decode through EOF/flush, exact rational strictly increasing uniform PTS and retained original origin. Missing, duplicate, variable cadence, corruption or incomplete evidence refuses admission. No guessed FPS/DTS replacement.
- Seeking is keyframe/approximate: decode forward to the exact certified PTS, refuse missing/overshoot. B-frame and nonzero/negative-origin fixtures are required. A bounded reopen-from-start fallback is explicit, not a hidden unlimited workaround.
- Prove colour/range/chroma/orientation/precision and exact source ordinal correspondence; `rgb24` alone does not prove OpenCV parity. Unknown HDR/high-bit cases stay refused.
- No shared concurrent codec context. I/O timeout does not prove hard codec-compute interruption. Investigate the pinned `_decode` handling of `InvalidDataError` after partial collected frames: no exception alone cannot certify an error-free complete stream.
- Preserve loader cancellation/stale/source binding, selected source offsets, EXR mapping, A/B and queue bounds. Fixtures/mock records prove software contracts, not actual codec certification.

[Decode/seek contract](https://pyav.basswood.io/docs/stable/api/container.html), [video-rate caveats](https://pyav.basswood.io/docs/stable/api/video.html), [reformatter](https://raw.githubusercontent.com/PyAV-Org/PyAV/v19.0.2/av/video/reformatter.py), [timeout contract](https://pyav.basswood.io/docs/stable/api/_globals.html).

## Licensing and Windows

PyAV source is BSD-3-Clause, but this is not a complete description of bundled FFmpeg/x264/x265 libraries. Upstream distinguishes shipped wheels from test GPL builds and discusses alternative commercial exceptions. Exact downstream binary redistribution terms remain unresolved: neither BSD-only nor definitively GPL-only is established. Review before bundling/manifests/notices. Windows wheel availability does not establish PyInstaller collection, codecs, Unicode paths, cancellation or clean-machine acceptance. [Source license](https://raw.githubusercontent.com/PyAV-Org/PyAV/v19.0.2/LICENSE.txt), [vendor selection](https://raw.githubusercontent.com/PyAV-Org/PyAV/v19.0.2/scripts/ffmpeg-latest.json), [vendor build](https://raw.githubusercontent.com/PyAV-Org/pyav-ffmpeg/9.0.2-1/scripts/build-ffmpeg.py), [upstream discussion](https://github.com/PyAV-Org/PyAV/issues/2270).

## Next action

PERM-004 in [permission ledger](permissions-required.md) proposes only a separate temporary package target and local provider experiment. Matching ARM64 wheel is approximately18.6MB (Windows x64 approximately28.2MB), not a model/media download. Network restriction prevents obtaining the absent package without approval; correctness and redistribution are separate unresolved technical/release gates. No production adoption is approved by installing an experiment.


## Current experiment preflight — 2026-10-24

PERM004 acquisition/import grant was used: PyAV19.0.2 installed only to `/private/tmp/corridorkey-pyav-evaluation-19.0.2`. Native av/OpenCV/NumPy loaded together;7 existing reader/range tests pass0.543s. This changes the earlier package-availability blocker, not the unmet rawPTS/error/seek/colour/precision/correspondence proof. No runtime dependency, active/global environment or packaging adoption. See [verification record](verification-summary.json); proceed with bounded provider experiments after publication.
