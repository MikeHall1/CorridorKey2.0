# Explicit OFX upstream capture and companion preparation

EXT-10 capture and companion mechanics are locally reviewed and CPU-tested.
This workflow uses supported OFX actions and the shared Python engine; actual
GPU, native Windows and Resolve viewer/export acceptance remains NOT VERIFIED.
It does not discover the editor's selected media or original file.

In the experimental cached effect, choose Capture parent, integer Capture In and
Out exclusive, and an explicit byte budget. Declare that upstream pixels are
opaque scene-linear BT.709/D65, then use Capture upstream. The effect samples its
own Source clip at those integer host times. Its existing CacheFile/CacheStart
parameters remain unchanged. Last capture manifest identifies a committed bundle.
A failed handoff parameter setter reports the committed path without rolling back
or changing the previous cache.

Capture runs during user-edited InstanceChanged, using official image/property/
parameter suites and optional progress V2 or V1. Progress is required for capture;
missing progress does not disable existing prepared-cache rendering. Every image
lease ends within the action. Capture requests the full validated RoD, handles
padded/negative row strides, copies top-down little-endian RGB float32, and polls
host progress/abort. No neural model, subprocess or host calls from a companion
thread are introduced. Recursive RoD/ROI requests do not hold the render mutex.

The initial contract requires connected Source, temporal access, full render
scale, zero-origin native square-pixel unfielded float RGBA, finite RGB and alpha
exactly1. Range must lie in the clip's declared frame bounds. Maximum120frames,
16,777,216pixels per frame, extent32768 per axis and512MiB total RGB payload;
CaptureMaxMiB defaults512 and may be lowered to1. Limits combine: large native
frames admit fewer frames. These are admission bounds, not a hard process RSS
limit or real-time claim. No resizing, alpha conversion, colour inference or
precision downgrade occurs. Unsupported conditions fail explicitly.

Capture creates an exclusive fresh child under the chosen real parent. Its fixed
capture.json has schema1 and13 fields: type, version, random capture identity,
effect-upstream scope, signed host start, count, width, height, FPS, pixel aspect,
explicit colour contract, fixed frames.rgb32 payload size/hash and per-frame
source digests. Digests use the existing SourceWitness domain, relative frame
index and native dimensions. The empty COMPLETE marker commits last, after file
flush/close and final progress/cancellation checks. Failed writes remove only
owned files; foreign conflicts are retained and identified. Sampled filesystem
pins and hashes fence ordinary changes, not a hostile owner of the filesystem.

Desktop Projects → OFX upstream capture… captures the current advanced settings
without opening media or starting inference. Hints and stage profiling currently
refuse explicitly. Inspect metadata does bounded local metadata work. Verify
captured pixels uses the selected read budget (default/max512MiB) to hash all
captured bytes and compare every relative source digest. It also rejects nonfinite
pixels and duplicate JSON fields.

Review the exact derived settings before Create reviewed EXR job. The capture
maps to local source start0/count/FPS, linear RGB with ignored already-opaque alpha,
legacy input profile (explicit linear709 assumption retained in provenance),
source_witness enabled and EXR output with required foreground/alpha. Existing
non-source processing choices, extra outputs and requested output EXR precision
remain. Corrections, declared source profiles, nonbegin starts, explicit cuts/
resets and video-only linear adjustments initially refuse. Input EXRs are always
FLOAT ZIP, and the normal decoder must reproduce exact float32 bits, including
signed zero, subnormals and signed HDR. There is no tolerance fallback.

Choose a conversion parent and a separate fresh render output. Materialization
creates an owned input snapshot and persistent capture/job binding; no render is
submitted. Review its full job before Submit reviewed render. The ordinary shared
worker, queue and selected assets own this render. Cancellation targets that job;
no second engine owner is started. Editing intake choices fences old I/O results.
Closing waits for the owned file operation and render acknowledgement. A display
failure after conversion retains the committed workspace and disables submission
rather than hiding the artifact.

After a completed render, Validate render and prepare cache verifies exact native
job paths, requested/effective settings, geometry/FPS/local range and consumed
source witness against the capture. It rejects protected cache destinations before
writes. Existing sealing produces CKOFX002; receipt validation reads actual cache
header, tables, source digests and all output frame hashes. A fresh attachment
receipt records CacheFile and CacheStart equal to the original signed host In.
This is a deliberate manual attachment; no host setters run from desktop code.
The previous host cache remains usable until the user explicitly changes it.

Working directories and completed records remain local. The initial dialog has no
reopen/restore of its transient prepared object; preserve the capture, materialized
binding, render records and receipt. Companion input workspaces and attachment
receipts are assembled under a fresh owned container's `pending/` child. After
file closes, validation and cancellation checks, renaming that child to `ready/`
publishes the result; only `ready/` is committed. If cleanup fails, a retained
`pending/` tree is incomplete staging, not a renderable job or validated attachment.
The native host capture bundle retains its separate COMPLETE-last contract above.
Ordinary jobs remain usable by existing queue and project tools. There is no background editor selection, automatic retime/proxy
or colour-management interpretation, or atomic two-parameter attachment guarantee.
Existing bound-cache rendering still validates every selected upstream source frame
and refuses fractional interpolation/scale/field/geometry/FPS/pixel mismatches.

The actual Resolve version/edition and its installed Developer docs, native
Windows x64 build, known host colour pipeline, CUDA/weights/local footage and
viewer/export/random/reverse/multi-instance/retime/proxy/persistence scenarios are
required for external acceptance. Fixture suites and CPU toy operators establish
local mechanics, not Resolve compatibility, neural quality or GPU performance.
