# Feature completeness

Classification applies to this checkout, not promises in the README. **Locally verified** below means the indicated CPU/fixture contract, never pretrained CUDA quality. Full-suite evidence and limitations are in [verification-evidence.md](verification-evidence.md). Historical `docs/acceptance.md` and `.ralph/plan.json` remain unchanged.

| Feature | Classification | Source / tests and evidence | Remaining acceptance |
|---|---|---|---|
| Automatic foreground extraction | Implemented but not verified | `runtime.py`, `model.py`, `foundations.py`; `test_io`, `test_hann_runtime`, `test_session` exercise synthetic models | Real weights/CUDA/representative footage; INF findings |
| First-frame subject hint | Implemented and locally verified (adapter only) | `video_io.py`, runtime hint readers; `test_io`, `test_correction_runtime` | Trained quality; cross-job preservation ARCH-001 |
| Full-sequence hints | Implemented and locally verified (CPU adapters) | sequential hint readers; `test_colour_runtime`, `test_hann_runtime` | Real video timing/encode and pretrained effect |
| Temporal mask propagation / Quick Key | Implemented and locally verified (synthetic sequences) | `quick_key.py`, `motion.py`, `draft.py`, `shot_analysis.py`; `test_motion`, `test_quick_runtime`, `test_past_draft` | Real occlusions/cuts, quality/performance; offline precompute is not live streaming |
| Native-resolution inference | Implemented but not verified with pretrained model | native Stage2/runtime, `test_video_geometry`, actual CPU EXR fixtures | GPU memory/quality at production resolution |
| Draft processing | Implemented and locally verified (contracts) | `spatial_draft.py`, `quick_key.py`; `test_spatial_draft` and Quick Key fixtures | Approximation quality and measured benefit |
| Manual frame corrections / trimaps | Implemented and locally verified | `corrections.py`, editor/service/canvas, `trimap_refinement.py`; correction/editor/refinement tests | Native target UX and real footage quality |
| Tracked corrections | Partially implemented | `correction_tracking*.py`; tracking/editor/overlay tests | Primarily sequences; representative occlusion/video/quality acceptance pending |
| Clean-plate assistance | Implemented and locally verified (synthetic guidance) | `clean_plate.py`, `clean_plate_editor.py`; clean-plate tests | Real alignment/exposure/quality benefit; not new trained neural channels |
| Click-to-select optional provider | Partially implemented | `subject_guidance*.py`; provider child/click fixtures | Actual optional provider/model not installed or executed |
| Recovery/checkpoint support | Partially implemented | `recovery*`, `durable_*`, `queue_*`; crash/corruption/replay/EXR prefix tests | No general direct temporal seek; physical power-loss and Windows/GPU acceptance absent |
| Batch processing | Broken for cross-job asset-collision case | `batch.py`, `session.py`; ARCH-001 actual CPU EXR proof | Batch-wide protected-input admission before any write |
| Output configuration / alpha export | Implemented and locally verified (EXR, config) | `video_io.py`, `config.py`, `test_io`, `test_colour_contract` | Actual FFmpeg codecs absent; colour-managed target interoperability |
| Desktop queues/settings/projects | Implemented and locally verified (offscreen) | `desktop_*`, worker bridge and native Qt tests; source child smoke | Interactive target UX and packaging; UI findings |
| Synchronized source/export A/B playback | Partially implemented | `playback.py`, `desktop_playback.py`, export indexes; playback/comparison tests | Video/VFR, target display and long-sequence latency |
| Progressive refinement | Broken/incomplete integration | `progressive_preview.py`, `desktop_progressive.py`; scheduler tests plus UI-001/UI-003/UI-004 | Playhead wiring, Escape shutdown, partial-range coordinates; real GPU response |
| Diagnostic overlays | Partially implemented | `tracking_overlays.py`, `diagnostic_ranges.py`; flow-map tests | Saved tracks, boundary/matte diagnostics and calibrated false-positive evaluation |
| Calibration | Implemented but not verified with real inference | `calibration.py`, `calibration_task.py`, desktop panel; trial/report fixtures | Actual hardware/sample cold/warm measurements; no automatic speed claim |
| Project portability | Partially implemented | `portable_project.py`, `portable_paths.py`, `project_collection.py`; portable/foreign project tests | Foreign partial/reference-only relink limitations; Windows filesystem proof |
| Performance optimization | Partially implemented and locally measured only for decoder fixtures | caches, persistent sessions, batching, telemetry; decoder 9/9 exact parity | Full trained end-to-end alternating benchmark; INF/ARCH overhead findings |
| Evaluation infrastructure | Implemented and locally verified | `evaluation*.py`, benchmark pair/quality/dossier; evaluation tests | Licensed reference footage, blinded quality review, real produced outputs |
| Windows release | Implemented recipe but not verified | `desktop_bundle.py`, `app_process.py`, launcher `.bat`; preflight refusal | Windows build, clean installation, CUDA/FFmpeg, signed/reviewed package |
| DaVinci Resolve interoperability | Partially implemented | `native/ofx`, `ofx_cache.py`, `ofx_capture.py`; native harness and capture fixtures | Actual Resolve viewer/export, project edits, retimes/proxies/colour; no live neural OFX claim |
| Additional GPU backends / multi-GPU | Planned only | development brief optional capabilities | Not a reason to add speculative framework now |
| Browser frontend | Not applicable | Qt Widgets application, no browser frontend | Playwright/Chrome not an appropriate substitute for Qt tests |

The ten product-extension rows are materially connected, but none of their broad product-quality promises should be accepted solely from documentation or CPU mocks. The release recommendation is **suitable for controlled testing**, not beta or production readiness.
