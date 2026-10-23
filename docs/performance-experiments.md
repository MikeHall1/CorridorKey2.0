# Performance experiments

All GPU performance hypotheses below are **unmeasured**. Source inspection establishes where work happens, not its cost or output equivalence. Profile before reprioritizing. First milestones preserve existing math and backend.

| # | Opportunity / evidence in checkout | Next experiment and risk |
|---|---|---|
| 1 | Optional job-local byte-bounded float32 RAM reuse implemented; low-VRAM already precomputes | CPU fixture: 24 -> 15 foundation frame evaluations for a 15-frame/8-chunk/4-stride case, identical blended batons with deterministic substitutes. Real GPU output tolerance and net transfer/throughput effects unmeasured; default remains off |
| 2 | Baton blending transfers weighted features to CPU | GPU accumulation under explicit budget; compare numeric reduction and peak VRAM |
| 3 | Launcher now retains a session; headless batch also reuses one model | Lifecycle/reset behavior CPU-tested; cold/warm runner implemented. Actual model-load savings, state parity and memory growth unmeasured |
| 4 | RVM, VFM and stage readers decode separately | Bounded decoded-frame sharing; measure RAM, I/O, cache state and colour equivalence |
| 5 | OpenCV/NumPy colour/pad/resize preprocessing | GPU path only after exact reference transfer/Lanczos validation |
| 6 | non_blocking transfers use ordinary CPU tensors | Pinned bounded buffers + streams/events; verify actual overlap, do not assume |
| 7 | Per-window allocation/stacking | Shape-stable buffers; reject stale contents and track peak memory |
| 8 | Prepasses/Stage 1 no_grad, Stage 2 inference_mode | Selected inference_mode trial; check inference-tensor lifetime constraints |
| 9 | Automatic policy prefers context size | Sweep windows with quality, latency and throughput measured separately |
| 10 | Stage 2 batch default 1 | Per-resolution/device sweep, no assumed monotonic speed |
| 11 | MoGe/C-RADIO caches float32, RVM/baton float16 | Optional foundation float16 only with reference precision tolerance |
| 12 | RAM feature payload budget and automatic VRAM selection cap implemented | CPU-tested admission/eviction and policy calculations; total process RAM/VRAM, disk budgets and host coexistence remain unvalidated/unimplemented |
| 13 | Separate RVM/VFM prepasses | Shared chronological prepass; preserve recurrent state, cold/warm timing |
| 14 | Eager model | Compile selected stable stages; record build time/failures and fallback |
| 15 | No export backend | Export feasibility per stage; operator/shape coverage, build and precision validation |
| 16 | OpenCV video decode | Hardware decode with transfer/primaries/range/bit-depth reference checks |
| 17 | Software x264; ProRes alpha | Optional hardware preview encoder only after capability/colour tests |
| 18 | Alpha-only render skips RGB; checker now built only when requested | CPU adapter test verifies no checker allocation for alpha output. GPU speed benefit unmeasured; test remaining combinations on CUDA |
| 19 | Scaling fails at render | Now rejected early; implement explicit Lanczos draft/output contract before exposure |
| 20 | Temp caches discarded | Verified durable identities/checksums/completion manifests; interrupted/corrupt/partial reuse tests |

Acceptance: same hardware/assets, >=3 alternating baseline/candidate measurements, fixed settings and separately auto settings, output comparison with tolerance selected beforehand, independent and combined changes. Stop an experiment on correctness regression, unsupported backend, resource overrun or no reproducible net benefit. No speedup multipliers are invented or combined.

Research backlog (brief §22) remains experimental: distilled preview/sparse-trained/learned-propagation models depend on licensed training data; boundary-focused compute/ROI/global-context and multiscale edge reconstruction depend on quality references; adaptive anchors and interactive/final models depend on temporal contracts; scene suggestions/reusable hints/automatic corrections need user-reviewed conditioning behavior; quantization needs precision tests; additional/Apple GPU backends need operator support; multi-GPU needs isolated scheduling; other OFX hosts need SDK validation; history dashboards need trustworthy reports; fine-tuning needs suitable data. For each future trial, register hypothesis, expected benefit, dependencies, held-out validation and explicit stop criteria before implementation. None is a guaranteed deliverable.
