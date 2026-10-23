# Sequential reader review-stack allocation

The shared inference runtime now constructs its concrete video/frame-directory
window readers with `include_review=False`. Model and foundation tensors, decode
and preprocessing work, cache lifetime and schedules are unchanged. The third
window result is `None`, instead of an unused independently stacked review
tensor. Normal Stage 1, low-VRAM Stage 2 and Quick Key neural fallback share the
same factory. `hann_stage1_stream` still calls injected factories with exactly
`reader_cls(source, meta)`; no new constructor requirement is imposed on those
public helper implementations.

Direct `SequentialVideoWindowReader` and `SequentialFrameDirWindowReader`
constructors default to `include_review=True`: their original three independent
stacked tensors remain available, including mutation independence when cached
foundation and review frames alias. Existing benchmark/review consumers retain
that default. There is no new user setting, approximation or dependency.

## Measured scope

`benchmark-results/product-next/reader-experiment.txt` is a reproducible local
CPU experiment with fixed generated PNG/MJPG footage, warmups and five matched
alternating pairs. `reader-experiment-results.json` retains raw times, exact
model/foundation hashes, source and implementation hashes and allocation
counters. All correctness/allocation assertions passed. Both workloads traversed
24 frames and performed 24 preprocessing evaluations; stack calls fell from
15 to 10. Maximum returned stacked tensor payload per eight-frame window fell
from 7,077,888 to 4,718,592 bytes (one third).

This is returned tensor payload, **not RSS, total process peak, cache storage or
VRAM**. Per-frame review preprocessing/cached views are deliberately unchanged.
Timing was **inconclusive** under the predeclared rule requiring improvement in
every pair. PNG median baseline/candidate ratio was 1.0688176528017028; MJPG
was 1.0139179162295053. One PNG pair and two MJPG pairs regressed. No
throughput improvement is accepted from this
experiment; no timing rerun or post-hoc threshold relaxation was used.

Focused tests cover real adjusted/spatial PNG/video reads, overlap/sparse
windows, two-vs-three stacks, default mutation independence, decoder close and
actual shared CPU-runtime EXR byte parity across normal, spatial, low-VRAM and
Quick Key paths. These use CPU neural substitutes, not pretrained model quality.
GPU, real-model throughput/quality and Windows/Resolve remain NOT VERIFIED.
