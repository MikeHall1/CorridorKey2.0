# Inference and performance audit — lane B

## Code Review Summary

**Target:** dirty worktree based on `7d808e61d8a42d77e30792ded6911aabca96a08f`, inspected 2026-10-24. This is a whole-pipeline audit, not a claim that the findings were introduced by the pending diff.

**Files Reviewed:** 35 source/test files, with depth stated below; supporting requirements and engineering documents also inspected.
**Total Issues:** 5 ranked findings.

### By Severity

- CRITICAL / P0: 0
- HIGH / P1: 3
- MEDIUM / P2: 2
- LOW / P3: 0

**Recommendation: REQUEST CHANGES.** Tighten the model/foundation loading contract before trusting a successful inference result. Keep actual GPU throughput, model quality, and production colour acceptance explicitly unverified.

What this subsystem does: it decodes local footage, extracts frozen foundation features, blends temporal Stage 1 features, predicts full-resolution alpha/foreground in Stage 2, and exports through shared writers. Explicit draft, guidance, correction and recovery paths reuse that engine. Assumed load is a single persistent worker processing production-resolution clips, with sequential jobs and optional multiple independent sessions—not a public multi-tenant service.

## Findings

### INF-001 — P1 / HIGH — MoGe can report successful loading with an uninitialized encoder

- **Evidence:** `third_party/MoGe/moge/model/v2.py:99-105` constructs the model and calls `load_state_dict(..., strict=False)` without checking missing/unexpected keys. `corridorkey/foundations.py:29-38` accepts that result, switches it to eval, and freezes it. `corridorkey/foundations.py:270-272` later consumes its encoder for every frame.
- **Confirmed reproduction:** execute the real inherited `MoGeModel.from_pretrained` method on a tiny subclass containing one `encoder.weight`, with a local checkpoint `{'model_config': {}, 'model': {}}`. It returns successfully with that randomly initialized parameter still present. Output: `MOGE empty state accepted: ['encoder.weight'] -0.02859961986541748` (the value itself is random). No pretrained model or GPU is needed to demonstrate the missing-key acceptance.
- **Impact / root cause:** an incomplete or wrong-version asset with valid serialization can produce finite but meaningless features and an apparently completed render. File presence, a checksum of user-selected bytes, and successful native-SDPA setup do not establish tensor compatibility. The outer CorridorKey trainable-state check intentionally excludes frozen foundations, so it cannot repair this omission.
- **Minimal fix:** require all inference-used MoGe encoder tensors to load. Fail with missing/unexpected tensor names. If unused prediction-head differences genuinely need compatibility, permit only a documented exact set; do not accept arbitrary missing encoder weights. Keep upstream attribution and record the integration patch.
- **Effort:** small, approximately half a day including focused tests; validate against the actual approved checkpoint when available.
- **Verification:** missing one encoder tensor, empty state, unexpected encoder tensor and wrong shape must fail; the approved complete checkpoint must load. Existing setup-error tests do not cover this path.
- **Confidence:** high; loader behavior reproduced. Actual visual impact magnitude remains unmeasured.

### INF-002 — P1 / HIGH — Required foundation statistics silently degrade to default or invalid normalization

- **Evidence:** `corridorkey/foundations.py:170-171` skips a supplied non-file stats path. `corridorkey/foundations.py:193-215` ignores absent keys, prints and skips wrong-shaped tensors, accepts nonpositive/nonfinite standard deviations, then prints that stats loaded. `corridorkey/foundations.py:261-278` clamps invalid nonpositive standard deviations to `1e-6` during normalization. `corridorkey/runtime_paths.py:54-61` checks existence, not file type or stats schema.
- **Confirmed reproduction:** a real safetensors file with wrong-shaped `moge_mean=[100,200]`, `moge_std=[-1]`, and missing RVM keys is accepted by a small `FrozenFoundations` instance. Output retains `moge_mean=[0]`, accepts `moge_std=[-1]`, retains `rvm_std=[1]`, and says `loaded stats`. The same branch applies to enabled-model buffers with their actual channel widths.
- **Impact / root cause:** a malformed required asset silently changes the learned feature distribution. Missing means/stds become identity normalization; a negative std becomes division by `1e-6`. Finite wrong features need not trip the final nonfinite guard, so the job can finish without the requested model semantics. This fails the root-cause guard: the default buffers and skip branch substitute for fixing/admitting the asset contract.
- **Minimal fix:** runtime loading must require a regular readable stats file and exact required keys/shapes for enabled foundations, finite means/stds, and strictly positive stds. Validate the entire document before copying any buffer. If optional identity stats remain useful for training or fixtures, separate that explicit mode from runtime loading.
- **Effort:** small, approximately half a day with tests.
- **Verification:** missing file, directory, missing key, wrong width, NaN/Inf and zero/negative std must fail before any partial stats mutation; valid stats must retain the existing normalization result exactly.
- **Confidence:** high; malformed-asset acceptance reproduced.

### INF-003 — P2 / MEDIUM — C-RADIO implementation is pinned to the first asset root in a process

- **Evidence:** `corridorkey/foundations.py:56-75` uses one constant `_corridorkey_cradio_snapshot` package/module name in `sys.modules`; subsequent calls do not compare its origin with the new `repo_path`. Lines 77-82 load the new directory's configuration/weights through the old class. Independent `InferenceSession` instances can select distinct asset roots (`corridorkey/session.py:105-129`).
- **Confirmed reproduction:** create two temporary local `hf_model.py` fixtures, A and B, each returning its implementation marker and requested weight path. Loading A then B prints `('A', '.../A')` followed by `('A', '.../B')`.
- **Impact / root cause:** a second session using a different C-RADIO snapshot can mix the first snapshot's Python implementation with the second snapshot's weights/config. It may fail in obscure ways or compute different features than the requested asset identity. Releasing a model does not remove these imported modules. Desktop workflows that start a fresh worker avoid this particular sequence, but the supported shared Python session API does not guarantee that isolation.
- **Minimal fix:** key the private module namespace by a stable resolved snapshot identity, including imported package children, or explicitly refuse a second differing snapshot and require process restart. Clean up only this loader's partially published modules if import fails. Do not silently reuse the first root.
- **Effort:** small-to-medium, about one day including relative-import and failure tests.
- **Verification:** same-root reuse, A→B, B→A, relative imports and a failed first import followed by a valid load. Confirm classes originate from the requested root or produce the explicit isolation error.
- **Confidence:** high; origin mixing reproduced with real import machinery.

### INF-004 — P1 / HIGH — C-RADIO retries any internal TypeError as an API compatibility fallback

- **Evidence:** `corridorkey/foundations.py:243-246` catches every `TypeError` from the entire foundation forward call and reruns without `feature_fmt='NCHW'`.
- **Confirmed reproduction:** a tiny C-RADIO substitute raises `TypeError('internal math failure, NOT unsupported keyword')` when called with the keyword and returns a valid feature tensor without it. `FrozenFoundations.forward` succeeds with calls `[{'feature_fmt': 'NCHW'}, {}]`; the original failure is lost.
- **Impact / root cause:** actual internal type errors can be disguised as successful compatibility execution; expensive inference may execute twice or leave partial mutable state before the retry. There is no documented/version-checked external boundary or regression proving that only the unsupported keyword is handled. This is a root-cause-guard blocker, consistent with the current pending change that correctly stops swallowing MoGe setup errors.
- **Minimal fix:** use the pinned C-RADIO API directly and propagate internal errors. If an older supported snapshot truly lacks the argument, choose its documented adapter before invoking inference, using an explicit supported version/signature check; test both contracts and preserve errors from either implementation. Do not broadly retry a failed forward.
- **Effort:** small, a few hours plus approved-snapshot verification.
- **Verification:** an internal TypeError must propagate after exactly one call; explicitly supported old/new signatures must produce the required feature layout without catching arbitrary inference failures.
- **Confidence:** high; diagnostic suppression reproduced. Frequency on the shipped pretrained model is not known.

### INF-005 — P2 / MEDIUM — Spatial refinement plus trimaps reopens and redecodes video from the start on every frame

- **Evidence:** with spatial draft and alpha refinement active, `corridorkey/runtime.py:1491-1495` reads native guidance. For a trimap on the same frame, `corridorkey/runtime.py:1498-1500` immediately reads it again. `corridorkey/spatial_draft.py:55-69` closes/reopens its forward-only reader when a requested frame precedes `next_frame`; the first read has already incremented that cursor. The actual video reader decodes from its selected-range origin through the requested frame (`corridorkey/video_io.py:505-515,534-546`).
- **Confirmed reproduction:** the existing shape-aware CPU runtime fixture, five real PNG inputs, `model_scale=.5`, enabled alpha refinement, and a five-frame trimap produces `frames_requested=10`, `source_frames_decoded=10`, `reader_restarts=5`. A sequential decoder fixture following the actual video reader's cursor semantics for 240 frames produces 480 requests, **29,160 decoded frame instances**, and 240 restarts, rather than 240 sequential decoded frames. The fixture counts are not a measured video/GPU speedup.
- **Impact / root cause:** two refinements independently request identical native RGB without reusing the first result. Image sequences pay duplicate reads; video pays quadratic decoding work, defeating a draft workflow on longer clips and delaying cancellation inside repeated native decode loops.
- **Minimal fix:** retain the current frame's native guidance within the existing per-frame loop and reuse it for trimap processing. This needs no new general cache or option and must preserve the existing refinement order.
- **Effort:** small, a few hours including focused regression.
- **Verification:** spatial refinement alone, trimap alone and their combination retain identical outputs; combined mode requests native guidance once per frame. A forward-decoder fixture must show linear decode counts and no same-frame restart. Validate one real local video when available.
- **Confidence:** high for duplicate reads/restarts; quadratic video work follows the directly inspected reader loop and was reproduced with its cursor semantics. No real-video wall-time claim.

## Feature claims versus implementation

| Area | Observed implementation | Acceptance boundary |
|---|---|---|
| Shared inference | CLI/session worker converge on one `run_inference`; persistent model reuse has single-owner locking, and touched models are discarded after failures. | No alternative GUI neural engine observed. Model-load correctness has INF-001–004. |
| Devices/precision | CUDA matting only; bfloat16 neural/foundation inputs and autocast; float16 RVM/baton maps, float32 IID caches and CPU blending. | No Apple-GPU matting or end-to-end 16-bit accuracy claim. Hardware suitability/parity unverified. |
| Preprocessing/colour | Native padding; explicit signed-asinh linear model RGB; clipped SDR foundation proxies; source-primary conversion is explicit for linear sequences. | No automatic ICC/OCIO/log/PQ/HLG management. Legacy video decodes uint8; high-bit video and model training primaries remain unverified. |
| Temporal stages | RVM loops in temporal order with segment resets; Stage 1 processes contiguous windows and blends batons; Stage 2 flattens B×T for per-frame spatial computation. | Model/checkpoint parity, real visual consistency and measured hardware costs are not established by fixtures. |
| Hints/corrections | First/full/carried hints enter the existing single-channel interface; authored and tracked constraints refine output; trimaps are final refinement, not trained extra input channels. | Input hint alignment/quality needs actual footage; no claim that mask tracking is matting. |
| Quick Key | Interval 1 routes full-rate; sparse anchors use native context; offline motion uses future anchors; hold and past-only causal modes exist; current source RGB is retained. Explicit stop/approximate/neural policies and work counters are present. | Whole-clip prepasses remain; causal propagation is not a live zero-startup pipeline. Reliability is heuristic. No N-fold speed guarantee or equivalence to reconstructed foreground. |
| Tracking/clean plates/click guidance | Tracking uses shared flow with identity/range/boundary checks; clean-plate registration/exposure/classification are heuristic; optional local SAM2 provider is isolated from matting. | Real tracked-reference quality and actual SAM2 execution were not run; coarse Keep guidance may make interiors opaque and is not a hair/blur alpha prediction. |
| Full resolution/drafts | Native outputs remain native; explicit spatial drafts resize linear input before proxy/encoding and upsample outputs; native constraints/refinement remain separate. | Spatial draft is an approximation, not native neural detail. INF-005 affects a supported combination. |
| Cache/memory | Job-local byte-bounded IID cache, typed durable adapters at runtime hooks, full-clip disk maps, and an explicit temporary-map admission plan. | Retained-payload budgets are not whole-RAM/VRAM hard caps. This lane inspected IID cache implementation and replay integration, not every durable store's complete adversarial filesystem behavior. |
| Profiling/performance | Runtime records calls, frames, inclusive stage scopes and rendered-first-frame latency; requested/effective settings are separate. | No pretrained FPS, latency, speedup or professional-quality result was manufactured. CUDA and model assets are unavailable locally. |

## Validation and root-cause review

- Environment actually checked: `/private/tmp/corridorkey-compatible-venv/bin/python`, Torch **2.9.1**, NumPy **1.26.4**, OpenCV **4.11.0**, `torch.cuda.is_available() == False`. The user's `.venv` was not changed.
- `python -m unittest discover -s tests -p 'test_foundation_setup.py' -v`: **2 passed**. These confirm the pending MoGe setup change propagates failure; they do not validate weight completeness.
- `python -m unittest discover -s tests -p 'test_sparse_readers.py' -v`: **4 passed**, including signed-HDR draft and sparse reader behavior.
- Inline local proofs below ran successfully. They use temporary local files and tiny substitutes; no model downloads, network calls, pretrained claims or repository runtime edits.
- Modified-file diagnostics: `ruff check --output-format concise corridorkey/foundations.py corridorkey/video_io.py` reported **10 findings** (5 per file). `pyright --pythonversion 3.12 --pythonpath /private/tmp/corridorkey-compatible-venv/bin/python corridorkey/foundations.py corridorkey/video_io.py --outputjson` reported **23 errors, 0 warnings** (10 and 13 respectively). This is **not static-clean**; this targeted run does not classify all pre-existing diagnostics or replace the coordinator's full no-new-diagnostic comparison. Raw Pyright result was written locally to `/private/tmp/corridorkey-inference-audit-pyright.json`.
- The pending `foundations.py` change removes the MoGe SDPA swallow; the pending `video_io.py` change adds a native geometry check. Both address their primary contract rather than introducing fallback paths. Existing stats and C-RADIO fallback behavior still fail the root-cause guard as documented above.
- Security-sensitive unrestricted legacy `torch.load` and executing local C-RADIO Python code were identified and handed to the coordinator's security lane, avoiding duplicate ranked findings here.

## Reproduction details

Use the compatible interpreter above from the repository root. This minimal script exercises INF-001–004 without model weights:

```python
from pathlib import Path
import tempfile, sys
import torch
from safetensors.torch import save_file
from corridorkey.foundations import FrozenFoundations, _load_local_cradio_so400m

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    f = FrozenFoundations(device='cpu', use_moge=False, use_cradio=False)
    save_file({'moge_mean': torch.tensor([100., 200.]),
               'moge_std': torch.tensor([-1.])}, str(root / 'stats.safetensors'))
    f.load_stats(str(root / 'stats.safetensors'))
    print(f.moge_mean.tolist(), f.moge_std.tolist(), f.rvm_std.tolist())
    template = '''MARKER={marker!r}
class RADIOConfig:
 @classmethod
 def from_pretrained(cls,*a,**k): return None
class RADIOModel:
 @classmethod
 def from_pretrained(cls,path,**k): return (MARKER,path)
'''
    for name in ('A', 'B'):
        folder = root / name
        folder.mkdir()
        (folder / 'hf_model.py').write_text(template.format(marker=name))
    print(_load_local_cradio_so400m(root / 'A'))
    print(_load_local_cradio_so400m(root / 'B'))

sys.path.insert(0, str(Path('third_party/MoGe').resolve()))
from moge.model.v2 import MoGeModel
class Tiny(MoGeModel):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.encoder = torch.nn.Linear(1, 1, bias=False)
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / 'model.pt'
    torch.save({'model_config': {}, 'model': {}}, p)
    print(list(Tiny.from_pretrained(p).state_dict()))

class Radio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
    def forward(self, x, **kwargs):
        self.calls.append(kwargs)
        if kwargs:
            raise TypeError('internal math failure, NOT unsupported keyword')
        from types import SimpleNamespace
        return SimpleNamespace(features=torch.ones(1, 1, 1, 1))
f = FrozenFoundations(device='cpu', use_moge=False, use_cradio=False)
f.cradio = Radio()
f.register_buffer('cradio_mean', torch.zeros(1))
f.register_buffer('cradio_std', torch.ones(1))
print(f(rgb_cradio=torch.ones(1, 3, 16, 16))['cradio'].item(), f.cradio.calls)
```

INF-005 real source/EXR orchestration proof, with existing CPU neural substitutes:

```python
import sys
sys.path.insert(0, 'tests')
import numpy as np
from test_spatial_draft import SpatialDraftTests
from test_trimap_refinement import TrimapRefinementTests
from corridorkey.corrections import save_corrections
from corridorkey.identities import source_identity
from corridorkey.spatial_draft import NativeGuidance
from types import SimpleNamespace

case = SpatialDraftTests()
case.setUp()
try:
    source = case.source()
    t = TrimapRefinementTests()
    t.root = case.root
    p = t.asset(np.full((65, 97), 128, np.uint8))
    doc = t.document(p, shape=(65, 97), frames=5,
                     source_hash=source_identity(source)['sha256'])
    correction = case.root / 'trimap.json'
    save_corrections(correction, doc)
    _, diag, _ = case.run_job(source, refine=True, correction=correction,
                              outputs={'alpha'})
    print(diag.execution['native_trimap_guidance'])
finally:
    case.doCleanups()

class SequentialVideoFixture:
    def __init__(self):
        self.next_frame = self.frames_decoded = 0
    def close(self):
        pass
    def read(self, frame):
        self.frames_decoded += frame - self.next_frame + 1
        self.next_frame = frame + 1
        return SimpleNamespace(rgb=frame)
g = NativeGuidance(SequentialVideoFixture, check=lambda: None)
for frame in range(240):
    g.read(frame)
    g.read(frame)
print(g.stats)
g.close()
```

## Inspected files and boundaries

Production modules (30): `runtime.py`, `foundations.py`, `model.py`, `stage1.py`, `stage2.py`, `blocks.py`, `rope.py`, `rvm.py`, `video_io.py`, `config.py`, `session.py`, `runtime_paths.py`, `scheduling.py`, `quick_key.py`, `motion.py`, `draft.py`, `foundation_cache.py`, `cache.py`, `refinement.py`, `trimap_refinement.py`, `spatial_draft.py`, `color_utils.py`, `colour_contract.py`, `correction_tracking.py`, `correction_tracking_context.py`, `clean_plate.py`, `subject_guidance.py`, `resources.py`, `temp_budget.py`, `durable_foundations.py`, all under `corridorkey/`.

Vendored loader (1): `third_party/MoGe/moge/model/v2.py`. The `vfms/{cradio,moge,rvm}` directories contain `.gitkeep` rather than the configured pretrained assets; C-RADIO's actual exported implementation therefore could not be inspected. MoGe contains 39 Python sources; this review examined the runtime loader boundary and traced the SDPA entry point, not all vendored encoder mathematics.

Fixtures (4): `tests/test_foundation_setup.py`, `tests/test_quick_runtime.py`, `tests/test_spatial_draft.py`, `tests/test_trimap_refinement.py`. The separate sparse-reader test module was executed, not counted as a full source inspection.

Depth: complete runtime orchestration, foundations, main model, configs/session, scheduling, Quick Key/motion, core cache, colour helpers and spatial guidance were read. Large transformer modules were inspected at their forward/attention/chunking boundaries rather than every constructor or training-only branch. Correction/segmentation/clean-plate paths were reviewed at kernel, provenance and runtime integration boundaries; desktop interaction and persistence authoring remain other lanes. Reader/preprocessing/hint sections were inspected; writers, recovery publication, worker transport, benchmark orchestration and native OFX received only call-boundary context here.

Supporting context: `AGENTS.md`, `docs/development-brief.md`, `docs/product-improvements-brief.md`, `docs/architecture.md`, `docs/colour.md`, `docs/options.md`, `docs/status.md`, `docs/code-health.md`, current issue/audit context, and `requirements.txt`.

**Not checked:** actual pretrained inference, CUDA kernels or peak-memory behavior, photographic/held-out quality, SAM2 execution, Windows packaging and Resolve viewer/export. No claims of real-time performance or professional acceptance follow from this audit.
**Remaining risk:** passing CPU fixtures/static regression gates do not prevent successful-looking renders with incomplete foundation assets; repair INF-001/002/004 before treating successful loading as compatibility evidence.
