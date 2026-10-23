# Optional click guidance: provider investigation

EXT-04 now has a headless optional-provider foundation, not a connected click UI.
No provider/model was installed, downloaded or executed. Official-source research
and deterministic provider fixtures are not latency/VRAM/quality or platform acceptance.

Recommend one optional **SAM2.1 image provider**, initially with a deliberately
selected local tiny checkpoint/configuration. Its image API directly accepts
positive/negative XY clicks and previous low-resolution mask logits for iterative
refinement. [SAM2 image API](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)

Cutie core consumes an initial object mask; its GUI adds a separate RITM model
for click interaction. Adding both would introduce another model without a
demonstrated need. Reuse the existing motion infrastructure for an initial
tracked-guidance path. [Cutie scripting](https://github.com/hkchengrex/Cutie/blob/main/scripting_demo.py),
[Cutie GUI](https://github.com/hkchengrex/Cutie/blob/main/gui/main_controller.py)

| Consideration | SAM2 | Cutie |
|---|---|---|
| Click interface | Native positive/negative points and iterative mask prompts | Core masks; GUI requires RITM |
| Documented minimum | Python3.10+, Torch2.5.1+, torchvision0.20.1+ | Python3.8+, documented Torch1.12+ |
| Added dependencies | Hydra, iopath and image packages | Broader GUI/web/training dependencies |
| Platform evidence | Linux guidance; Windows recommends WSL | README reports Ubuntu testing |
| Local cost/quality | NOT VERIFIED | NOT VERIFIED |

Sources: [SAM2 setup](https://github.com/facebookresearch/sam2/blob/main/setup.py),
[SAM2 installation](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md),
[Cutie README](https://github.com/hkchengrex/Cutie/blob/main/README.md),
[Cutie package](https://github.com/hkchengrex/Cutie/blob/main/pyproject.toml).
Native Windows integration remains unverified; upstream minimum versions alone
do not establish compatibility with the application's pinned environment.

SAM2 identifies code/checkpoints under Apache2.0. Cutie/RITM code licences are
MIT; inspected Cutie material does not separately establish checkpoint licensing.
Do not redistribute unverified weights. [SAM2 licence statement](https://github.com/facebookresearch/sam2/blob/main/README.md#license),
[Cutie licence](https://github.com/hkchengrex/Cutie/blob/main/LICENSE),
[RITM licence](https://github.com/hkchengrex/Cutie/blob/main/gui/ritm/LICENSE).

Proposed connected implementation (not yet delivered): dependency-free immutable
click/request/capability/result contracts, native-coordinate/budget/provenance
validation, then one lazy SAM provider. Use local `build_sam2` and
`SAM2ImagePredictor`; do not use `from_pretrained`, which downloads weights.
Pin exact source/config/checkpoint hashes and explicitly select device, precision
and post-processing. [Builder](https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py)

Own provider operations in a child, with one active/one queued latest request and
source/frame/prompt generation fencing. Reuse existing clean processing-owner
handoff; never allocate segmentation alongside a resident matting owner without
an explicit supported resource policy. Cancel cooperatively around native calls,
then force/reap the owned child if required. Retain one bounded image embedding
and bounded previous logits; clear on binding changes.

Show coarse masks as guidance, never alpha. An explicit previewed trimap conversion
with an unknown boundary band can use existing correction import/history. Save
clicks, selected candidate, provider/config/model hashes and conversion settings
in versioned provenance. Optional range guidance should reuse tracked corrections;
native SAM video propagation is a separate later capability.

Remaining prerequisites: deliberate pinned optional installation, local matching
checkpoint, explicit device/precision/budgets and actual platform/latency/VRAM/
representative-footage tests. Upstream can operate without its CUDA extension,
but post-processing differs and must be explicit. CPU tests must inject a fixture
provider labelled **provider-interface test; no SAM2 inference executed**. They
must cover click ordering/clear/refine, coordinate mapping, cancellation/ownership,
stale results, unavailable dependencies, embedding bounds and save/import/undo.


The stdlib-only `subject_guidance` interface owns one immutable source-frame
snapshot, embedding/result/logit payloads and generation. It rejects concurrent
work, fences stale results and defers provider reset until native work retires.
Explicit byte/pixel bounds and a resource lease apply to operations. Cleanup
uncertainty prevents reuse. The lazy local SAM2 adapter uses only deliberately
selected config/checkpoint files; it never installs or downloads dependencies.
Files are pinned and bounded before optional imports and rechecked around loading.
Config is limited to4MiB and checkpoint to2GiB. A pre-existing global Hydra
context refuses rather than changing another owner's configuration. Float32 is
the default; inherited CUDA TF32/non-highest matmul precision refuses without
changing global settings. BF16 must be selected explicitly.

Seventeen deterministic tests cover source/generation changes, positive/negative
clicks and iterative logits, cancellation, lease/reset ownership, file mutation
and precision drift. They do not execute SAM2. No provider controls are exposed
until the connected owner can establish the resource handoff; an idle inference
worker may still retain its GPU model. Accepted schema4 masks are ordinary
corrections and do not need provider weights to render. Click UI, native-coordinate
inspection/acceptance, integration with tracking and actual optional-model runtime
acceptance remain pending.
