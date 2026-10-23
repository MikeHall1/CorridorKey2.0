# Optional subject clicks

The correction editor's collapsed **Subject clicks** panel accepts a deliberately
selected local SAM2 configuration and matching checkpoint. Initial execution is
CPU float32 in an isolated child, using the current application interpreter.
No dependency, model or footage is installed or downloaded. Source checkout
execution is implemented; frozen application dispatch, GPU/MPS handoff and actual
optional-model compatibility remain NOT VERIFIED.

**Prepare selected frame** captures source pixels and native dimensions without
loading the model. Enable click mode to add positive/negative clicks on that
captured source, or enter native XY coordinates explicitly. Clicks are separate
from brush strokes and never modify source or authored mask pixels. Undo click
and Clear selection reset prompts; Predict is always explicit. Changed prompts
invalidate the prior acceptable result. The captured source frame and file
selections are displayed; prepare again to use different inputs.

**Predict** launches the optional provider on the existing background correction
owner. Its child retains one frame embedding and iterative logits, reused only
for compatible ordered prompts. Clear selection preserves that frame embedding;
Discard releases the owner. Cancellation, deadline, crash or protocol errors
retire the owned child before reuse. The dialog waits for verified child cleanup
before closing; cleanup uncertainty stays visible and prevents a second owner.
There is one active operation; a newer request cannot revoke its cancellation.
Native operations are isolated so the parent can terminate the child on timeout.
Detached descendants and native Windows process behavior remain unverified.

Inspect the coarse binary mask before **Accept as Keep**. This creates the
existing Keep correction at strength1 over the selected frame: pixels inside the
selection become alpha1; outside pixels remain unchanged. This can remove hair,
blur or transparency inside the coarse selection and is not an alpha-matting
model. Adjust strength/range/direction through ordinary correction controls.
Accepted Keep entries can be anchors for the existing Preview/Inspect/Accept
tracking workflow, with its existing cut/reset/reliability limitations.

Acceptance is one undoable schema4 mask/origin revision. Provenance records
clicks, source offset, exact native linear RGB hash, explicit assumed-sRGB uint8
proxy clipping/quantization, provider config/checkpoint hashes, precision/device,
previous-logit hash, mask hash and fixture/execution evidence. The saved final
correction does not need provider weights to render. Native source files and
render settings are unchanged. Source changes refuse prediction/acceptance;
source decoder changes refuse later rendering until regeneration or explicit
origin detachment. See [clean-plate guidance](clean-plate-guidance.md) for shared
schema4 dependency and binding behavior.

Bounds:4Mi native pixels,64MiB encoded source admission,128clicks,16KiB accepted
settings, one32MiB framed JSON message with bounded binary decoding, bounded
write/reply queues and64KiB stderr tail. Default child startup deadline20seconds,
operation deadline120seconds, then terminate/kill/reap. These are observation and
retained-payload bounds, not process RAM or model-transient-allocation guarantees.
Only native binary provider masks are connected initially; no implicit resizing
or fallback precision/device is used. Nonfinite source values refuse; signed/HDR
values remain intact in source and are clipped only in the declared display
proxy for optional segmentation.

Focused validation in the existing compatible environment:

```sh
PYTHONPATH=tests QT_QPA_PLATFORM=offscreen python -W error -m unittest test_subject_guidance test_subject_guidance_process test_subject_guidance_editor test_desktop_subject_guidance -v
python -m unittest discover -s tests -v
```

Provider-interface and real-child transport fixtures are distinct from actual
SAM2 execution. A real product-child missing-assets check verifies actionable
unavailability and cleanup without loading a model. Real local SAM2 execution,
latency, model/host RAM, subject quality and target platform packaging remain
NOT VERIFIED. Optional-provider research and licensing prerequisites are in
[subject-guidance-provider.md](subject-guidance-provider.md).
