# Common job settings and deliberate local assets

The optional Qt workspace now offers **Common job settings…**, **Prepare current job for recovery…** and **Local assets…**. All inference still uses the existing shared Python/PyTorch worker. No approximation, precision or backend fallback is selected implicitly.

## Common settings

The common form exposes exports/format/EXR precision, sequence FPS/linear/alpha association, cutout association, despill strength/screen, hint/carry trust, memory scheduling/cache/Hann/Stage2 controls, Quick Key propagation/uncertainty/adaptive controls and writer settings. Frame limit/start and profiling remain on the main workspace. Other core options remain in advanced JSON.

Opening copies the current complete settings. **Apply to future jobs** validates the combined core/writer configuration through the existing shared codec and updates the workspace without submitting work. Settings already queued remain immutable. Unexposed advanced fields survive. Numeric fields use exact text rather than rounded/clamped spin boxes; optional blank fields mean unset. If the workspace JSON changed since opening, Apply refuses to overwrite it; reopen the form to load the current state. Invalid JSON can be repaired in the advanced editor. Invalid combinations fail explicitly, including low-VRAM with RAM reuse, invalid output association and unsupported Quick Key combinations.

Video is clipped display output; EXR is linear under the current engine contract, and inherited half saturation remains. Selecting float is deliberate. Automatic scheduling is an estimate-based profile selection, not a GPU allocation limit. Quick Key is an explicit approximation. Source type/timeline and real model compatibility remain checked at their existing boundaries.

## One-job recovery recipes

Choose EXR and explicit scheduling first. Set the current source/output parent/hints/core/writer settings, then **Prepare current job for recovery…** and choose a separate existing recipe parent. Preparation runs filesystem work off the GUI thread and creates a fresh private `corridorkey-recipe-UUID` directory with an owner marker and immutable `job.json`. The saved batch freezes all job settings, uses a unique future output destination under the selected output parent, and includes an explicit runtime-weight override when selected. It copies no media, hints, corrections or model assets and creates no render output. Invalid/oversized recipes fail before owned-directory creation where preflight applies. An I/O failure may leave its newly owned incomplete recipe directory; no user source is removed.

Preparation submits **zero jobs**. In Recovery choose a separate cache workspace and explicitly Run. A durable queue workspace is required for the existing checksummed dispatch snapshot. Reuse the recipe for **Verify / resume**; preparing another recipe creates another requested destination and is not a replacement for resuming the existing request. Existing journal/output/cache verification remains authoritative. Corrupt or incomplete work recovers into fresh destinations; old artifacts remain. Saved recipe/media/model paths must remain available. Referenced corrections are validated by the existing core, not embedded in the recipe.

## Local assets

Choose an asset root containing the existing `weights/` and `vfms/` layout, optionally overriding the runtime checkpoint. The tree supplies runtime/foundation statistics, RVM, MoGe and C-RADIO paths. Safetensors retains precedence over legacy `.pt` when both default files exist. Bundled code/third-party import paths and recovery implementation identity continue to use the actual code root. `RuntimePaths` remains available from `corridorkey.runtime`; its dependency-free definition is in `runtime_paths.py`.

**Check local assets/backend** reads presence/type/size metadata and uses a bounded owned child to inspect actual Torch/CUDA availability. It does not deserialize a checkpoint, hash the entire tree, validate tensor compatibility, download assets or prove inference readiness. Even a nonempty dummy file passes presence, and is explicitly not model validation. Backend status appears above inventory; unavailable CUDA remains unsupported. There is no CPU/MPS fallback. Live GPU memory is NOT VERIFIED.

**Stop worker**, wait for clean shutdown, then **Use for future jobs**. Applying a checked selection requires unchanged paths and clean processing ownership; pending queue activation or uncertain benchmark/worker cleanup prevents a new owner. The next lazy production worker receives `--asset-root` and the optional `--weights`. No model loads or jobs submit during selection. Normal sessions retain one selected model; recovery uses fresh sessions with the same selected asset root and fixed implementation root. An explicit batch `weights` field overrides the worker default; omitted batch weights now use that default consistently. Recovery identities bind actual loaded assets as before. Asset changes still invalidate reuse through actual context/content verification.

Selection lasts for the app session and is retained by explicit New worker/queue switches through the configured factory. Desktop projects do not save it. Recovery recipes save an explicit runtime override, while foundation asset selection remains worker/session scoped. Developer benchmarks continue to use their own manifest/environment, rather than inheriting these selections. Full download management and packaged dependency/asset distribution remain unfinished; the existing deliberate download script is separate.

## Evidence and limits

Milestone22 starts from422 passing tests. Cycle counts:426,429,433,438; final440 tests. Full checker results and the final gate are recorded in the current checkpoint. No static baseline refresh, checker weakening, dependency upgrade or user-environment modification was performed.

`benchmark-results/m22-generated-summary.json` records actual form-driven recipe preparation, native EXR/controller/cache operation with an explicit constant-alpha CPU fixture, verified-existing resume and cache restore after corruption. Recipe bytes and the corrupt original were preserved. A separate production worker rejected unavailable CUDA without fallback after actual local metadata/backend inspection. GPU FPS is null and neural quality NOT VERIFIED. UI captures were inspected; `assets-current.png` puts the actual backend status before inventory.

The expanded full suite exposed intermittent native Qt crashes during garbage collection/painting. New-only fixture cleanup was insufficient. All affected existing/new test-owned desktop/canvas widgets now undergo explicit GUI-thread deferred destruction after their owners stop, with destroyed-signal assertions. Fault traces, failed gates and reruns are retained. This addresses test-owned lifetime; the underlying native crash mechanism and interactive target-platform behavior require further validation. No tests or GC checks were disabled.

Native Cocoa/Windows dialogs/drag/drop, power-loss/storage/containment, accessibility, packaging, pretrained GPU performance, held-out quality and Resolve host integration remain NOT VERIFIED. Queue paging/archive, synchronized A/B/range playback, total resource/disk feedback, advanced corrections and remaining recovery/colour/host work remain in the full brief.
