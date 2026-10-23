# Managed inference scratch ceiling

The workflow is locally verified against the shared CPU orchestration path. This is one enforceable resource policy; total RAM, VRAM, host sharing and physical disk limits remain unfinished requirements.

Set **Common job settings… → Engine → Managed scratch ceiling bytes** before submitting a job. Blank means unlimited and preserves the previous behavior. An integer from0 through9223372036854775807 sets the ceiling in exact bytes;0 permits no managed scratch-map payload. Applying affects future jobs. Existing queued requests keep their original settings.

The shared engine computes the aggregate logical size of the scratch files it will create, using the actual selected source geometry/frame count, effective memory mode, model dimensions and final scheduled Quick Key anchors. It refuses an excessive plan before creating output writers or scratch mappings. It does not silently switch memory modes, precision, scheduling, backend or quality. Source probing, output directory creation and optional shot-analysis traces can precede admission.

The covered files are the float16 RVM prepass, float32 low-VRAM foundation prepasses, float16 low-VRAM Stage1 baton and float32 Quick Key anchor alpha maps. Only the files required by the chosen execution path count. All persist until owned job cleanup, so their logical lengths count together even if a mapping has already closed. Reopening an existing file for reading does not charge it again. Allocation fences check the plan before creating writable mappings.

This is not a physical free-space reservation or whole-disk cap. Filesystem overhead, sparse-file allocation, rendered outputs, logs, durable caches, other attempts, other applications and RAM/VRAM are outside this ceiling. A job admitted by this policy can still fail for insufficient actual disk or memory; existing cancellation, writer failure and cleanup handling still applies. A full-rate streamed job without RVM can require zero scratch-map bytes.

The same `temp_map_budget_bytes` optional setting travels through desktop advanced JSON, immutable job requests, batch manifests and the CLI:

```sh
python infer.py --input /local/source --output_dir /local/new-output --temp_map_budget_bytes 1073741824
```

Omit the flag or use JSON null for unlimited. This setting is independent of the retained foundation-cache byte budget and the heuristic automatic VRAM selection cap. No new dependency, environment setting or global allocator policy is required.

Diagnostics record the plan, required bytes, configured ceiling and managed allocation outcome under `execution.temp_maps`. Verification will distinguish the tiny CPU orchestration fixtures from actual pretrained inference and target-platform resource acceptance.

Verification: actual Qt settings/direct shared CPU runtime accepts exactly49494 bytes and refuses49493 before maps/writers, with identical EXR output bytes to unlimited and original sources preserved. Independent review additionally exercised automatic low-VRAM selection and cut-expanded anchors. Full655CPU tests/no skips pass44.675s; one subsequent fixture-only type assertion passes its affected test; complete final static/compile/import/dependency checks pass with no new findings. Existing diagnostics remain. Exact states and initial failing-gate evidence are preserved in `benchmark-results/m35-verification-summary.json` and `m35-final/composite-evidence.json`. This does not establish actual GPU memory, pretrained output quality, Windows or Resolve acceptance.
