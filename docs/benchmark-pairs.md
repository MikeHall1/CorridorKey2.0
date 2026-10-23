# Controlled paired measurements

`python -m corridorkey.benchmark_pair --manifest /local/pair.json --output /local/new-report.json` runs the existing inference benchmark engine in alternating baseline/candidate order. The desktop developer panel accepts the same pair manifest and inspects/exports its actual report. File selection and launching use the existing owned background task; cancellation/close retain process ownership. This command is also admitted through the explicit packaged-app dispatch route; actual Windows bundle execution remains unverified.

Pair manifest:

```json
{
  "schema_version": 1,
  "kind": "inference_pair",
  "baseline": "baseline.json",
  "candidate": "candidate.json",
  "repetitions": 3,
  "changed_settings": ["foundation_cache_mb"]
}
```

Baseline/candidate files use the existing `kind=inference` manifest format and point to actual batch configurations. Configure the baseline's RAM feature cache0 and candidate's cache64MiB, for example, with otherwise equal settings. Each declared changed setting must actually differ, and undeclared settings must match. At least three paired repetitions are required (default3); order is AB, BA, AB, and so on. Both sides must use the same model-state policy, source/hints/corrections/weights, writer, output/range/FPS/precision, seed and quality templates. Detailed profiling is refused. **Automatic VRAM scheduling is refused** for controlled pairs; automatic-selection comparison remains a separate experiment rather than an uncontrolled change to a matched pair.

The runner freezes normalized batch/configuration snapshots in a private workspace, gives samples fresh output destinations and checks input/reference/asset/configuration identities for drift. Actual requested settings must equal the frozen request, all effective runtime settings must be present, each side must remain stable across repetitions and cross-side differences must be explicitly declared. These checks prevent a changed effective chunk/stride/low-VRAM policy from masquerading as an effect of cache size alone. Source files and original output destinations are not overwritten.

Cold samples use the existing fresh-process measurement. Each warm sample uses its own fresh session plus a complete unmeasured warmup; the measured inner elapsed excludes warmup. This is not one shared resident session spanning all pairs. Filesystem caching is uncontrolled, and identity hashing itself reads inputs. Reports retain those conditions, raw per-side reports and quality results. CPU fixtures verify protocol/ownership; they do not measure pretrained GPU throughput.

Only complete successful matched finite samples produce aggregate times and speedup (baseline elapsed/candidate elapsed). Failure/cancellation retains partial evidence with null speedup; no partial-run aggregate is presented as complete. Missing quality references remain unavailable. Reports do not automatically accept quality tradeoffs or claim an optimization from timing alone. Cross-revision comparisons are unsupported by this same-engine/interpreter command.

The final report is created exclusively: an existing output file is preserved. Raw sample reports may be large; the existing bounded desktop report inspector refuses oversized data rather than silently truncating it. Private sample workspace destinations are diagnostic artifacts whose lifetime is limited to the run; use normal renders for retained production exports. Keep reports/media local. No assets/dependencies are downloaded or installed.

CK-BEN-03 in [acceptance](acceptance.md) remains in progress until representative NVIDIA/weight/footage runs and associated quality evidence are available. Existing independently recorded baseline inspection remains supported.
