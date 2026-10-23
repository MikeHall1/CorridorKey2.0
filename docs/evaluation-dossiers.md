# Evaluation dossiers

Developer benchmarks → Evaluation dossier builds a local dossier from selected existing `inference_pair` reports. It runs no models, downloads nothing, and uploads nothing. The CLI shares the same implementation:

```sh
python -m corridorkey.evaluation_dossier build --manifest dossier.json
python -m corridorkey.evaluation_dossier export --manifest dossier.json --output-parent /existing/export-parent
```

The manifest explicitly maps report jobs to cases, categories and splits:

```json
{
  "schema_version": 1,
  "kind": "evaluation_dossier",
  "reports": [{"id": "experiment", "path": "paired-report.json"}],
  "cases": [{"id": "hair-01", "report_id": "experiment", "job_index": 0,
             "categories": ["hair", "motion-blur"], "split": "held-out",
             "required": true, "review_pair_index": 0}],
  "policy": {"relative_regression_limit": 0.05,
             "zero_baseline_absolute_tolerances": {}}
}
```

Report paths resolve relative to the dossier manifest. Case/category/split declarations and chronological threshold predeclaration remain user assertions. Report run 0/job index selects each timed side's **candidate** quality measurement; a template's external baseline is not the opposite engine. Duplicate report paths/bytes and duplicate job selectors are rejected. The report records input hashes, actual coverage, policy and a semantic token. Export rebuilds and revalidates the captured inputs before sealing a fresh UUID directory. A destination inside a known source/output/input tree is refused. Partial directories remain explicitly unusable without `COMPLETE.json`.

Each metric keeps its own units. Paired repetitions become case medians; categories use equal-case medians. Descriptive pooled pixel/channel/pair errors are separate. Categories may overlap, so their case counts must not be summed as suite coverage. Reports retain category worst case-median and individual repetition regressions. Missing, failed, incompatible or incomplete measurements stay unassessed or partial. There is no combined quality score or product acceptance verdict.

The provisional relative regression screen defaults to 5% for lower-is-better errors. With baseline zero the percentage is unavailable; screening requires an explicitly declared absolute tolerance with exactly matching units. For example `"alpha.mae": {"units": "alpha/pixel", "value": 0.001}` under `zero_baseline_absolute_tolerances`. No epsilon or artificial zero percentage is substituted. Temporal stability must be considered alongside spatial accuracy; a stable blank matte is not evidence of good keying.

Comparisons require compatible source/range, frames/FPS, reference/mask/flow identities, geometry, metric definitions, units, populations and relevant colour/association/background contracts. New native evaluation reports record their evaluation contract, including constant composite backgrounds. Legacy reports lacking essential evidence remain unassessed rather than assuming equivalence.

Exports contain `analysis.json`, escaped `analysis.html`, spreadsheet-safe `analysis.csv`, a private assignment key, and a separate `review/` folder. By default the review folder contains a metadata template and blank ratings, with media preparation and human review pending. To include selected paired raw media, explicitly enable the checkbox and provide a covering integer byte ceiling (0..9223372036854775807), or pass `--include-review-media --max-media-bytes N`. Zero permits no media copying. Both assignments must have matching frame/role coverage for every mapped case. Source sizes, hashes and identities are checked while copying; growth, changes, cancellation or final input changes prevent sealing.

Only share `review/` with reviewers. Opaque case/file names, randomized engine assignments and neutral public ordering conceal the engine mapping; the private key stays outside that folder. Raw PNG/EXR/NPY bytes are preserved, so **embedded metadata anonymity is not verified**. This is assignment blinding, not a security boundary. Use an appropriate colour-managed external viewer for HDR/raw assets. No display conversion, professional assessment, reviewer preference, or GPU quality acceptance is invented. Ratings remain pending until humans actually review the footage.

Admission limits: 32 reports, 128 MiB total input report bytes, 4096 mapped cases, bounded repetitions/category memberships and 16 MiB dossier JSON. The desktop table limits visible rows to 4096 and retains the full bounded report for export. Existing local CPU fixtures verify mechanics only; actual representative footage, references and blinded professional review remain external acceptance work.
