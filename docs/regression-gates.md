# Report-level regression screens (PERF-J)

`python -m corridorkey.regression_gate` reads an **existing** completed version1
`inference_pair` report. It does not run inference, alter a benchmark, choose a
winner, or claim machine performance from portable tests. Legacy runtime defaults
and existing benchmark commands are unchanged.

## Declare before evaluation

Write a version1 policy, commit/review it **before collecting candidate results**,
and pin its exact byte SHA256 in the invoking automation. Supply that pin rather
than calculating it from an untrusted policy at evaluation time:

```sh
python -m corridorkey.regression_gate \
  --report benchmark-results/paired.json \
  --policy benchmarks/suites/my-regression-policy.json \
  --policy-sha256 <previously-recorded-64-hex-byte-digest> \
  --output benchmark-results/new-regression-screen.json
```

The required policy shape is:

```json
{
  "schema_version": 1,
  "kind": "regression_policy",
  "context_sha256": "<canonical expected context SHA256>",
  "repetitions": 3,
  "performance": {"max_elapsed_regression": 0.05, "max_relative_spread": 0.1},
  "quality": [{"job": 0, "metric": "alpha.mae", "units": "alpha/pixel", "max_increase": 0.0001}],
  "required_evidence": ["numerical_parity", "observed_memory"],
  "visual_review_required": true,
  "accepted_tradeoffs": ["Explicit description of deliberately accepted changes; does not waive a failed gate."]
}
```

This is a shape illustration, **not a runnable or accepted policy**. The context
is `regression_gate.context(report)` serialized by `regression_gate.digest`.
Establish its expected value from an independently reviewed workload specification
or prior matched calibration report, not by blessing the candidate being gated.
It binds actual repository/assets/weights/corrections, hardware/dependencies,
requested and effective per-job settings, actual source/hints/writer and model-reuse records, actual model lifecycle/full warmup/cache metadata, frozen quality templates and result-free quality/reference/frame/definition contracts,
reported timing/warmup/filesystem-cache conditions and exact repetition count.
Ephemeral output destinations alone are excluded. Same-engine configuration
comparisons only: cross-revision pairs are not supported by the pair runner.
Uncontrolled filesystem cache is recorded, not silently called cold or warm.

A policy pin proves unchanged bytes relative to the supplied pin, **not chronology,
authorship or authenticity**. Every output explicitly marks chronological
predeclaration NOT VERIFIED. External CI/review records must establish it. Likewise
report hashes identify the raw bytes inspected; they do not authenticate measurements.

## Decisions and evidence

- The existing strict report reader validates terminal pair structure and alternating
  order. The gate additionally rematches frozen jobs and actual requested/effective
  settings and checks identity stability across **every** sample. It requires
  3..128 paired repetitions. Missing context, partial/cancelled source runs,
  mismatched policies, unsupported schemas and invalid data are `not_verified`.
- Performance uses raw measured elapsed time, never the report's cached speedup.
  Candidate/baseline elapsed ratios must stay within `1 + max_elapsed_regression`;
  equality passes. Any ratio envelope crossing the limit is `inconclusive`.
  Spread is `(max-min)/median`, checked on each side and paired ratios. Exceeding
  `max_relative_spread` is `inconclusive`, even for apparent improvements or regressions.
  This is an observed envelope, **not a confidence interval or statistical power claim**.
- Quality limits are absolute **increases in error** for explicit job/metric/unit
  selections. Supported metrics are the existing dossier's alpha MAE, boundary
  MAE, forward-gradient MAE, foreground/composite linear RGB MAE and temporal
  residual MAE. Existing quality schema, reference/source/range/binding,
  metric/population/geometry and repeated measurement contracts are revalidated.
  Equality passes; any observed increase above tolerance fails. Missing requested
  metrics stay `not_verified`; no error is invented from timing.
- `numerical_parity` (direct output-to-output numerical comparison), `observed_memory`
  and human visual review have no independently bound acceptance evidence in this
  pair schema. Requiring them yields `not_verified`. Reference-agreement MAE is
  **not** direct pixel parity or absolute visual quality. Required visual review
  cannot be waived by arbitrary approval JSON. Tradeoffs remain explicit recorded
  declarations, not overrides. Add real supported evidence before enabling product
  acceptance; do not turn off requirements merely to obtain green output.
- Aggregate precedence: `fail`, `not_verified`, `inconclusive`, then `pass`.
  `pass` means only the selected supported screen passed. With empty quality and
  evidence requirements it is deliberately just a timing screen, not quality or
  product acceptance. Cancellation during evaluation gives `cancelled`.

Bounded regular-file reads reject duplicate keys, nonfinite JSON and oversized or
excessively nested inputs. Output is exclusively created at a new destination,
with shared report-path alias protection; sources and policies are never overwritten.
Outputs preserve raw input byte hashes/sizes, exact policy, context digest, raw
elapsed ratios and per-repetition quality changes. Exit0 means screen pass;
exit1 includes fail/inconclusive/not-verified; cancellation exits130.

## Portable validation

```sh
/private/tmp/corridorkey-compatible-venv/bin/python -m unittest discover -s tests -p test_regression_gate.py -v
/private/tmp/corridorkey-compatible-venv/bin/ruff check corridorkey/regression_gate.py tests/test_regression_gate.py
/private/tmp/corridorkey-compatible-venv/bin/pyright --pythonpath /private/tmp/corridorkey-compatible-venv/bin/python corridorkey/regression_gate.py tests/test_regression_gate.py
```

Analytic fixtures deliberately regress raw timings and actual generated EXR alpha
reference-agreement errors, exercise exact thresholds, noise, missing/mismatched
context, policy tampering, cancellation and source/output aliases. These are
portable correctness checks, not GPU throughput, learned-model quality, memory,
Windows or Resolve acceptance.
