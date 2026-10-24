# Verification and static regression checks

## Environment

Use Python3.12 and the declared runtime/development manifests. Select the appropriate
Torch build separately as documented in the [README](../README.md); CPU verification
does not enable CUDA inference. The runtime requires NumPy>=1.26,<2. Install the
optional desktop test stack for Qt tests; keep desktop/build dependencies separate
from headless requirements. Do not alter an active environment merely to run checks.

## Commands

```sh
QT_QPA_PLATFORM=offscreen OPENCV_IO_ENABLE_OPENEXR=1 python -m unittest discover -s tests -v
python scripts/code_health.py --output-dir benchmark-results/code-health
python scripts/verify.py --python /path/to/python
```

The neutral verifier runs unit tests, compilation, static regression and whitespace
checks. Its finite per-gate budget is1800s; this is not an aggregate job deadline or
a guarantee that all descendant native work is forcibly interrupted. Timeout is a
failure and partial output is retained. Packaging uses this verifier without any
assistant/task-plan state. Logs remain local under `benchmark-results/`.

## Baseline policy

`tests/static-baseline.json` is the reviewed diagnostic fingerprint baseline.
Migration to this public QA path preserves diagnostic keys, occurrence counts,
checker versions and Python metadata; only its classification-document pointer was
relabeled. It is not regenerated automatically. Never suppress diagnostics, relax
checker settings or refresh the baseline merely to turn a failure green.

The static gate records every raw finding and compares path/rule/severity/message
fingerprints. A no-new-regressions result is **not static-clean**. Existing debt
remains visible; investigate actual errors rather than assuming each legacy finding
is harmless. Run targeted checks for changes and complete gates at coherent boundaries.

Hosted CI records interpreter/platform/baseline identity and emits escaped raw-report
chunks after verification, including on failure. Collect exact diagnostics before
classifying platform differences. Windows path aliases are filesystem identity issues,
not permission to ignore arbitrary paths. Hosted passes, local fixtures and declarations
have distinct scopes; none proves GPU quality, Windows packaging or Resolve compatibility.
