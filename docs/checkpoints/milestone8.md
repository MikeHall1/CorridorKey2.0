# Run checkpoint — milestone 8, 2026-10-21

Five implementation/verification cycles completed. The preceding goal turn was verified progress (milestone 7), archived in `checkpoints/milestone7.md`. The user updated the objective during the fifth-cycle boundary to add continuous code-health/type-safety work. The full updated file was read. This run stops at that boundary; no extra implementation loop, supervisor or competing editor was started.

## Implemented this run

1. Native uint16 RGB/grayscale image decode into float32 without eight-bit quantization, per-decoded-frame dtype observations and explicit non-finite/geometry-change rejection. Model/foundation conversion remains bfloat16; this is not sixteen-bit end-to-end inference.
2. Explicit scene-linear RGBA input unpremultiplication and linear EXR premultiplied cutout export. Defaults retain ignored source alpha and straight output. Composite fixtures include zero/soft/one coverage, signed/HDR RGB, solids and generated textured backgrounds.
3. Optional native spatial soft-alpha adjustment, blue/green despill, explicit linear space, linear-Y preservation and alpha-edge weighting. Defaults keep radius 0/off and inherited green despill equations. Exact 0/1 endpoints remain fixed; no accuracy improvement is claimed.
4. Runtime integration evaluates adjustment once per neural/output frame before authored constraints. Both memory paths verify author-mask precedence, association roundtrip and metadata, and fail/clean up invalid model/decode outputs. Source/colour/coverage assumptions, half saturation and unsupported legacy high-bit video behavior are explicit. No automatic primaries/ICC/OCIO/PQ/HLG transform is claimed.
5. Updated docs, ran repeated correctness/compilation/import/dependency/syntax checks, executed an actual unavailable-CUDA CLI prerequisite check, and recorded the new code-health baseline and classified open issues. The serializer/type fix is deliberately queued for the next run after contract/tool verification rather than exceeding this run's limit.

## Verification actually performed

- Temporary numeric env: **164/164 tests pass**, no skips, all three repetitions (`benchmark-results/cpu-adapters-milestone8.json`).
- System env: **89 pass / 75 skip**, all three repetitions (`cpu-stdlib-milestone8.json`). It now has NumPy/OpenCV, so this historical filename is not evidence of a dependency-free run.
- Existing project `.venv`: **164/164 pass once** (`project-venv-code-health-m8.txt`). No packages there were changed by this run.
- Compilation/whitespace: PASS via `.ralph/runs/manual-check/verification.json`.
- Full nonignored repository Python AST scan: **100 files / 0 errors**, `benchmark-results/code-health-milestone8.json`.
- Selected engine imports: PASS in temporary env, with no model instantiated.
- `pip check`: PASS for installed-package consistency in temporary/project envs, but **project dependency compatibility is NOT VERIFIED / nonconforming** because both use NumPy 2 against `numpy<2`.
- Ruff/Pyright/basedpyright/Pylance: **NOT VERIFIED**, tooling/editor diagnostics unavailable. No clean static-analysis claim is made.
- Actual CLI colour-control request: `colour-m8-prerequisite.json` records missing CUDA and null FPS. This is expected environment rejection, not successful inference.

CPU runtime tests use deterministic neural substitutes. No real-model output quality, GPU speed/precision parity, encoded timing, photographic/held-out composite validation or Resolve compatibility has been proven. Test-process timing is not inference throughput. Generated fixtures/reports remain local ignored artifacts.

## Problems found and fixed during feature work

Native uint16 inputs were unsupported and now retain levels through CPU decode. Invalid values/overflow could otherwise reach clipping/export; explicit validation now fails them instead of treating clipping as repair. Refinement is placed before authored constraints and avoids repeated processing of neural anchors. Default despill matches old equations exactly in finite CPU fixtures. The filter sigma has an explicit numerical lower bound, and extreme despill arithmetic fails clearly. These changes predate the updated code-health objective; the new classify-before-fix process applies to subsequent remediation.

## Newly classified issues and next action

Read `docs/code-health.md` and `.ralph/issues.json`. The next tasks are now `code-health`, then `serializer-types`:

- HEALTH-TYPE-001: typing problem; dataclass classes satisfy is_dataclass and reach instance-only asdict. Current inspected production callers pass instances, so this is not claimed as a confirmed valid-input production failure. Actual Pylance/Pyright warning remains NOT VERIFIED.
- HEALTH-SERIAL-002: potential defect; mixed sets reproduce an incomparable-type sorting error and unsupported serialization policy needs definition.
- HEALTH-TOOLS-003: environment issue; no installed/configured owned Ruff/Pyright baseline.
- HEALTH-ENV-004: environment issue; interpreters differ, current NumPy versions violate the project bound and the selected temporary env lacks multiple runtime dependencies.

First establish justified isolated tooling and a verified compatible interpreter/dependency environment without modifying user `.venv` or weakening diagnostics/requirements. Then inspect the serializer contract/callers, narrow dataclass instances appropriately, reject unsupported classes deliberately, preserve recursive containers/Paths, define deterministic mixed-set behavior and add focused tests. Re-run the original diagnostic and related suites. Do not globally suppress warnings, mass-format unrelated source or infer checker success from test success.

## Remaining product work and external prerequisites

The complete development brief remains active. Colour work is still in progress: higher-bit video, managed source primaries/transfers and real GPU/host colour/precision acceptance are unfinished. Reference quality metrics, durable verified recovery, desktop UI/editor/queue, packaging and official SDK/worker integration remain queued after urgent code health. Scalar FPS does not establish variable-rate alignment.

This Mac has no validated inference backend. CUDA hardware, model/foundation weights, permitted development/reference/held-out footage and any training dataset remain missing, as do actual Windows/package/export and Resolve SDK/NVIDIA-host acceptance. No media was uploaded; no source/assets downloaded, pushed, published or deployed. Existing changes remain uncommitted. The task and issue ledgers plus this checkpoint preserve the resume state; the whole goal is not complete.
