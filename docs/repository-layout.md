# Repository contents and local-only files

This repository distributes the program, plugin source, build inputs, tests and
maintained user/developer documentation—not a particular workstation's sessions.

| Paths | Git policy |
|---|---|
| `corridorkey/`, entry points and launchers | Keep shared program implementation and supported interfaces |
| `native/ofx/`, `third_party/` | Keep plugin/harness/build inputs, dynamic modules, licences and provenance |
| `tests/`, `tests/static-baseline.json`, `benchmarks/suites/` | Keep portable fixtures, regression gates and workload contracts |
| `.github/workflows/`, requirements, `scripts/verify.py`, `scripts/code_health.py` | Keep reproducible build/test/verification configuration |
| README, LICENSE, notices and maintained feature/architecture guides | Keep product documentation and honest limitations |
| Local assistant folders/instructions, task ledgers, prompts and session state | Ignore; keep only on the workstation |
| `docs/audits/`, `docs/checkpoints/`, `docs/overnight/`, environment dumps and review/task reports | Ignore; local evidence, not application inputs |
| Environments, caches, editor state, generated builds, media, outputs and model assets | Ignore; never sweep into commits or bundles |

A file being AI-assisted is not a reason to remove useful program code or regression
tests. The distinction is its product/build role and consumers. Optional platforms,
backends and dynamic imports are not obsolete because one machine does not use them.

`.gitignore` does not untrack files already in Git. Hygiene changes remove selected
paths from the index while preserving their local bytes. Normal commits remove them
from the current repository tree, **not existing history**. Do not rewrite published
history or delete local work as a side effect. Review explicit paths before staging;
never commit environments, secrets, footage, model weights or unknown local artifacts.

[Verification](code-health.md) and [packaging](packaging.md) do not require local
assistant configuration. Only legitimate QA baseline data remains public; no task
plan is required to build or test a clean checkout.
