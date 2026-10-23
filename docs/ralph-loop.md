# Continuous implementation loop

The user requested continuous implementation until the complete brief is done. An active goal in this chat drives continuation. `.ralph/plan.json` is the resumable task/acceptance ledger; each pass implements, tests, fixes and records evidence, then selects more ready work. Five cycles is a checkpoint interval: save evidence and a concise progress report, then continue actionable work in the same loop. No milestone sign-off is needed. Unavailable GPU, held-out footage, packaging and Resolve host acceptance stay open and cannot be reported as passed. The current [repository review objective](review-objective.md) authorizes coherent checkpoint commits and normal pushes to the verified fork review branch. The loop itself grants no additional publication, large-asset download or environment-change authorization.

The chat goal is the active implementation loop. A separate CLI supervisor is provided for later standalone resumes; **do not run both in this checkout at once**. It was exercised with subprocess test doubles, not launched as a competing background editor or an OS service.

```sh
python3 scripts/ralph_loop.py status
python3 scripts/ralph_loop.py prompt
python3 scripts/ralph_loop.py verify --python /path/to/compatible/python
# After stopping the chat goal, for a standalone resume:
python3 scripts/ralph_loop.py run --python /path/to/runtime/python
```

Standalone passes use `codex exec --sandbox workspace-write --approve-for-me --json`, reusing the installed CLI's authentication/configuration and approval review. No credentials are read or copied by the supervisor. The invocation follows [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode); current local flags were checked with `codex exec --help`. No extra Ralph plugin is necessary.

Every pass writes events, last response, stderr and task-selected verification evidence under ignored `.ralph/runs/`. The implementation agent selects focused checks and reuses unchanged evidence. Complete applicable CPU/static checks run before accepting substantial integrated work; the supervisor does not duplicate full suites after every pass. The explicit verify command runs CPU unittests, compilation, syntax/import/dependency/Ruff/Pyright checks and whitespace checks. See `code-health.md`: a no-regression result retains all known static findings and explicitly reports static_clean=false; it is not a clean static-analysis claim. Missing checkers/dependencies fail verification rather than being silently skipped. Inspect skipped optional checks; green CPU checks alone do not meet GPU/quality/release acceptance. Recorded failed checks feed the next repair pass. Changes are retained on failure; no reset or auto-commit occurs. The runner locks out other standalone instances, stops after three passes with no file progress, and continues by default until stopped, an execution failure, or no actionable work remains. Each agent invocation retains its 30-minute timeout. An optional positive --max-iterations budget can bound an explicitly requested invocation, including values above five; reaching it saves a checkpoint without claiming completion.

The runner atomically saves `.ralph/checkpoint.json` after verified cycles and at the run boundary, including completed evidence, remaining acceptance, last logs and the next action. The active chat also saves a readable `docs/checkpoint.md`.

Create `.ralph/STOP` or interrupt the supervisor to stop it. The stop file is polled while an agent pass runs; remove it to resume. POSIX termination targets that managed child process group; Windows uses child termination, so inspect descendant workers before starting another owner. An abandoned `runner.lock` is not auto-deleted: first confirm its recorded process is no longer active. When no actionable work remains, external acceptance returns exit 2, never success. Even an all-done ledger requires final applicable verification before exit 0 and review of external evidence; ledger state alone is not full product acceptance.

Current external prerequisites: CUDA hardware and weights with permitted held-out footage; target-platform package smoke testing and licence review; Resolve SDK/Windows/NVIDIA host tests. Keep implementing independent ready work before declaring an impasse. This loop cannot create unavailable GPU evidence or guarantee the app keeps executing after its host is closed.
