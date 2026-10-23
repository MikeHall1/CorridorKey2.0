# Repository review objective

User-authorized scope, 2026-10-23. This review has its own acceptance conditions;
completion does not establish full product acceptance.

Perform a fresh, evidence-based review of the CorridorKey2.0 repository, resolve the issues found, tidy the repository, and continue through a persistent RALPH implementation/review/verification loop until the review has a measurable acceptance outcome.

Repository:
the repository root

GitHub fork:
`https://github.com/MikeHall1/CorridorKey2.0`

The previous goal accumulated approximately nine hours of execution. Treat its implementation and records as work to inspect, not proof of correctness. Preserve valuable work and verify the actual current state.

**1. Establish the current state**

Read:

- `AGENTS.md` and applicable nested instructions.
- `docs/development-brief.md`
- Current checkpoint, status, architecture, options and code-health documents.
- `.ralph/plan.json`
- `.ralph/issues.json`
- Existing acceptance and verification records.

Inspect:

- Current branch, remotes and upstream tracking.
- Working-tree changes, including untracked files.
- Recent commits and changes produced by the previous goal.
- Repository structure, dependencies, entry points and packaging.
- Existing tests, static checks and runtime evidence.
- Available hardware, model assets and SDKs.

Do not assume the repository is still at milestone27. Determine its actual state.

Preserve existing uncommitted work. Distinguish pre-existing changes from new changes and never discard unrelated work.

Establish one fresh, applicable local verification baseline. Record pre-existing failures separately from regressions introduced during remediation. If an equivalent fresh baseline was just completed against the same state, reuse it and explain why it remains applicable.

**2. Perform a fresh repository review**

Review the complete first-party application and its integration boundaries. Prioritise changed and high-risk areas for deeper inspection.

Use parallel native subagents where available, with clear ownership and up to four active subagents. Suggested review areas:

- Desktop workflows, concurrency, cancellation and shutdown.
- Processing, temporal correctness, colour, precision and resource management.
- Packaging, dependencies, repository organisation and documentation.
- Tests, acceptance evidence, Resolve integration and specification coverage.

The main agent coordinates findings, integration and shared records.

Review against applicable AGENTS.md instructions and the development brief.

Look for:

- Incorrect or incomplete behaviour.
- Broken end-to-end workflows.
- Race conditions, deadlocks and ownership mistakes.
- Source-file modification or data-loss risks.
- Stale caches, invalid cache keys and incomplete invalidation.
- Incorrect source offsets, frame ordering or range calculations.
- Silent fallbacks or misleading quality/backend claims.
- Unbounded memory, disk usage or worker growth.
- Placeholder implementations presented as working features.
- Packaging and optional-dependency failures.
- Missing or ineffective verification.
- Duplicate, obsolete or misleading documentation.
- Unused code, abandoned experiments and unnecessary abstractions.
- Attribution or licensing omissions.
- Gaps between implementation, ledgers and acceptance claims.

Separate confirmed defects from questions and optional improvements. Do not generate speculative findings merely to prolong the review.

**3. Maintain one persistent issue ledger**

Use `.ralph/issues.json` and the existing project records rather than creating competing tracking systems.

For each confirmed issue, record:

- Stable ID.
- Severity and affected requirement.
- Location and concrete failure scenario.
- Evidence or reproduction.
- Root cause, when established.
- Proposed remediation.
- Acceptance condition.
- Owner and status.
- Verification and review evidence.
- Relevant commit.

Deduplicate findings. Do not mark an issue resolved because code changed; require appropriate verification.

Retain relevant existing open issues and reconcile inaccurate statuses.

**4. Run the RALPH loop continuously**

For each actionable issue or coherent group:

1. Select the highest-priority work.
2. Define acceptance conditions.
3. Inspect affected interfaces.
4. Implement a focused correction.
5. Run proportionate verification.
6. Review the resulting diff against AGENTS.md.
7. Address review findings.
8. Update persistent records.
9. Create a checkpoint and commit when the batch is coherent.
10. Push the checkpoint commit to the verified fork branch.
11. Continue to the next issue.

Use checkpoints after approximately three to five meaningful cycles, or earlier for a substantial completed fix. The interval is not a stopping limit.

Do not create empty commits, cosmetic milestones or documentation-only churn merely to satisfy the checkpoint schedule.

Parallelise independent work. Avoid overlapping file writes and tests against files being actively modified. Only the main agent maintains authoritative shared ledgers and integration records.

Use one supervisor. Do not create recursive continuation loops.

**5. Tidy the repository**

Make the repository easier to understand, build, test and maintain.

Where justified:

- Remove confirmed dead first-party code and obsolete scaffolding.
- Consolidate duplicated configuration and documentation.
- Correct stale commands, paths, options and status claims.
- Organise files according to the established architecture.
- Update `.gitignore` for reproducible local artifacts.
- Keep generated output, benchmark artifacts, caches and model weights out of ordinary source commits.
- Preserve small intentional fixtures needed for verification.
- Document supported entry points and development commands.
- Preserve licences, notices and useful historical records.

Do not make a broad cosmetic rewrite or mass-format unrelated files.

Do not delete footage, model assets, user outputs or unknown untracked files. Remove only artifacts confirmed to be disposable, reproducible and repository-owned.

Avoid unnecessary changes to vendored dependencies.

**6. Verification and review**

Run targeted tests after focused changes. Add meaningful regression coverage for confirmed defects.

Use relevant compilation, import/dependency checks, Ruff and Pyright.

Run the complete applicable CPU suite and established static checks at substantial integration boundaries and final local acceptance. Reuse passing evidence when it still covers the unchanged state.

Do not rerun the full suite after every small edit or checkpoint.

Preserve checker settings and distinguish existing findings from new regressions. Do not suppress failures or weaken tests to obtain a pass.

Use an independent review subagent for substantial or high-risk changes when available. Review stable diffs and identify the reviewed state.

Resolve critical and high-severity findings before accepting a batch. Record the disposition of other findings. Unresolved specification violations remain open.

A code review does not replace testing, and passing unit tests do not replace actual workflow validation.

**7. Commit and push at every meaningful checkpoint**

Pushing checkpoint commits to my GitHub fork is explicitly authorised.

Before the first push:

- Verify the remote points to `MikeHall1/CorridorKey2.0`.
- Identify the intended development branch and tracking configuration.
- Use the existing appropriate development branch or create a clearly named review branch.
- Do not push to the original upstream repository.
- Prefer a review branch over pushing directly to the default branch.

For each checkpoint:

- Inspect the exact changes being committed.
- Stage relevant files deliberately.
- Exclude credentials, private paths where inappropriate, footage, model weights, caches, large generated files and unrelated changes.
- Run relevant checks and record unresolved limitations.
- Create a descriptive commit explaining the concrete change.
- Push normally to the verified fork branch.
- Confirm the push succeeded and record the commit ID.

Checkpoint commits may contain documented incomplete work, but must not falsely claim acceptance.

Do not force-push, rewrite published history, merge into the default branch, publish releases or alter repository permissions.

If authentication, branch protection or remote divergence blocks a push, record the exact problem. Do not bypass protections. Preserve a local commit where possible and continue safe independent work.

**8. Keep acceptance honest**

Continue until:

- Every confirmed locally resolvable review issue is fixed and verified.
- Repository cleanup is complete and reviewed.
- Applicable integrated checks pass, or remaining failures have an explicit evidenced disposition.
- A final review finds no unresolved blocking defects.
- Requirement and issue records accurately reflect implementation and evidence.
- The final checkpoint is committed and successfully pushed, or a specific push blocker is documented.

Do not claim full product completion unless the development brief’s required GPU, quality, packaging and Resolve acceptance criteria have also been satisfied.

When external hardware, weights, representative footage, SDKs or target environments are unavailable:

- Record the exact prerequisite and affected criteria.
- Prepare reproducible acceptance commands.
- Continue independent work.
- Keep those criteria explicitly unverified.

Never treat mocked inference or CPU fixtures as evidence of GPU performance, visual quality or Resolve compatibility.

Do not mark a blocked issue resolved merely to reach zero open issues.

**9. Prevent unproductive loops**

After three materially different unsuccessful attempts at one issue, reassess its root cause and obtain a focused second opinion where useful.

If no justified next experiment exists, record the blocker and continue independent issues.

Stop when the review acceptance conditions are met or all remaining work is genuinely blocked. Do not invent additional work to keep the loop running.

If platform execution limits are reached, save and push a safe checkpoint where possible. Do not claim continuation beyond the actual session or runner capabilities.

**10. Context and reporting**

Keep checkpoints concise and persistent.

Manage context at natural boundaries. If supported native compaction is available, checkpoint before using it. Otherwise request the required user action once. Do not restart completed work after compaction.

At each checkpoint report:

- Issues resolved and remaining.
- Concrete behaviour changed.
- Cleanup performed.
- Review findings and disposition.
- Verification executed and applicable reused evidence.
- External blockers.
- Branch, commit ID and push result.
- Next action.

Begin with the fresh review, then implement and verify the highest-priority confirmed issues. Continue autonomously toward the acceptance conditions above.