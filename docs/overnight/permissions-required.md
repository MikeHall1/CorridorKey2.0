# Overnight permission ledger

Current run: OMX `omx-1792790265529-l6n3bp`, G004; branch `review/corridorkey-audit-20261023`. This ledger records **platform approval requirements**, not missing hardware or presumed consent. Explicit product/checkpoint authority does not override filesystem/network approval controls. No approval requests are pending from this overnight continuation.

## PERM-001 — Publish accepted checkpoint

- **Affected task:** checkpoint Git staging/commits/normal fork push only. Independent source work and local verification continue.
- **Proposed action:** stage the reviewed explicit product/record paths, create coherent local commits, verify remote tip and normal-push the development branch.
- **Why needed:** preserve accepted increments in Git and trigger the repaired hosted Ubuntu/Windows diagnostic workflow.
- **Permission source:** current environment filesystem policy grants `.git` read access only; escalation documentation requires approval for writes outside sandbox permissions. Network is restricted. Git performer confirmed earlier approvals were one-off commands without reusable prefix rules. Latest user overnight policy: “If an action explicitly requires permission that has not already been granted ... Do not perform it ... Record it in the permission ledger.”
- **Affected systems:** this repository `.git/index`, objects, refs/reflogs and tracking refs; only `https://github.com/MikeHall1/CorridorKey2.0.git`, branch `review/corridorkey-audit-20261023`. No global config, default-branch merge, release, history rewrite or media upload.
- **Download/cost:** ordinary Git metadata/source transfer; size not measured, no paid compute purchase.
- **Risk/rollback:** selected-file mistakes or remote drift; recheck manifests, hashes and staged diff immediately before commit. Normal push must refuse divergence; no force. Retain source and local commits on failure.
- **Safe work completed:**437-source frozen local gate,1626 tests PASS167.833s/no skips; no-new-static gate, independent code APPROVE. Prior checkpoint41e7c32 remains remote-confirmed. Implementation and raw local evidence retained; independent architectural gate CLEAR.
- **Blocked:** new commits/push and the new-push hosted CI run. Existing CI read-only inspection remains possible.
- **Morning procedure:** explicitly approve narrow repository Git writes and exact fork-branch network operations. Then revalidate current scope (later overnight work may supersede the437-source snapshot), stage **explicit reviewed paths only**, commit accepted coherent slices, and run:

  ```sh
  git diff --check
  git diff --cached --stat
  GIT_TERMINAL_PROMPT=0 git ls-remote origin refs/heads/review/corridorkey-audit-20261023
  GIT_TERMINAL_PROMPT=0 git push origin HEAD:refs/heads/review/corridorkey-audit-20261023
  GIT_TERMINAL_PROMPT=0 git ls-remote origin refs/heads/review/corridorkey-audit-20261023
  ```

  Exclude `.gitignore`, `AGENTS.md`, `.codex/`, `.serena/`, `.DS_Store`, raw audit/benchmark evidence, media, weights and caches unless separately owned/approved. Do not blindly commit all changes.
- **Required verification:** committed source matches accepted hashes; index contains only reviewed paths; remote SHA equals pushed SHA; inspect exact new-SHA hosted jobs. Passing local fixtures do not establish Windows/GPU acceptance.
- **Status:** **GRANTED AND COMPLETED — source checkpoint published2026-10-24.** User-authorised reviewed161-path commit `44c753a4950c19ecec68d6b1ce850490ef928c36` was normally pushed and independently confirmed as the fork review-branch tip. All121 excluded staged entries retained identical mode/OID/stage/flags. Previous author identity reused command-locally; no global/repository Git configuration change. This confirmation is saved in a subsequent documentation-only commit; no force/tag/default-branch/release action.

## External prerequisites — not permission requests

CUDA/NVIDIA hardware, approved local weights and representative licensed footage; actual FFmpeg; optional segmentation backend; native Windows packaging/clean machine; Resolve SDK/host/viewer/export remain distinct missing prerequisites. No download, purchase, global installation or media upload is proposed by this ledger. See `../acceptance.md` and `../product-performance-brief.md`.

## PERM-002 — Local loopback fixture acceptance

- **Affected task:**15 HTTP download fixture cases only; all other authorised unit/Qt/worker/static checks continue.
- **Proposed action:** allow local loopback listen/bind for the existing publisher fixtures, not external connections/media upload.
- **Why needed:** exercise real owned local HTTP transfer/catalogue admission through the existing fixtures.
- **Permission source:** restricted sandbox; actual prior `PermissionError: [Errno 1] Operation not permitted` at bind, `docs/audits/evidence/unit-tests.log:5548–5815`. Earlier broad-suite escalation was one-off, not a reusable grant. Overnight user policy forbids new ungranted actions or bypass.
- **Affected systems:** ephemeral local TCP listeners, temporary generated fixture files; no production services, accounts or user media.
- **Download/cost:** tiny generated local fixture payloads; no external download/purchase. Exact total bytes not measured.
- **Risk/rollback:** loopback-only owned server; fixture teardown must close/join listeners and remove only its temporary directories. Do not substitute an external server or alter fixture security to evade approval.
- **Safe work:** complete discovery will be partitioned using exact IDs; every admissible case runs.15 blocked cases are NOT VERIFIED, not silently skipped/green. Prior1626-case authorised snapshot remains historical evidence only; relevant download source/test hashes unchanged, not a claim all current dependencies are identical.
- **Blocked IDs:**14 methods in `test_asset_download.AssetDownloadTests`, plus `test_desktop_asset_download.DesktopAssetDownloadTests.test_actual_owned_http_plan_download_and_explicit_path_selection`. Exact frozen ID inventory/evidence: `benchmark-results/product-next/overnight-suite-partition.json`; nonexecuted procedure `overnight-suite-procedure.txt` alongside it.
- **Morning command after explicit grant and source revalidation:**

  ```sh
  QT_QPA_PLATFORM=offscreen OPENCV_IO_ENABLE_OPENEXR=1 \
    /private/tmp/corridorkey-compatible-venv/bin/python -m unittest \
    tests.test_asset_download.AssetDownloadTests \
    tests.test_desktop_asset_download.DesktopAssetDownloadTests.test_actual_owned_http_plan_download_and_explicit_path_selection -v
  ```

- **Required verification:** all15 current cases execute/pass with real loopback fixtures, no leaked listener/process; record exact source identity and command. At final acceptance broaden only where later changes invalidate other evidence.
- **Status:** **GRANTED AND VERIFIED — 2026-10-24.** Narrow escalated command completed all15 cases PASS1.021s (wall1.123s),0failures/errors/skips. Actual localhost fixtures used, no external downloads. Record: `benchmark-results/product-next/perm002-loopback-acceptance.json`. This closes the prior15-case verification gap; not a blanket future network grant.

## PERM-003 — Optional benchmark host metadata (read-only)

- **Affected task:** CPU-brand and physical-RAM fields for PERF-I component measurement only. Feature implementation and component observations remain usable with these fields UNAVAILABLE; no cross-machine certification is claimed.
- **Proposed action:** read the two ordinary system metadata values below after a narrow platform grant.
- **Why needed:** improve reproducibility of the local CPU component observation, not install software or measure per-job RSS.
- **Permission source:** actual `sysctl` calls returned exit1, `sysctl: sysctl fmt -1 1024 1: Operation not permitted`; recorded in `benchmark-results/product-next/profiling-overhead-results.json` → `environment.cpu_brand` and `environment.physical_ram_bytes`. No retry, alternate mechanism or escalation was attempted. Overnight policy requires deferral after an ungranted boundary.
- **Affected systems:** read-only macOS generic CPU-brand/physical-memory metadata; no serial number, credentials, global config or user files.
- **Download/cost:** none.
- **Risk/rollback:** read-only; no mutation to undo. Keep later metadata observation timestamp distinct from original experiment and do not rewrite raw measurements.
- **Safe work completed:** predeclared CPU-only scope experiment and independent artifact review, source445 unchanged; OS/Python/arm64/logicalCPUcount/timer metadata captured. Exact RAM/CPU-brand remain UNAVAILABLE.
- **Blocked:** these two optional metadata observations only.
- **Morning command after explicit approval:**

  ```sh
  /usr/sbin/sysctl -n machdep.cpu.brand_string hw.memsize
  ```

- **Verification:** record command/time/status and values without unique machine IDs; associate with the experiment only if the same host is confirmed. Do not rerun unchanged experiments just to fill metadata.
- **Status:** **DEFERRED — optional read-only platform grant**. Not a prerequisite for code work or permission to change global settings.


## Checkpoint23 safe-work update (2026-10-24)

PERM001/002/003 statuses unchanged; no deferred command executed or new approval sought. Latest safe local evidence:1805authorisedPASS208.176s/462stablePythonfiles/fullstatic0new, independent A/I source+repair/artifact approvals, raw `benchmark-results/product-next/batch13-summary.json`. Subsequent accepted work stilluncommitted; HEAD/lastpush41e7c32. Refresh any old437-source/1626-case/publication-whitelist references before later approval; historical evidence is not a current staging list. PERM002 still exact15realHTTP cases, not silently skipped/passed. Hardware/assets/FFmpeg/Windows/Resolve and Semgrep initialization remain separate technical/external limitations.


## PERM001 additional index-ownership warning — checkpoint25

Current index observation: **255 staged paths /256 porcelain entries**, including74 `.codex` paths; HEAD still41e7c32. Index mtime2026-10-24T08:06:17.268707Z, no lock at observation (not proof an unknown actor stopped). Coordinator performed no staging/Git writes; independent reviewer reports only read-only diffs. Attribution is **unverified**; no assertion this was the user, another worker or a hook. Application468-source snapshot still matches1856-test/fullstatic0new acceptance. Raw nonsecret metadata: `benchmark-results/product-next/checkpoint25-index-observation.json`.

**Do not consume, unstage/reset or rewrite that index automatically.** Preserve unrelated staged work and inspect ownership/whitelist/staged content before later explicitly approved publication. Prior staging lists are historical. This is an additional PERM001 publication-safety condition, not a new approval grant or reason to discard code/source evidence. No deferred command executed; no repair/bypass attempted.


## Checkpoint26 safe-work update (2026-10-24)

PERM001/002/003 unchanged, no deferred action executed or approval sought. E002/OPT001 accepted:1867authorisedPASS231.924s/470Pythonstable/fullstatic0newactual15+legacy/independentAPPROVE. Single fixed warmCPUcomponent cohort9observedgains/7inconclusive/0policyregressions; notGPU/endtoend/peak proof. HEAD/lastpush41e7c32; allnewworkuncommitted, unknownstagedindex preserved (checkpoint25 observation, ownershipunverified). Exactreviewedwhitelist/ownership required before any later publication. Fifteen realHTTPchecks remainNOTVERIFIED. Optional processinventory`ps` returnedOperationnotpermitted; notneededforownedtoolhandle completion, no retry/approvalbypass/unknownlivenessclaim. ExternalGPU/footage/Windows/Resolve/Semgrep-init limits separate. Evidence`benchmark-results/product-next/batch16-summary.json`.


## Checkpoint27 safe-work update (2026-10-24)

PERM001/002/003 unchanged; no deferred action/approvalretry/Gitmutation. E003 accepted1910authorisedPASS248.686s/472Py/fullstatic0newactual16+legacy/independentactualreuse+integrity APPROVE;15realHTTPNOTVERIFIED. Source `8df1240a6235484c446e8c385f3150f0358a9bd8e52913f1deba9464f58cb05a`; batch17-summary.json. Unknownstagedindex preserved; lastHEAD/push41e7c32, newerworkuncommitted. No newbenchmark/dependencyinstall/mediaupload; actualtargethardware/assets/host requirements separate. LOCALmixedDoc dependency work stillopen, no fullgoalcompletion.


## Checkpoint28 safe-work update (2026-10-24)

PERM001/002/003 remain unchanged. No deferred action, approval retry or Git mutation was performed. E004 accepted:1,923 authorised tests passed in269.822s;475 Python files stable, full static0 new versus actual checkpoint27/legacy, independent actual parity/integrity review approved.15 real HTTP tests remain NOT VERIFIED. Source `6793829cd3f2c26efcfd1a111bd552bf3731ad7026cc7282b5010bb7341d8fdd`; batch18-summary.json. Unknown staged index remains untouched; last HEAD/push41e7c32, subsequent work uncommitted. No new benchmark, environment/dependency install or media upload. Genuine neural-prefix work remains LOCAL; actual GPU/footage/Windows/Resolve/FFmpeg and Semgrep initialization are separate external/technical limitations.


## Checkpoint29 safe-work update (2026-10-24)

PERM001/002/003 remain unchanged; no deferred command, approval retry or Git mutation. Stage2 prefix and test-only RSS-oracle repair accepted:1,932 authorised PASS316.400s/477 stable Python files/full static0 new actual checkpoint28+legacy/independent approvals. Initial1,931-case/one-RSS-assertion failure retained; no production memory defect or RAM reduction claimed.15 real HTTP tests remain NOT VERIFIED. Source `9e7935bbbf0e5ff451f9e02393168ed14fec89344349f3850765a2e6f1a93925`; batch19-summary.json. Unknown index untouched, lastHEAD/push41e7c32, later work uncommitted. Temporal RVM/Hann prefix remains LOCAL; GPU/footage/Windows/Resolve/FFmpeg and Semgrep initialization remain separate prerequisites. No new downloads, environment edits or performance claims.


## Checkpoint30 safe-work update

**Accepted checkpoint30 — 2026-10-24:** RVM/Hann now reuse matching complete neural inputs/state after genuine conditioning edits under the existing opt-in policies, while current correction/carry and full output provenance remain authoritative. Critical carry fixture reuses the first raw Hann event then evaluates all5 changed suffix events; B equals clean B and differs from A. **1,941 authorised tests PASS430.142s**,0 failures/errors/skips;1,956 discovered,15 HTTP tests **NOT VERIFIED/PERM002**. **479 Python files stable**, full compile/import/dependency/static gate0 new versus actual checkpoint29/legacy;511 Ruff/678 Pyright errors+17 warnings remain. Independent exact-source review APPROVE. Source `63ab1315bba387fe3d6dcbde9624930b3ba6b4a4cf9f045d12bd736a9b96e724`; `benchmark-results/product-next/batch20-summary.json`. No new performance measurement or GPU-quality claim. Last push41e7c32; later work uncommitted/PERM001, unknown index untouched. **Conditioned full-context progressive previews remain LOCAL work; full product/target acceptance stays open.** PERM001/002/003 unchanged; no deferred action, approval retry, environment install or Git mutation. OpenCV-only timestamps cannot certify CFR; missing ffprobe CLI is a technical prerequisite, not an approval request.


## Checkpoint31 safe-work update

**Accepted checkpoint31 — 2026-10-24:** Conditioned progressive sequence previews preserve the original full temporal context, hints, corrections and complete Stage2 groups while writing only the selected EXR. Dedicated `preview_complete` admission cannot become a final export. Independent review found and reproduced a manifest snapshot race; the root fix retains the original manifest pin/hash and rejects replacement before workspace creation. **1,948 authorised tests PASS359.475s**,0 failures/errors/skips;1,963 discovered,15 HTTP tests **NOT VERIFIED/PERM002**. **480 Python files stable**, fresh full compile/import/dependency/static gate0 new versus actual checkpoint30/legacy;511 Ruff/678 Pyright errors+17 warnings remain. Independent repaired source APPROVE (`96d578ca`). Source `78bf81173ad8c85a61619c33297909a5c2ae8a1fc4510e6d35122d686752cef7`; `benchmark-results/product-next/batch22-summary.json`. Each selected preview still processes the full timeline: no latency/GPU-quality gain claimed. Last push41e7c32; later work uncommitted/PERM001, unknown index untouched. Next local work: optional existing two-anchor disagreement display. Full product/target acceptance remains open. PERM001/002/003 unchanged; no deferred command, approval retry or Git/environment mutation. Missing CFR timing-provider correspondence is a technical prerequisite, not an approval request.

## PERM-004 — Isolated optional timestamp-provider experiment

- **Affected task:** CFR video synchronized-playback provider evaluation; not existing sequence playback or accepted preview/tracking work.
- **Proposed action:** obtain public pinned PyAV19.0.2 binary wheel into a new, separately owned `/private/tmp` target, then perform local import/provider fixtures. Do not modify the user's `.venv`, compatible verification environment, manifests, global packages or packaging.
- **Why needed:** installed OpenCV cannot establish raw PTS provenance. PyAV can bind decoded pixels and timestamps in one provider; actual codec/colour/index/error/seek behavior still needs local proof. See [decision and sources](cfr-provider-decision.md).
- **Permission source:** current execution environment `<permissions instructions>` states “Network access is restricted.” Overnight request§3: “If an action explicitly requires permission that has not already been granted ... Do not perform it ... Record it in the permission ledger.” No network escalation/install attempt was made; no approval is pending. Missing package availability itself is technical, not assumed permission.
- **Affected systems/files:** public PyPI/network and only a new owned temporary target. Interpreter remains `/private/tmp/corridorkey-compatible-venv/bin/python`; process-local `PYTHONPATH` exposes experiment, never global configuration.
- **Size/cost:** matching macOS ARM64 wheel approximately18.6MB; no paid compute, footage, model weights or private-source transfer. Registry may return ancillary metadata. Windows x64 wheel approximately28.2MB, not proposed for this Mac experiment.
- **Risk/rollback:** third-party native code; bundled-library redistribution terms unresolved, not simply BSD-only/GPL-only. This is not permission to ship it. Retain logs/version/library inventory. Remove only the verified owned temporary target if rejecting the experiment; never uninstall from active/global environments.
- **Safe work completed:** read-only installed metadata/cache discovery and official source/version/licensing/API review; existing video admission retained fail-closed, existing local implementation continues.
- **What remains blocked:** actual provider import/decode/PTS/colour/index/seek/cancellation experiments. Production admission and packaging additionally need independent technical proof and redistribution review even after this approval.
- **Status:** **ACQUISITION/IMPORT PREFLIGHT GRANTED AND COMPLETED — 2026-10-24.** Pinned PyAV19.0.2 installed only into the stated owned temporary target; av+OpenCV+NumPy reader/range preflight7PASS0.543s. No active/global environment, repository dependency or packaging change. Actual PTS/CFR/precision/provider experiments and redistribution review remain unfinished; installation is not production adoption. Raw records: `perm004-pyav-install.json`, `perm004-import-reader-preflight.json` under local `benchmark-results/product-next/`.

### Proposed morning command (review first)

Installed `uv0.13.5` help verified these options. The existence guard prevents adopting or overwriting an unknown target. Public index is explicit; this is not a source build or Python download.

```sh
target=/private/tmp/corridorkey-pyav-evaluation-19.0.2
test ! -e "$target" || exit 1
mkdir -m 700 "$target" || exit 1
printf '%s\n' 'CorridorKey optional PyAV evaluation; no production adoption' > "$target/.ck-experiment-owner"
env -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_INDEX_URL \
  "$HOME/.local/bin/uv" --no-config --no-cache pip install \
  --python /private/tmp/corridorkey-compatible-venv/bin/python \
  --target "$target" --no-deps --only-binary :all: --no-python-downloads \
  --default-index https://pypi.org/simple 'av==19.0.2'
```

### Verification after approval

```sh
PYTHONPATH=/private/tmp/corridorkey-pyav-evaluation-19.0.2 \
  /private/tmp/corridorkey-compatible-venv/bin/python -c \
  'import av; print(av.__version__); print(av.library_versions)'
PYTHONPATH=tests:. OPENCV_IO_ENABLE_OPENEXR=1 \
  /private/tmp/corridorkey-compatible-venv/bin/python -m unittest \
  tests.test_reader_review tests.test_source_range -v
```

The import/reference-reader checks are preflight, **not CFR admission**. Then implement and run the decision document's generated-media provider tests: exact raw rational timestamps/pixels, missing/duplicate/VFR/error/EOF/flush, source offsets/seek/B-frames/origin, explicit colour and OpenCV inference-input/output correspondence, ownership/stale/cancel bounds. Keep footage local. Only independently accepted proof can justify a repository dependency or playback admission change; the test harness does not yet exist.


## Checkpoint32 final safe-work state

I006 accepted after1,955 authorised tests PASS350.965s/481 stable Python files/full static0newactual31+legacy;15HTTP remain NOT VERIFIED. All owned test/static/review processes terminal. No Git/index/environment mutation or pending approval. PERM001–004 remain separate, deferred decisions; latest proposed provider experiment is not production adoption. Current evidence/remaining prerequisites: [handoff](morning-handoff.md), `benchmark-results/product-next/batch23-summary.json`. No command executes automatically on return.


## Current permission review — supersedes historical deferred statuses

PERM002 closed by actual15-case pass. PERM004 acquisition/import preflight completed in isolated target; actual provider tests remain work, not permission to ship. PERM001 publication explicitly authorised by user after staged review; commit/push pending exact reviewed-path execution. PERM003 remains optional/deferred. Current portable coverage is1970 cases across two runs, plus29 fresh affected tests after test-only EOF cleanup; not one new whole-suite run. See [compact verification](verification-summary.json). Old checkpoint narratives below/above retain historical evidence, not current permission state.


## Publication confirmed

Source checkpoint `44c753a4950c19ecec68d6b1ce850490ef928c36` is verified on `MikeHall1/CorridorKey2.0`, branch `review/corridorkey-audit-20261023`.161 reviewed paths published;121 unrelated staged entries preserved exactly. PERM001 and002 are closed for this execution. PERM004 acquisition/import preflight completed in isolated target; actual codec/provider proof remains. PERM003 optional/deferred. The earlier native prerequisite-block audit is historical because new grants changed work availability. Source identity and composite portable evidence: [verification](verification-summary.json).


## PERM-005 — README model downloads (completed)

- **Affected task:** obtain the four model groups listed by the project README.
- **Authority:** current user request, “can you pull the weights from hugging face, if you view the readme”; narrow platform grants separately authorised metadata and the exact5.08GB transfer. Not inferred from silence/overnight instructions.
- **Action/files:** existing anonymous downloader resolved pinned original publishers, then created fresh ignored `weights/downloaded-assets/corridorkey-assets-wz21cm42`; no existing assets/globalcache/environment/settings overwritten or activated.
- **Actual size/cost:**35files/5,076,141,689payloadbytes; public HuggingFace and GoogleDeepMind storage, no paidcompute/mediaupload. Available local storage checked before transfer.
- **Verification:** exact sizes/publisher-declared checksums, completed sealed inventory and RuntimePaths presence check; nonexecuting identity/structure audit complete. MoGe legacyformat, tensors, modelcompatibility/GPU/quality remainunverified. No pickle/modelcode executed.
- **Risk/rollback:** preserve publisher licences; keep only this owned bundle if needed. No automatic deletion, uninstallation or overwrite. Bundle remains out of Git.
- **Status:** **GRANTED AND COMPLETED**. Actual assets now present locally; this closes the prior missing-asset prerequisite, not CUDA/hardware/quality/host acceptance.
- **Evidence/next:** [local asset root and provenance](model-assets.md), [compact verification](verification-summary.json). Future model use must select the explicit root and meet backend requirements; no silent fallback.
