# Local model file auditing

Open **Local assets…**, choose the asset tree/checkpoint, then **Audit selected files… → Audit selected files**. This optional workflow reads supported container structure and streams local SHA256 hashes. It does not load a model, execute snapshot code, initialize a processing backend or download anything. Existing presence/backend checking and **Use for future jobs** retain their behavior, including legacy checkpoint selection.

Each row separately reports presence and structural **PASS**, **FAIL** or **UNVERIFIED**. A completed scan can contain failed entries. PASS establishes only the stated structural checks; model compatibility, publisher authenticity, tensor values, successful inference, GPU support and visual quality remain unverified. Local hashes identify the selected files and do not authenticate a publisher. C-RADIO's tree hash identifies its local file/path/stat inventory rather than a publisher's single-file digest.

**Save report…** explicitly re-audits the captured sources and writes the exact completed report to a new file. Changed sources or selections require another audit. Existing output files are never overwritten, even if identical. Reports cannot be saved inside the audited C-RADIO tree because adding a report would change the audited inventory. Saved and partial destinations remain visible. A private report stage may remain after failure; a fully published report remains usable with a warning if removing its stage alias fails. No automatic cleanup or retries occur.

## Structural scope

- Safetensors: bounded JSON header, duplicate/finite/type checks, tensor dtype/shape/offset extents, no gaps/overlap/unclaimed payload. Runtime checkpoints require the actual runtime format metadata; optional configuration/metadata defaults remain supported. No tensors are allocated.
- Foundation statistics: report the six documented mean/std vector names, ranks, pair shapes and dtypes. Missing or questionable vectors are UNVERIFIED because the existing loader skips missing/mismatched buffers. Model-dependent channel widths and numerical normalization values are not established.
- RVM NPZ: pre-admit the ZIP directory, bound members and expanded sizes, inspect NPY headers/dtypes/shapes/extents without loading arrays. Relative slash-separated tensor names are supported without extracting files. Object/structured dtypes, conflicting/traversal names, encryption, unsupported compression, overlapping payloads and excessive metadata are refused. Payload CRC, tensor values and RVM parameter mapping remain unverified. ZIP64 central directories are outside this audit; supported local ZIP64 headers do not imply central-directory support.
- Legacy `.pt`/`.pth`/`.bin`: hash opaque bytes and report UNVERIFIED. No pickle or `torch.load` is executed. Runtime support is unchanged.
- C-RADIO: bounded recursive file/configuration/shard inventory; check required local loader/config files and supported safetensors structure. Loader code is hashed and never executed. Alternate legacy formats and actual class compatibility remain unverified.

Symlinked local/Hugging Face snapshots are supported deliberately: the audit records lexical aliases and resolved target identities, detects changed targets and refuses directory cycles. This differs from owned recovery/queue workspaces, which refuse links. Ordinary filesystem reads may update access times. Auditing is a snapshot check, not an operating-system lock preventing later external edits.

## CLI and ownership

```sh
python -m corridorkey.asset_audit audit --asset-root /existing/assets
python -m corridorkey.asset_audit audit --asset-root /existing/assets --weights /existing/checkpoint.safetensors --output /existing/parent/new-audit.json
```

Packaged dispatch uses `--corridorkey-module corridorkey.asset_audit` in place of `python -m corridorkey.asset_audit`. Standard output is the audit JSON, or an export failure result when publication fails. A complete scan returns zero even when individual formats fail; inspect the row statuses. Failed/cancelled scans or failed exports return nonzero. Reports include selected/resolved paths, local identity, per-entry details, explicit unverified scope and a token binding the report contents.

Hash reads are at most1 MiB per block with admitted-size growth checks. Header/ZIP-directory metadata is at most8 MiB; NPY headers64 KiB; safetensors65536 tensors; C-RADIO4096 files/directories; report4 MiB. ZIP member count is bounded4096, declared expansion16 GiB and compression ratio1000. These are structural admission limits, not total process/storage budgets or proof that an archive can be fully loaded. Full hashing and export revalidation can take substantial time for real weights.

The desktop owns hashing and export in a background actor with immutable selected paths, coalesced progress, cancellation and bounded terminal delivery. Parent selection edits invalidate export. Existing presence checks and auditing exclude each other; parent/main close waits for actual owned work and pending results. No selection is automatically applied and no jobs are submitted.

## Evidence and remaining work

`benchmark-results/m33-generated-summary.json` records actual Qt audit, exact fresh report export and deliberate external fixture mutation followed by stale-report refusal. Tiny generated files exercise PASS/FAIL/UNVERIFIED together; source bytes remain unchanged by the application and no jobs are submitted. Core tests separately use valid minimal formats, slash-separated RVM names and sentinel local model code; a fresh process confirms the audit imports neither Torch nor NumPy.

This does not establish pretrained asset compatibility, publisher authenticity, GPU performance, Windows packaging or Resolve acceptance. Managed download planning, pinned CorridorKey publisher revision, deliberate cancellable downloads and full target validation remain unfinished. No real model asset was downloaded for this work.

Integrated verification:622tests/no skips45.100s,248Pythonfiles/fullapplicablegate with0newstatic findings; existingbacklog retained. Finalpostgatepurealiasmessage delta passesaffectedQt1case0.296s/scopedRuffPyright/independentreview. Sourceidentities/gatescopes/review/environment: `benchmark-results/m33-verification-summary.json`, full logs `m33-final/`.
