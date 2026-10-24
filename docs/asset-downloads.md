# Deliberate model asset downloads

The workflow is fixture-tested and now has an actual production transfer: the README
asset set (35files/5.08GB) was downloaded2026-10-24 from pinned public publishers,
with exact-size/checksum verification and a nonexecuting local structure audit.
See [asset root/provenance](overnight/model-assets.md). No model tensors/code were
loaded. MoGe legacyformat, pretrained compatibility, actualGPU/quality and Windows
remain unverified; downloaded weights are not bundled in Git.

Open **Local assets… → Download model assets…**. Opening reads the offline publisher catalog only. **Resolve online plan** explicitly requests bounded publisher metadata and exact byte sizes. It does not download model weights. Review the publishers, licence links, pinned revisions, local file paths, checksums and total payload bytes. The C-RADIO snapshot includes local Python code that the runtime will load only during later explicit model use.

Choose an existing destination parent separate from current assets, and enter a payload byte budget covering the reviewed total. Changing the destination or budget retains the immutable metadata plan; it clears authorization to select any previous result and updates the execution summary. **Download reviewed plan** captures the plan, destination, budget and protected paths for one background operation.

Each attempt creates a fresh unique `corridorkey-assets-*` child. It never overwrites existing assets or modifies the global Hugging Face cache. Failed or cancelled attempts keep their actual partial destination and counts. No automatic retries, resume, cleanup or path selection occur. **Select downloaded paths in Local assets** fills the asset tree/checkpoint fields and invalidates the previous presence check. **Check**, stopping the worker and **Use** remain separate actions. No inference jobs are submitted by downloading or selecting paths.

## Provenance and integrity

The offline catalog follows `scripts/download_weights.py`, `THIRD_PARTY_NOTICES.md` and the repository licence. CorridorKey's requested HEAD is resolved to a specific40-character commit before file URLs are built. C-RADIO and MoGe retain the existing pinned revisions. The C-RADIO snapshot includes the publisher's files except the existing `*.pth.tar` duplicate exclusion. Required model/configuration files must be present.

Executable plans require known sizes and a supported declared checksum for every file. Large Hugging Face files use published LFS SHA256; ordinary Git files use the Git object SHA1 of `blob <decimal-size>\0<body>`, rather than the raw body SHA1. RVM retains the repository's published legacy MD5 checksum, with an additional observed local SHA256 identity. These checks establish the stated transfer identities; they do not establish neural compatibility or visual quality.

Plan tokens detect changed content and validation independently enforces the fixed catalog, pinned origins, supported checksums and safe destination layout. A token is not a publisher signature. Signed redirect URLs and authentication state are not persisted in the plan/provenance. Requests are anonymous; access failures are explicit, without forwarding credentials or silently reading global authentication/cache state.

Payloads stream into private files with expected-length growth checks before additional writes. Each file is verified before publication; the complete `asset-download.json` provenance marker is published last. A complete tree means all reviewed payload files were transferred and verified, not that a model was instantiated. Compatibility, GPU support, tensor values and professional quality require actual target validation.

## Limits and cancellation

The plan limits payload to1 TiB and4096 files, metadata responses to8 MiB and plan JSON to4 MiB. File paths must satisfy portable collision rules, components of at most255 UTF8 bytes, depth at most32, and at most16384 distinct raw parent directory paths before creation. The explicit user byte budget covers payload bytes, including downloaded partial payload, rather than filesystem metadata, provenance overhead, other attempts or total process/storage budgets. There is no free-space reservation.

Known current asset paths are captured before downloading; destination parents inside those paths are refused before creation. A common ancestor can host a fresh sibling. The CLI's repeated `--protect` argument supplies equivalent known paths. These guards do not identify unrelated footage or output directories that were never supplied.

HTTP operations use bounded reads, finite socket/deadline checks, cancellation, identity content encoding and at most five redirects. Production URLs require HTTPS. Native DNS resolution and operating-system connection calls can delay cancellation beyond Python checks; the UI waits for actual owned work rather than reporting a stopped owner prematurely. Production DNS/CDN and target filesystem behavior remain unverified. Cooperating operation ownership is not a hostile-filesystem transaction or power-loss durability guarantee.

Parent presence checks, auditing and downloads exclude each other. Progress is coalesced, work inputs are immutable, and parent/main close waits for actual worker termination and terminal delivery. Partial or failed trees cannot be selected through this workflow. A late cancellation cannot erase an already verified committed outcome.

## Headless commands

```sh
python -m corridorkey.asset_download catalog
python -m corridorkey.asset_download plan --output /existing/parent/new-plan.json
python -m corridorkey.asset_download download --plan /existing/parent/new-plan.json --output-parent /existing/download-parent --max-bytes EXACT_REVIEWED_BUDGET --protect /existing/assets
```

`catalog` is offline. `plan` (alias `resolve`) performs metadata requests. `download` fetches the reviewed payload only when explicitly invoked. Saving a plan requires a fresh file and never overwrites an existing one. JSON terminal results report actual destination, usability, payload/file counts, partial files, error/warning and scope. Packaged dispatch uses `--corridorkey-module corridorkey.asset_download` in place of `python -m corridorkey.asset_download`. The original `scripts/download_weights.py` flags and behavior remain available as a source companion.

## Historical fixture verification scope

`benchmark-results/m34-generated-summary.json` records actual Qt catalog → metadata plan → checked fresh download → explicit selection using an owned tiny local HTTP publisher fixture. All fixed public URLs are mapped to127.0.0.1 by a test-only injected transport; the RVM checksum is replaced with a tiny fixture oracle inside that process. No remote publisher/model data is requested. The workflow preserves existing asset bytes, leaves presence validation unready until requested, submits no jobs and refuses a new download inside the selected tree before file/network I/O.

Loopback server binding requires a sandbox exception; automatic review approved the isolated local test commands. This does not authorize downloading real weights. Production downloads, actual pretrained compatibility, Windows packaging, GPU performance and Resolve acceptance remain separate unfinished requirements.

Final integrated verification:645 tests/no skips in45.313s,252 Python files, complete imports/dependency/compile/whitespace and no-new-static gate pass. Existing static findings remain. Independent final review and main integration review accepted fixes189–196; exact source/config/environment identities and scope are in `benchmark-results/m34-verification-summary.json`, with gate archive `benchmark-results/m34-final/`. Native Qt lifetime issue126 remains open; passing fixtures do not establish its root cause.
