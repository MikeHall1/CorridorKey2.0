# README model assets — downloaded2026-10-24

The user explicitly requested the README weights. The existing anonymous, pinned,
checksum-verifying downloader retrieved **35files /5,076,141,689bytes** in774.006s.
A fresh bundle was used; original weights/placeholders were not overwritten.

## Local asset root

`weights/downloaded-assets/corridorkey-assets-wz21cm42`

This is the asset root, **not** the checkpoint filename. It contains:

- `weights/corridorkey_v2.safetensors` and `weights/foundation_stats.safetensors`
- `vfms/cradio/C-RADIOv4-SO400M/`
- `vfms/moge/model.pt`
- `vfms/rvm/pretrain_rvm_small16_256_204031069.npz`
- `asset-download.json`: sealed plan, transfer identities and35-file inventory.

The entire bundle is Git-ignored, including publisher Python files. No weights,
media or publisher model code are included in the source/provenance commit. No
model code was executed, environment modified or asset selection activated.
For desktop use, this root belongs in the existing **Local assets** asset-tree
selection; its checkpoint is the first weights path above. Existing requests and
queued jobs were not changed. The shared worker supports explicit `--asset-root`.

## Sources and verification

- corridorkey: https://huggingface.co/nikopueringer/CorridorKey2.0; revision cf44a8db3f024bb49fd59daa408848ccf05f7477
- cradio: https://huggingface.co/nvidia/C-RADIOv4-SO400M; revision c0457f5dc26ca145f954cd4fc5bb6114e5705ad8
- moge: https://huggingface.co/Ruicheng/moge-2-vitb-normal; revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368
- rvm: https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz; revision fixed publisher object / published legacychecksum

Every transfer matched exact metadata lengths and publisher-declared hashes.
RVM retains the repository's legacyMD5 plus an observed localSHA256; these are not
publisher signatures. The subsequent existing **non-executing** audit completed:
all runtime paths exist, main/stats safetensors, RVM NPZ and C-RADIO tree structure
pass. MoGe is an opaque legacy checkpoint: bytes transferred/hashed, but its format
and tensors remain UNVERIFIED because pickle/tensor loading was deliberately not
performed. Download checks and the standalone identity audit have different scopes;
the latter alone reports publisher verification NOT VERIFIED.

The [compact verification record](verification-summary.json) records component
hashes and exact scope. Large raw reports remain local under
`benchmark-results/product-next/model-assets-*` and inside the bundle marker.

## Not accepted by downloading

Actual pretrained loading/compatibility, tensor values, NVIDIA/CUDA inference,
visual/temporal quality, performance, Windows packaging and Resolve remain
**NOT VERIFIED**. The current Mac verification environment is CPU-only; there is
no implicit CPU/MPS fallback or CUDA installation. Retain existing publisher
licences and notices; this download does not broaden their terms.
