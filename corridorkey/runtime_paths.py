"""Dependency-free asset location contract; code and asset roots stay distinct."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RuntimePaths:
    package_root: Path
    weights: Path
    foundation_stats: Path
    rvm_ckpt: Path
    moge_checkpoint: Path
    cradio_dir: Path

    @staticmethod
    def _resolve_user_or_package_path(root: Path, value: str | os.PathLike | None, default: str) -> Path:
        if value is None:
            return (root / default).resolve()
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path
        return (root / path).resolve()

    @staticmethod
    def _prefer_safetensors(root: Path, stem: str) -> str:
        # Legacy pickle .pt files are still accepted if that is all that is present.
        if not (root / f"{stem}.safetensors").exists() and (root / f"{stem}.pt").exists():
            return f"{stem}.pt"
        return f"{stem}.safetensors"

    @classmethod
    def from_package(cls, package_root: str | os.PathLike,
                     weights: str | os.PathLike | None = None, *,
                     asset_root: str | os.PathLike | None = None) -> RuntimePaths:
        code_root = Path(package_root).resolve()
        root = Path(asset_root).resolve() if asset_root is not None else code_root
        return cls(
            package_root=code_root,
            weights=cls._resolve_user_or_package_path(
                root, weights, cls._prefer_safetensors(root, "weights/corridorkey_v2")),
            foundation_stats=(
                root / cls._prefer_safetensors(root, "weights/foundation_stats")).resolve(),
            rvm_ckpt=(root / "vfms/rvm/pretrain_rvm_small16_256_204031069.npz").resolve(),
            moge_checkpoint=(root / "vfms/moge/model.pt").resolve(),
            cradio_dir=(root / "vfms/cradio/C-RADIOv4-SO400M").resolve(),
        )

    def validate(self) -> None:
        missing = [
            p for p in (
                self.weights, self.foundation_stats, self.rvm_ckpt,
                self.moge_checkpoint, self.cradio_dir,
            )
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing runtime asset(s): " + ", ".join(str(p) for p in missing)
                + "\nOpen Local assets → Download model assets… to review a download plan, "
                  "or run `python -m corridorkey.asset_download plan`. "
                  "The legacy source helper is `python scripts/download_weights.py`.")
