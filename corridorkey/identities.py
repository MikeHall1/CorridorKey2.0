"""Local content identities shared by correction preflight and benchmarks."""
import hashlib
import json
from pathlib import Path

from .scheduling import frame_paths


def file_sha256(path, *, check=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if check is not None:
                check()
            digest.update(block)
    if check is not None:
        check()
    return digest.hexdigest()


def sequence_identity(path, files):
    """Serialize already verified ordered frame hashes using the existing wire identity."""
    files = [dict(item) for item in files]
    return {"path": str(Path(path)), "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def source_identity(path, *, check=None):
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path, check=check)}
    files = [{"name": Path(file).name, "sha256": file_sha256(file, check=check)}
             for file in frame_paths(path)]
    return sequence_identity(path, files)
