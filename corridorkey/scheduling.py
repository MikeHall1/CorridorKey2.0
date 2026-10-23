"""Dependency-free chronological scheduling and source discovery."""
from pathlib import Path
import re


def frame_paths(frame_dir):
    def key(path):
        # Tagged segments remain comparable for names with mixed digit positions.
        parts = tuple((1, int(p)) if p.isdigit() else (0, p.casefold())
                      for p in re.split(r"(\d+)", path.name))
        return parts, path.name
    return [str(p) for p in sorted(
        (p for p in Path(frame_dir).iterdir()
         if p.is_file() and p.suffix.lower() in {".exr", ".png", ".jpg", ".jpeg"}),
        key=key)]


def window_starts(T_all: int, chunk: int, stride: int) -> list[int]:
    if any(type(v) is not int or v <= 0 for v in (T_all, chunk, stride)):
        raise ValueError("frame count, chunk and stride must be positive integers")
    if stride > chunk:
        raise ValueError("hann_stride must be <= hann_chunk to avoid uncovered frames")
    if T_all <= chunk:
        return [0]
    starts = list(range(0, T_all - chunk + 1, stride))
    if starts[-1] != T_all - chunk:
        starts.append(T_all - chunk)
    return starts
