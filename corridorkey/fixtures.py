"""Generate tiny deterministic PNG sequences for correctness, not quality claims."""
import argparse
import json
from pathlib import Path
import struct
import zlib


def _png(path, width, height, channels, pixels):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + pixels[y * width * channels:(y + 1) * width * channels]
                    for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def generate(output, frames=17, width=64, height=36, fps=24):
    if min(frames, width, height, fps) <= 0:
        raise ValueError("fixture dimensions, frame count and FPS must be positive")
    output = Path(output)
    # Fail before writing any data rather than mixing stale/new sequence frames.
    output.mkdir(parents=True, exist_ok=False)
    for name in ("rgb", "alpha"):
        (output / name).mkdir()
    for t in range(frames):
        center = width * (0.2 + 0.6 * t / max(1, frames - 1))
        alpha = bytearray()
        rgb = bytearray()
        for y in range(height):
            for x in range(width):
                a = max(0, min(1, min(width / 8 - abs(x - center), height / 4 - abs(y - height / 2)) + 0.5))
                value = round(255 * a)
                alpha.append(value)
                rgb.extend((value, 255, value))
        _png(output / "rgb" / f"frame_{t:06d}.png", width, height, 3, bytes(rgb))
        _png(output / "alpha" / f"frame_{t:06d}.png", width, height, 1, bytes(alpha))
    (output / "fixture.json").write_text(json.dumps({
        "schema_version": 1, "synthetic": True, "provenance": "corridorkey.fixtures",
        "usage": "generated correctness fixture; not professional quality reference",
        "frames": frames, "width": width, "height": height, "fps": fps,
        "colour": "8-bit display-coded RGB; synthetic display-space composite",
        "alpha": "coverage 0..255, soft moving rectangle", "cuts": []}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args.output)
