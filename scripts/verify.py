"""Run the repository CPU, compilation, static regression and whitespace gates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Accepted CPU suite took 350.965s; allow the same 30-minute budget as the CI job.
GATE_TIMEOUT_SECONDS = 1800


def verify(python, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [[python, "-m", "unittest", "discover", "-s", "tests", "-v"],
                [python, "-m", "compileall", "-q", "corridorkey", "scripts", "tests", "infer.py", "run_interactive.py"],
                [python, "scripts/code_health.py", "--output-dir", str(output_dir / "code-health")],
                ["git", "diff", "--check"]]
    results = []
    for index, command in enumerate(commands):
        try:
            completed = subprocess.run(command, check=False, cwd=ROOT, capture_output=True,
                                       text=True, timeout=GATE_TIMEOUT_SECONDS)
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
            status = 'PASS' if returncode == 0 else 'FAIL'
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors='backslashreplace') if isinstance(exc.stdout, bytes) else exc.stdout or ''
            stderr = exc.stderr.decode(errors='backslashreplace') if isinstance(exc.stderr, bytes) else exc.stderr or ''
            returncode, status = None, 'TIMEOUT'
        log = output_dir / f"check-{index}.txt"
        log.write_text(stdout + stderr, encoding='utf-8')
        results.append({"command": command, "returncode": returncode, "status": status,
                        "timeout_seconds": GATE_TIMEOUT_SECONDS, "log": str(log)})
    result = {"created_utc": datetime.now(UTC).isoformat(),
              "success": all(r["returncode"] == 0 for r in results), "checks": results,
              "note": "Inspect skips and code-health/health.json. Static no-regressions does not mean static-clean. CPU tests are not GPU inference/quality or release acceptance."}
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmark-results/verification')
    args = parser.parse_args(argv)
    result = verify(args.python, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
