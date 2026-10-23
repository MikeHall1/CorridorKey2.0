#!/usr/bin/env python3
"""Resumable implement/verify loop. Active chat and CLI must not run concurrently."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / ".ralph/plan.json"
STATES = {"done", "ready", "in_progress", "external"}


def load_plan(path=PLAN):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("tasks"), list):
        raise ValueError("invalid loop plan")
    seen = set()
    for task in data["tasks"]:
        if task.get("id") in seen or task.get("status") not in STATES:
            raise ValueError("duplicate task or invalid task state")
        seen.add(task["id"])
        if not task.get("acceptance"):
            raise ValueError("every task requires acceptance criteria")
        if task["status"] == "done" and not task.get("evidence"):
            raise ValueError("done tasks require evidence; unavailable hardware is not passed")
    return data


def next_task(plan):
    return next((t for t in plan["tasks"] if t["status"] in {"in_progress", "ready"}), None)


def prompt(plan):
    task = next_task(plan)
    if task is None:
        return "No actionable tasks remain. Report external prerequisites honestly; do not claim product completion."
    return f"""Work in {ROOT}. Read AGENTS.md, docs/development-brief.md, docs/status.md,
.ralph/plan.json and relevant source. Preserve all existing local work.
Continue the user's Ralph loop. Implement the next coherent increment:
{json.dumps(task, indent=2)}
Perform one implementation/verification cycle. Do actual implementation, run
focused meaningful tests, fix failures, update docs/status.md and the task ledger
with evidence and reviewed diff identity, then identify the next task. Update
docs/acceptance.md. Reuse unchanged evidence; do not run the full suite per cycle.
Run complete applicable CPU/static checks before accepting substantial integrated
work. Five cycles is a checkpoint interval, not a stop condition.
Do not stop for a milestone approval. Do not push/publish, bypass approvals,
download footage/large weights, fabricate measurements, or mark external GPU,
quality, packaging or host validation as passed. Keep all source media local.
Do not run another Ralph supervisor or spawn competing writers in this checkout.
If prerequisites block one task, record it as external and continue independent
ready work. Full completion requires all acceptance criteria, not merely tests
passing. Do not weaken checks to manufacture success.
"""


def verify(python, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [[python, "-m", "unittest", "discover", "-s", "tests", "-v"],
                [python, "-m", "compileall", "-q", "corridorkey", "scripts", "tests", "infer.py", "run_interactive.py"],
                [python, "scripts/code_health.py", "--output-dir", str(output_dir / "code-health")],
                ["git", "diff", "--check"]]
    results = []
    for index, command in enumerate(commands):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        log = output_dir / f"check-{index}.txt"
        log.write_text(completed.stdout + completed.stderr)
        results.append({"command": command, "returncode": completed.returncode, "log": str(log)})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "success": all(r["returncode"] == 0 for r in results), "checks": results,
              "note": "Inspect skips and code-health/health.json. Static no-regressions does not mean static-clean. CPU tests are not GPU inference/quality or release acceptance."}
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def fingerprint():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, capture_output=True)
    digest = hashlib.sha256()
    for name in sorted(set(result.stdout.split(b"\0")) - {b""}):
        path = ROOT / name.decode()
        if path.is_file():
            digest.update(name)
            digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(plan, *, cycles, reason, last_logs=None, path=None):
    task = next_task(plan)
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cycles_completed": cycles,
        "checkpoint_interval": 5,
        "boundary_reason": reason,
        "completed": [{"id": t["id"], "evidence": t["evidence"]}
                      for t in plan["tasks"] if t["status"] == "done"],
        "remaining": [{"id": t["id"], "status": t["status"], "acceptance": t["acceptance"]}
                      for t in plan["tasks"] if t["status"] != "done"],
        "next_action": task if task is not None else "Resolve recorded external prerequisites.",
        "last_verification_logs": str(last_logs) if last_logs is not None else None,
        "product_complete": False,
    }
    target = Path(path) if path is not None else ROOT / ".ralph/checkpoint.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
    return result


def run(args):
    if shutil.which(args.codex) is None:
        raise FileNotFoundError("codex executable unavailable; install/authenticate the CLI before standalone use")
    directory = ROOT / ".ralph/runs"
    directory.mkdir(parents=True, exist_ok=True)
    lock = ROOT / ".ralph/runner.lock"
    # Exclusive creation prevents multiple standalone supervisors editing together.
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    stalled = 0
    cycles = 0
    logs = None
    boundary = "interrupted or failed; inspect logs before resuming"
    try:
        iteration = 0
        while args.max_iterations is None or iteration < args.max_iterations:
            if (ROOT / ".ralph/STOP").exists():
                boundary = "stop requested"
                print("Stop requested; no further pass started.")
                return 130
            plan = load_plan()
            if next_task(plan) is None:
                boundary = "no locally actionable tasks"
                external = [t["id"] for t in plan["tasks"] if t["status"] == "external"]
                print("External acceptance remains: " + ", ".join(external) if external else "All ledger criteria recorded done; review release evidence.")
                if external:
                    return 2
                return 0 if verify(args.python, directory / "final-check")["success"] else 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            logs = directory / f"{stamp}-{iteration}"
            logs.mkdir()
            before = fingerprint()
            command = [args.codex, "exec", "--sandbox", "workspace-write", "--approve-for-me",
                       "--cd", str(ROOT), "--json", "--output-last-message", str(logs / "last-message.txt"), "-"]
            with (logs / "events.jsonl").open("w") as stdout, (logs / "stderr.txt").open("w") as stderr:
                code = agent_pass(command, prompt(plan), stdout, stderr, args.iteration_timeout)
            # The agent owns risk-based checks and records commands/evidence.
            # Retain its edits and logs even on failure; do not duplicate suites.
            cycles += 1
            iteration += 1
            save_checkpoint(load_plan(), cycles=cycles, reason="implementation pass ended; inspect recorded acceptance evidence", last_logs=logs)
            stalled = stalled + 1 if fingerprint() == before else 0
            print(f"Pass {iteration}: agent={code}, logs={logs}", flush=True)
            if code:
                boundary = f"agent exited {code}"
                return code
            if stalled >= 3:
                boundary = "three cycles without file progress"
                print("Three passes made no file progress; inspect logs and prerequisites before resuming.")
                return 2
            # Checkpoint reporting never implies product acceptance.
        boundary = "explicit invocation iteration budget reached"
        print("Requested iteration budget reached; checkpoint saved. Work is not declared complete.")
        return 2
    finally:
        try:
            save_checkpoint(load_plan(), cycles=cycles, reason=boundary, last_logs=logs)
        finally:
            lock.unlink(missing_ok=True)


def agent_pass(command, instruction, stdout, stderr, timeout):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, stdout=stdout, stderr=stderr,
                               start_new_session=os.name != "nt")
    try:
        process.stdin.write(instruction)
        process.stdin.close()
        started = time.monotonic()
        while process.poll() is None:
            if (ROOT / ".ralph/STOP").exists():
                return 130
            if time.monotonic() - started >= timeout:
                return 124
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
        return process.returncode
    finally:
        if process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "prompt", "verify", "run"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="optional invocation budget; default continues until stopped or no actionable tasks remain")
    parser.add_argument("--iteration-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if ((args.max_iterations is not None and args.max_iterations < 1) or not math.isfinite(args.iteration_timeout)
            or args.iteration_timeout <= 0):
        parser.error("runs require a positive iteration budget when specified and a positive finite timeout")
    if args.action == "run":
        return run(args)
    plan = load_plan()
    if args.action == "status":
        for task in plan["tasks"]:
            print(f"{task['status']:12} {task['id']}: {task['title']}")
    elif args.action == "prompt":
        print(prompt(plan))
    else:
        result = verify(args.python, ROOT / ".ralph/runs/manual-check")
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
