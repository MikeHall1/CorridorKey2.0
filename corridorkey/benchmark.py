"""Headless local CPU correctness and CUDA inference benchmark runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable

from .app_process import module_command

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    from .identities import file_sha256
    return file_sha256(path)


def command(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def repository_identity():
    status = command("git", "status", "--porcelain=v1", "-z")
    tracked = command("git", "ls-files", "-z").split(b"\0")
    untracked = command("git", "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    digest = hashlib.sha256()
    # Hash actual files, including untracked implementation changes and deletions.
    for raw in sorted(set(tracked + untracked) - {b""}):
        path = ROOT / raw.decode()
        digest.update(raw + b"\0")
        digest.update((sha256(path) if path.is_file() else "missing").encode())
    return {"commit": command("git", "rev-parse", "HEAD").decode().strip(),
            "dirty": bool(status), "status": status.decode().replace("\0", "\n"),
            "working_tree_sha256": digest.hexdigest()}


def environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "transformers",
                 "timm", "safetensors", "scipy", "pillow", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware = {"cpu": platform.processor() or platform.machine(), "ram_bytes": None,
                "gpu": None, "cuda": None, "driver": None, "power_settings": None}
    if sys.platform == "darwin":
        for field, key in (("cpu", "machdep.cpu.brand_string"), ("ram_bytes", "hw.memsize")):
            try:
                value = command("sysctl", "-n", key).decode().strip()
                hardware[field] = int(value) if field == "ram_bytes" else value
            except (RuntimeError, OSError, ValueError) as exc:
                hardware[field + "_unavailable"] = str(exc)
    elif hasattr(__import__("os"), "sysconf"):
        import os
        hardware["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {"python": sys.version, "os": platform.platform(), "hardware": hardware,
            "dependencies": packages, "ffmpeg": shutil.which("ffmpeg"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "storage_free_bytes": shutil.disk_usage(ROOT).free}


def load_manifest(path):
    from .worker_protocol import MAX_FRAME_BYTES, decode_frame
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_FRAME_BYTES + 1)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError('Benchmark manifest exceeds the 4MiB size limit.')
    try:
        manifest = decode_frame(raw.rstrip(b'\n') + b'\n')
    except TypeError as exc:
        raise ValueError('Benchmark manifest must be an object.') from exc
    if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
            or manifest.get("kind") not in {"cpu_correctness", "inference", "decode_reuse"}):
        raise ValueError("supported manifest: schema_version=1, kind=cpu_correctness, inference or decode_reuse")
    repeats = manifest.get("repetitions", 3)
    if type(repeats) is not int or repeats < 3:
        raise ValueError("at least three measured repetitions are required")
    if manifest['kind'] == 'cpu_correctness':
        if set(manifest) - {'schema_version', 'kind', 'repetitions', 'description', 'timeout_seconds'}:
            raise ValueError('unknown CPU benchmark manifest field')
        timeout = manifest.get('timeout_seconds', 600)
        if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
            raise ValueError('CPU timeout_seconds must be finite, positive and at most 3600 seconds')
    if manifest['kind'] == 'decode_reuse':
        from .benchmark_decode_reuse import validate_manifest

        validate_manifest(manifest)
    if manifest["kind"] == "inference":
        if set(manifest) - {"schema_version", "kind", "batch_manifest", "model_state", "repetitions", "description", "quality_manifests"}:
            raise ValueError("unknown inference benchmark manifest field")
        if not isinstance(manifest.get("batch_manifest"), str) or not manifest["batch_manifest"]:
            raise ValueError("inference benchmark requires batch_manifest path")
        if manifest.get("model_state") not in {"cold_process", "warm_session"}:
            raise ValueError("model_state must be cold_process or warm_session")
        if "quality_manifests" in manifest:
            quality = manifest["quality_manifests"]
            if (not isinstance(quality, list) or not quality or
                    any(p is not None and (not isinstance(p, str) or not p.strip()) for p in quality)):
                raise ValueError("quality_manifests must be a nonempty list of paths/nulls, one per batch job")
    return manifest


def run(manifest_path, *, progress: Callable[[str, int, int], None] | None = None):
    manifest = load_manifest(manifest_path)
    if manifest["kind"] == "inference":
        return run_inference_benchmark(manifest_path, manifest, progress=progress)
    if manifest['kind'] == 'decode_reuse':
        return run_decode_reuse_benchmark(manifest_path, manifest, progress=progress)
    timeout = manifest.get('timeout_seconds', 600)
    report = {"schema_version": 1, "kind": "cpu_correctness",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest": manifest, "manifest_sha256": sha256(manifest_path),
              "repository": repository_identity(), "environment": environment(),
              "cache": {"model": "not_loaded", "filesystem": "uncontrolled"},
              "cpu_timeout": {"requested_seconds": manifest.get('timeout_seconds'),
                              "effective_seconds": timeout, "scope": "each CPU test subprocess"},
              "weights": {str(p.relative_to(ROOT)): sha256(p)
                          for folder in (ROOT / "weights", ROOT / "vfms")
                          for p in sorted(folder.rglob("*"))
                          if p.is_file() and p.name != ".gitkeep"},
              "input": None, "hints": None, "corrections": None,
              "inference": {"status": "unmeasured", "fps": None,
                            "model_only_fps": None, "quality": None},
              "warnings": ["CPU test-process duration is not inference throughput.",
                           "GPU, CUDA, driver and power telemetry are not collected by this CPU runner."],
              "runs": []}
    report['status'] = 'running'
    try:
        for index in range(manifest.get("repetitions", 3)):
            if progress is not None:
                progress('cpu_repetition', index, manifest.get('repetitions', 3))
            start = time.perf_counter()
            result = subprocess.run(
                module_command('unittest', 'discover', '-s', 'tests', '-v'),
                cwd=ROOT, capture_output=True, text=True, timeout=timeout)
            report["runs"].append({"index": index, "elapsed_seconds": time.perf_counter() - start,
                                   "returncode": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        report['status'] = 'complete' if all(r['returncode'] == 0 for r in report['runs']) else 'failed'
    except (KeyboardInterrupt, subprocess.TimeoutExpired, ValueError) as exc:
        report['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    times = [r["elapsed_seconds"] for r in report["runs"]]
    report["summary"] = {"success": report['status'] == 'complete',
                         "median_seconds": statistics.median(times) if report['status'] == 'complete' else None,
                         "min_seconds": min(times) if report['status'] == 'complete' else None,
                         "max_seconds": max(times) if report['status'] == 'complete' else None}
    return report


def run_decode_reuse_benchmark(manifest_path, manifest, *, progress=None):
    """Research reader experiment through the same CLI/developer runner."""
    from .benchmark_decode_reuse import run_decode_reuse, validate_manifest
    from .local_json import read_object

    path = Path(manifest_path).resolve()
    captured, raw = read_object(path, max_bytes=4 * 1024 * 1024, regular_file=True)
    if captured != manifest:
        raise ValueError('Decode investigation manifest changed before execution.')
    options = validate_manifest(captured)
    provenance = {'manifest': captured, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                  'repository': repository_identity(), 'environment': environment()}
    parent = (path.parent / options['artifact_parent']).resolve()
    arguments = {key: options[key] for key in
                 ('repetitions', 'cache_bytes', 'frame_count', 'shape_hw', 'chunk', 'stride', 'quick_interval')
                 if key in options}
    if 'shape_hw' in arguments:
        arguments['shape_hw'] = tuple(arguments['shape_hw'])
    result = run_decode_reuse(artifact_parent=parent, progress=progress, **arguments)
    result.update(provenance)
    return result


def asset_identity(path):
    """Content identity computed outside measured runs; footage never leaves disk."""
    from .identities import source_identity
    return source_identity(path)


def correction_identity(path, source):
    from .corrections import load_corrections
    document = load_corrections(path, source=source)
    data = document.to_dict()
    return {"path": str(path), "manifest_sha256": sha256(path), "identity": document.identity,
            "revision": data["revision"], "source": data["source"],
            "assets": [{"path": str(document.base_dir / asset["path"]), "sha256": asset["sha256"]}
                       for asset in document.asset_bindings()]}


def run_inference_benchmark(manifest_path, manifest, *, progress: Callable[[str, int, int], None] | None = None):
    from dataclasses import replace
    from .batch import jobs_from_data
    from .diagnostics import write_json_atomic
    from .local_json import read_object
    from .session import InferenceSession
    import tempfile

    manifest_path = Path(manifest_path).resolve()
    batch_path = (manifest_path.parent / manifest["batch_manifest"]).resolve()
    report = {"schema_version": 1, "kind": "inference", "manifest": manifest,
              "manifest_sha256": sha256(manifest_path), "batch_manifest_sha256": None,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "repository": repository_identity(), "environment": environment(),
              "model_state": manifest["model_state"], "runs": [], "warmup": [],
              "cache": {"filesystem": "uncontrolled", "intermediate": "per_job_temporary_only",
                        "output": "fresh_destination_per_run"},
              "inference": {"status": "unmeasured", "model_only_fps": None, "quality": None},
              "warnings": ["Alpha/colour reference quality requires optional quality_manifests; model-only throughput is unimplemented.",
                           "Cold-process duration includes interpreter startup, imports, model loading and shutdown.",
                           "Warm-session measurements follow one unmeasured full job per input.",
                           "No automated alternating baseline/candidate performance run or power/process-tree resource sampler is implemented."],
              "summary": {"success": False}}
    try:
        batch_data, batch_raw = read_object(batch_path, max_bytes=4 * 1024 * 1024)
        report["batch_manifest_sha256"] = hashlib.sha256(batch_raw).hexdigest()
        jobs, weights = jobs_from_data(batch_data, batch_path)
        quality_templates = None
        if "quality_manifests" in manifest:
            from .benchmark_quality import prepare_quality_templates
            quality_templates = prepare_quality_templates(manifest["quality_manifests"], jobs,
                                                          manifest_dir=manifest_path.parent)
        if any(job.profile_stages for job in jobs):
            raise ValueError("throughput benchmarks require profile_stages=false; run detailed diagnostics separately")
        report["assets"] = {path: asset_identity(path) for job in jobs
                            for path in (job.input_path, job.initial_hint_path, job.hint_video_path)
                            if path is not None}
        report["corrections"] = [correction_identity(config.corrections_path, job.input_path)
                                  if config.corrections_path is not None else None
                                  for job in jobs for config, _ in (job.configurations(),)]
        # A cold-process supervisor must not allocate a competing CUDA context.
        # Actual selected-device metadata comes from the processing job records.
        if manifest["model_state"] == "warm_session":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("inference benchmark requires a CUDA GPU and local runtime/foundation weights")
        try:
            report["environment"]["hardware"]["driver"] = command(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").decode().strip()
        except (RuntimeError, OSError) as exc:
            report["environment"]["hardware"]["driver_unavailable"] = str(exc)
        # Resolve the same asset defaults/legacy aliases as the processing core.
        from .runtime import RuntimePaths
        paths = RuntimePaths.from_package(ROOT, weights)
        paths.validate()
        weight_files = [paths.weights, paths.foundation_stats, paths.rvm_ckpt, paths.moge_checkpoint]
        weight_files.extend(sorted(p for p in paths.cradio_dir.rglob("*") if p.is_file()))
        report["weights"] = {str(p): sha256(p) for p in weight_files}

        output_root = ROOT / "benchmark-results"
        output_root.mkdir(exist_ok=True)
        # Keep generated exports for inspection. Never overwrite manifest outputs.
        run_root = Path(tempfile.mkdtemp(prefix="inference-", dir=output_root))
        report["artifact_dir"] = str(run_root)

        def run_jobs(session, repeat, records):
            for index, job in enumerate(jobs):
                job = replace(job, output_dir=str(run_root / str(repeat) / str(index)))
                try:
                    result = session.run(job)
                except BaseException:
                    if session.last_result is not None:
                        records.append(session.last_result.to_dict())
                    raise
                records.append(result.to_dict())

        if manifest["model_state"] == "warm_session":
            with InferenceSession(weights=weights) as session:
                if progress is not None:
                    progress('warmup', 0, len(jobs))
                run_jobs(session, "warmup", report["warmup"])
                for repeat in range(manifest.get("repetitions", 3)):
                    if progress is not None:
                        progress('warm_repetition', repeat, manifest.get('repetitions', 3))
                    measured = {"index": repeat, "jobs": []}
                    report["runs"].append(measured)
                    started = time.perf_counter()
                    try:
                        run_jobs(session, repeat, measured["jobs"])
                    finally:
                        measured["elapsed_seconds"] = time.perf_counter() - started
        else:
            for repeat in range(manifest.get("repetitions", 3)):
                if progress is not None:
                    progress('cold_repetition', repeat, manifest.get('repetitions', 3))
                # The child batch uses the same session/core; this is a fresh
                # interpreter per repetition, not merely reloading a model.
                # Only execute the validated preflight snapshots. Rereading the
                # original batch could alter settings or append unisolated jobs.
                batch = {"schema_version": 1, "kind": "inference_batch", "weights": str(paths.weights),
                         "jobs": [{"input": job.input_path, "output_dir": str(run_root / str(repeat) / str(index)),
                                   "hint_first_frame": job.initial_hint_path, "hint_video": job.hint_video_path,
                                   "settings": json.loads(job.config_json), "writer": json.loads(job.writer_json),
                                   "profile_stages": job.profile_stages} for index, job in enumerate(jobs)]}
                child_manifest = run_root / f"batch-{repeat}.json"
                child_report = run_root / f"report-{repeat}.json"
                write_json_atomic(child_manifest, batch)
                started = time.perf_counter()
                result = subprocess.run(module_command('corridorkey.batch', str(child_manifest),
                                         '--report', str(child_report)), cwd=ROOT,
                                        capture_output=True, text=True)
                measured = {"index": repeat, "elapsed_seconds": time.perf_counter() - started,
                            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                            "jobs": json.loads(child_report.read_text())["jobs"] if child_report.exists() else []}
                report["runs"].append(measured)
                if result.returncode:
                    raise RuntimeError(f"cold-process repetition {repeat} failed; inspect run logs")
        execution = report["runs"][0]["jobs"][0]["diagnostics"].get("execution") or {}
        for field in ("gpu", "vram_bytes", "cuda_version", "device_index"):
            if field in execution:
                report["environment"]["hardware"][field] = execution[field]
        times = [r["elapsed_seconds"] for r in report["runs"]]
        report["summary"] = {"success": True, "median_seconds": statistics.median(times),
                             "min_seconds": min(times), "max_seconds": max(times)}
        report["inference"]["status"] = "measured"
        if quality_templates is not None:
            from .benchmark_quality import evaluate_run_quality
            report["inference"]["quality"] = evaluate_run_quality(quality_templates, report["runs"])
            report["summary"]["success"] = report["inference"]["quality"]["success"]
    except BaseException as exc:
        report["summary"]["success"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["inference"]["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "unavailable_or_failed"
    return report


def _report_admission(manifest_path, output):
    """Snapshot report dependencies without decoding media or loading a model.

    Quality loaders remain the schema authority. Their metadata is pre-admitted
    at 4MiB and its returned digest must match the bounded captured bytes.
    Revalidation is a cooperating-process fence, not a filesystem reservation.
    """
    from .batch import jobs_from_data
    from .local_json import read_object
    from .output_paths import validate_report_destination

    manifest_lexical = Path(manifest_path).absolute()
    manifest_path = manifest_lexical.resolve()
    manifest, raw = read_object(manifest_path, max_bytes=4 * 1024**2, regular_file=True)
    if manifest != load_manifest(manifest_path):
        raise ValueError('Benchmark manifest changed during report admission')
    metadata = {manifest_path: hashlib.sha256(raw).hexdigest()}
    bindings = {manifest_lexical: manifest_path}
    protected = []
    jobs, weights = [], None
    if manifest['kind'] == 'inference':
        batch_lexical = manifest_path.parent / manifest['batch_manifest']
        batch = batch_lexical.resolve()
        bindings[batch_lexical] = batch
        data, raw = read_object(batch, max_bytes=4 * 1024**2, regular_file=True)
        metadata[batch] = hashlib.sha256(raw).hexdigest()
        jobs, weights = jobs_from_data(data, batch)
        quality_paths = manifest.get('quality_manifests')
        if quality_paths is not None:
            from .evaluation_colour_media import ROLES as COLOUR_ROLES
            from .evaluation_colour_media import load_colour_manifest
            from .evaluation_media import ROLES as ALPHA_ROLES
            from .evaluation_media import load_sequence_manifest
            from .evaluation_temporal_media import ALPHA_ROLES as TEMPORAL_ALPHA
            from .evaluation_temporal_media import (
                TRANSITION_ROLES,
                load_temporal_manifest,
            )

            if len(quality_paths) != len(jobs):
                raise ValueError('quality_manifests must have one path or null per batch job')
            loaders = {'alpha_sequence_evaluation': (load_sequence_manifest, ALPHA_ROLES),
                       'colour_sequence_evaluation': (load_colour_manifest, COLOUR_ROLES),
                       'temporal_sequence_evaluation': (load_temporal_manifest, TEMPORAL_ALPHA | TRANSITION_ROLES)}
            for value in quality_paths:
                if value is None:
                    continue
                lexical = manifest_path.parent / value
                path = lexical.resolve()
                bindings[lexical] = path
                header, raw = read_object(path, max_bytes=4 * 1024**2, regular_file=True)
                if header.get('kind') not in loaders:
                    raise ValueError('Unsupported benchmark quality manifest kind')
                loader, roles = loaders[header['kind']]
                quality, digest = loader(path)
                if digest != hashlib.sha256(raw).hexdigest():
                    raise ValueError('Quality manifest changed during report admission')
                metadata[path] = digest
                if 'source' in quality:
                    protected.append(quality['source'])
                protected.extend(frame[role] for frame in quality['frames'] for role in roles if role in frame)

    def admit(report=None):
        if any(lexical.resolve() != resolved for lexical, resolved in bindings.items()):
            raise ValueError('Benchmark dependency path changed before report publication')
        for path, digest in metadata.items():
            _, current = read_object(path, max_bytes=4 * 1024**2, regular_file=True)
            if hashlib.sha256(current).hexdigest() != digest:
                raise ValueError('Benchmark dependency metadata changed before report publication')
        directories = ()
        if report is not None and report.get('artifact_dir') is not None:
            artifact = report['artifact_dir']
            if not isinstance(artifact, str) or not artifact.strip():
                raise ValueError('Benchmark artifact directory is malformed')
            directories = (artifact,)
        return validate_report_destination(output, jobs, manifest_path=manifest_path, weights=weights,
                                           package_root=ROOT, protected_paths=(*metadata, *protected),
                                           protected_directories=directories)

    return admit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "benchmarks/suites/cpu.json"))
    parser.add_argument("--output", default=str(ROOT / "benchmark-results/cpu.json"))
    args = parser.parse_args()
    output = Path(args.output)
    try:
        admit = _report_admission(args.manifest, output)
        admit()
        report = run(args.manifest, progress=lambda stage, completed, total: print(f'{stage}: {completed}/{total}', flush=True))
        admit(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        from .diagnostics import write_json_atomic
        write_json_atomic(output, report)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"{report['kind']}: {'passed' if report['summary']['success'] else 'failed'}; report: {output}")
    if report.get('status') == 'cancelled' or report.get("inference", {}).get("status") == "cancelled":
        return 130
    return 0 if report["summary"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
