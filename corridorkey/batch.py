"""Run a local JSON job batch through one persistent model session."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import InferenceConfig, WriterConfig
from .diagnostics import write_json_atomic
from .session import InferenceSession, JobRequest
from .control import CancelledError
from .local_json import read_object


def load_jobs(path, *, regular_file=False):
    path = Path(path).resolve()
    try:
        manifest, _ = read_object(path, max_bytes=4 * 1024 * 1024, regular_file=regular_file)
    except TypeError as exc:
        raise ValueError('Batch manifest must be an object.') from exc
    return jobs_from_data(manifest, path)


def jobs_from_data(manifest, path):
    """Validate captured batch JSON using the original manifest path for relatives."""
    path = Path(path).resolve()
    if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
            or manifest.get("kind") != "inference_batch"):
        raise ValueError("batch manifest requires schema_version=1, kind=inference_batch")
    if set(manifest) - {"schema_version", "kind", "weights", "jobs"}:
        raise ValueError("unknown batch manifest field")
    if not isinstance(manifest.get("jobs"), list) or not manifest["jobs"]:
        raise ValueError("jobs must be a nonempty list")

    def local(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("paths must be nonempty strings")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    jobs = []
    outputs = set()
    for entry in manifest["jobs"]:
        if not isinstance(entry, dict) or set(entry) - {
                "input", "output_dir", "settings", "writer", "hint_first_frame", "hint_video", "profile_stages"}:
            raise ValueError("unknown or invalid job field")
        if "input" not in entry or "output_dir" not in entry:
            raise ValueError("each job requires input and output_dir")
        settings, writer = entry.get("settings", {}), entry.get("writer", {})
        if not isinstance(settings, dict) or not isinstance(writer, dict):
            raise ValueError("settings and writer must be objects")
        settings = dict(settings)
        if "outputs" in settings:
            values = settings["outputs"]
            if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
                raise ValueError("outputs must be a nonempty array of names")
            aliases = {"raw_fg": "fg", "foreground": "fg"}
            settings["outputs"] = {aliases.get(v.strip().lower(), v.strip().lower()) for v in values}
        if "temp_dir" in settings:
            settings["temp_dir"] = local(settings["temp_dir"])
        if settings.get("corrections_path") is not None:
            settings["corrections_path"] = local(settings["corrections_path"])
        try:
            config = InferenceConfig(**settings)
            writer_config = WriterConfig(**writer)
        except TypeError as exc:
            raise ValueError(f"invalid settings: {exc}") from exc
        profile = entry.get("profile_stages", False)
        if type(profile) is not bool:
            raise ValueError("profile_stages must be a boolean")
        job = JobRequest.create(local(entry["input"]), local(entry["output_dir"]), config, writer_config,
                                initial_hint_path=local(entry["hint_first_frame"]) if entry.get("hint_first_frame") is not None else None,
                                hint_video_path=local(entry["hint_video"]) if entry.get("hint_video") is not None else None,
                                profile_stages=profile)
        job.validate_source(config)
        if job.output_dir in outputs:
            raise ValueError("each batch job must have a distinct output_dir")
        outputs.add(job.output_dir)
        jobs.append(job)
    weights = local(manifest["weights"]) if manifest.get("weights") is not None else None
    return jobs, weights


def run_batch(jobs, *, weights=None, report_path=None, session_factory=InferenceSession,
              cancellation=None, progress=None, manifest_path=None):
    from .output_paths import prepare_report_destination, validate_batch_output_paths
    jobs = tuple(jobs)
    report_context = {'manifest_path': manifest_path, 'weights': weights}
    validate_batch_output_paths(jobs, **report_context)
    admit_report = None
    if report_path is not None:
        admit_report = prepare_report_destination(report_path, jobs, **report_context)
    report = {"schema_version": 1, "kind": "inference_batch", "status": "running", "jobs": [],
              "note": "Single process, persistent model; not a cold-process benchmark. No source media uploaded."}
    report_admitted = False

    def persist():
        if report_path is not None and report_admitted:
            assert admit_report is not None
            admit_report()
            write_json_atomic(report_path, report)

    try:
        with session_factory(weights=weights) as session:
            report_context.update(package_root=session.package_root, asset_root=session.asset_root,
                                  weights=session.weights)
            validate_batch_output_paths(jobs, **report_context)
            if report_path is not None:
                admit_report = prepare_report_destination(report_path, jobs, **report_context)
                report_admitted = True
                persist()
            for job in jobs:
                # Stop on the first failure. Completed earlier jobs remain in the report.
                try:
                    controls = {}
                    if cancellation is not None:
                        controls["cancellation"] = cancellation
                    if progress is not None:
                        controls["progress"] = progress
                    result = session.run(job, **controls)
                except BaseException:
                    if session.last_result is not None:
                        report["jobs"].append(session.last_result.to_dict())
                    raise
                report["jobs"].append(result.to_dict())
                persist()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "cancelled" if isinstance(exc, (KeyboardInterrupt, CancelledError)) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        from .lifecycle import cleanup_all
        cleanup_all([persist], primary_error=sys.exception())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--report", required=True, help="Local JSON job report, updated after every job.")
    args = parser.parse_args(argv)
    try:
        jobs, weights = load_jobs(args.manifest)
        run_batch(jobs, weights=weights, report_path=args.report, manifest_path=args.manifest)
    except (KeyboardInterrupt, CancelledError):
        return 130
    except Exception as exc:
        print(f"Batch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
