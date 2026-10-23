"""Local supervised worker for the shared engine; no alternate inference backend."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    serve = sub.add_parser('serve', help='serve framed requests on owned stdin/stdout pipes')
    serve.add_argument('--capacity', type=int, default=4)
    serve.add_argument('--weights')
    serve.add_argument('--asset-root', type=Path, help='Local asset tree; bundled implementation stays fixed.')
    run = sub.add_parser('run', help='run an inference batch through one supervised local child')
    run.add_argument('--batch', required=True, type=Path)
    run.add_argument('--report', required=True, type=Path)
    run.add_argument('--capacity', type=int, default=4)
    run.add_argument('--workspace', type=Path, help='use the existing recoverable EXR batch controller')
    run.add_argument('--max-cache-bytes', type=int, default=0)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--durable-foundations', action='store_true')
    run.add_argument('--durable-rvm', action='store_true')
    run.add_argument('--durable-hann', action='store_true')
    run.add_argument('--durable-stage2', action='store_true')
    run.add_argument('--durable-writer-prefix', action='store_true',
                     help='seal EXR prefixes; restore only with explicit --resume into fresh output')
    args = parser.parse_args(argv)
    if not 1 <= args.capacity <= 64:
        parser.error('capacity must be1..64')
    if args.action == 'serve':
        from .worker_core import SharedEngine
        from .worker_transport import serve_stdio
        return serve_stdio(capacity=args.capacity, engine=SharedEngine(weights=args.weights, asset_root=args.asset_root))
    if args.max_cache_bytes < 0:
        parser.error('max-cache-bytes must be nonnegative')
    if args.workspace is None and (args.resume or args.durable_foundations or args.durable_rvm or args.durable_hann or args.durable_stage2 or args.durable_writer_prefix or args.max_cache_bytes):
        parser.error('recovery options require --workspace')
    from .worker_client import run_worker_batch
    report = run_worker_batch(args.batch, args.report, capacity=args.capacity, workspace=args.workspace,
                              max_cache_bytes=args.max_cache_bytes, resume=args.resume,
                              durable_foundations=args.durable_foundations, durable_rvm=args.durable_rvm,
                              durable_hann=args.durable_hann, durable_stage2=args.durable_stage2,
                              durable_writer_prefix=args.durable_writer_prefix)
    if report.get('report_write', {}).get('status') == 'not_written':
        print(f"worker batch: {report['status']}; report NOT WRITTEN: {report.get('error', report['report_write'])}")
    else:
        print(f"worker batch: {report['status']}; report: {args.report}")
    return 0 if report['status'] == 'complete' else 130 if report['status'] == 'cancelled' else 1


if __name__ == '__main__':
    raise SystemExit(main())
