"""Explicit launch/pipe smoke check; not GPU, GUI or model acceptance."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from .worker_client import WorkerClient


def smoke(prefix: list[str], parent: Path) -> Path:
    if not parent.is_dir():
        raise ValueError('Choose an existing report parent directory.')
    directory = Path(tempfile.mkdtemp(prefix='desktop-smoke-', dir=parent))
    report: dict = {'status': 'running', 'command_prefix': prefix,
                    'scope': 'Backend child launch, real worker handshake/status/clean shutdown only. GUI, CUDA inference, models, render/benchmark/recovery and Windows containment NOT VERIFIED.'}
    client = None
    try:
        backend = subprocess.run([*prefix, '--corridorkey-module', 'corridorkey.backend_probe'],
                                 capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(backend.stdout)
        if not isinstance(data, dict) or type(data.get('cuda_available')) is not bool:
            raise ValueError('Invalid backend probe record.')
        report['backend'] = data
        client = WorkerClient(command=[*prefix, '--corridorkey-module', 'corridorkey.worker', 'serve'])
        identifier = client.send('status', {})
        event = client.next_event(timeout=10)
        if event['kind'] != 'status' or event['request_id'] != identifier:
            raise ValueError('Worker status response differs from the actual request.')
        if event['payload']['active'] is not None or event['payload']['pending']:
            raise ValueError('Smoke worker unexpectedly contains processing jobs.')
        report['worker_status'] = event
        report['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as exc:
        logging.getLogger(__name__).exception('Desktop smoke failed')
        report.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
    finally:
        if client is not None:
            report['clean_shutdown'] = client.close()
            report['cleanup_error'] = client.cleanup_error or client.failure
            if not report['clean_shutdown']:
                report['status'] = 'failed'
        with (directory / 'report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    return directory / 'report.json'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', action='store_true', help='Check the source dispatcher; never native-bundle evidence.')
    source.add_argument('--executable', type=Path, help='Explicit trusted local CorridorKey bundle executable.')
    parser.add_argument('--output-parent', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.executable is not None and not args.executable.is_file():
        parser.error('Select an existing bundle executable.')
    prefix = [str(args.executable.resolve())] if args.executable else [sys.executable, '-m', 'corridorkey.app_process']
    report = smoke(prefix, args.output_parent.resolve())
    print(report)
    status = json.loads(report.read_text(encoding='utf-8'))['status']
    return 0 if status == 'passed' else 130 if status == 'cancelled' else 1


if __name__ == '__main__':
    raise SystemExit(main())
