"""Test-only terminal CLI reports for desktop process/status ownership checks."""
import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

SCENARIOS = ('long_clip', 'batch50', 'repeated', 'concurrent_isolated', 'cancel',
             'interruption_recovery', 'memory_growth', 'encoder_failure', 'disk_full')


def report_for(manifest, output, mode):
    raw = Path(manifest).read_bytes()
    statuses = ['measured'] * len(SCENARIOS)
    if mode in ('unavailable', 'abnormal'):
        statuses[0] = 'not_verified'
    elif mode == 'failed':
        statuses[0] = 'failed'
    elif mode == 'cancelled':
        statuses = ['cancelled'] * len(SCENARIOS)
    cases = [{'id': f'fixture-{scenario}', 'scenario': scenario, 'required': True,
              'status': status, 'success': status == 'measured',
              'reason': 'Explicit test-only lifecycle fixture; no production stress or GPU acceptance',
              'error': 'Injected failure' if status == 'failed' else None,
              'artifact_dir': None, 'cleanup': {'verified': True, 'owners': []},
              'evidence': {'fixture': True}, 'duration_seconds': 0.001}
             for scenario, status in zip(SCENARIOS, statuses, strict=True)]
    summary = {'success': all(status == 'measured' for status in statuses), 'expected_cases': len(cases),
               'required_cases': len(cases), 'required_measured_cases': statuses.count('measured'),
               'cleanup_verified': True}
    summary.update({status + '_cases': statuses.count(status) for status in ('measured', 'not_verified', 'failed', 'cancelled')})
    return {'schema_version': 1, 'kind': 'production_stress',
            'status': 'failed' if mode == 'failed' else 'cancelled' if mode == 'cancelled' else 'complete',
            'summary': summary, 'cases': cases,
            'manifest': {'path': str(Path(manifest).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)},
            'limits': {}, 'artifact_dir': str(Path(output).parent.resolve()),
            'scope': 'Injected desktop lifecycle fixture only; not measured production stress or GPU evidence',
            'acceptance_verdict': None, 'error': 'Injected failure' if mode == 'failed' else None}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    data = json.loads(args.manifest.read_text())
    mode = data.get('fixture_mode', 'complete')
    if mode == 'escaped':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        child_code = """import os,signal,subprocess,sys,time
from pathlib import Path
signal.signal(signal.SIGINT,signal.SIG_IGN)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
heartbeat_code='import signal,sys,time;from pathlib import Path;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);p=Path(sys.argv[1]);i=0\\nwhile True:\\n p.write_text(str(i));i+=1;time.sleep(.01)'
child=subprocess.Popen([sys.executable,'-c',heartbeat_code,sys.argv[2]])
Path(sys.argv[1]).write_text(str(os.getpid())+','+str(child.pid))
while True: time.sleep(.01)
"""
        subprocess.Popen([sys.executable, '-c', child_code, data['pid_file'], data['heartbeat']], start_new_session=True)
        while not Path(data['pid_file']).exists() or not Path(data['heartbeat']).exists():
            time.sleep(.01)
        print('stress_escaped_ready', flush=True)
        while True:
            time.sleep(.01)
    if mode == 'wait':
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, interrupt)
        print('stress_fixture_ready', flush=True)
        try:
            while True:
                time.sleep(.02)
        except KeyboardInterrupt:
            mode = 'cancelled'
    report = report_for(args.manifest, args.output, mode)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False)
    print('Explicit injected lifecycle report; no production acceptance', flush=True)
    return {'complete': 0, 'unavailable': 2, 'abnormal': 79, 'failed': 1, 'cancelled': 130}[mode]


if __name__ == '__main__':
    sys.exit(main())
