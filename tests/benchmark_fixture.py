"""Trusted test-only CLI fixture; no public production fixture mode."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
data = json.loads(args.manifest.read_text())
if data.get('mode') == 'descendant':
    child = subprocess.Popen([sys.executable, '-c', '''import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGINT,signal.SIG_IGN)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
index=0
while True:
    Path(sys.argv[2]).write_text(str(index))
    index+=1
    time.sleep(.01)
''', data['marker'], data['heartbeat']])
    while not Path(data['marker']).exists():
        time.sleep(.01)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print('fixture_ready', flush=True)
    while True:
        time.sleep(.05)
if data.get('mode') == 'crash':
    os._exit(78)
if data.get('mode') == 'wait':
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print('fixture_ready', flush=True)
    while True:
        time.sleep(.05)
if data.get('mode') == 'logs':
    os.write(1, b'x' * (1024 * 1024))
report = {'schema_version': 1, 'kind': 'cpu_correctness', 'status': 'complete',
          'summary': {'success': True, 'median_seconds': None}, 'runs': [],
          'inference': {'status': 'unmeasured', 'fps': None}, 'test_fixture': True}
args.output.write_text(json.dumps(report))
