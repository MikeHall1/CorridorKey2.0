"""Python-only injection entrypoint for the real stress-suite parser/driver.

The user manifest cannot select a command or a backend. This test-only CLI
supplies deterministic tiny CPU workers, preserving the production entrypoint.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.stress_suite import main
from corridorkey.worker_client import WorkerClient

CREATIONS = {}


def worker_factory(*, root, scenario, capacity, weights=None, asset_root=None):
    root = Path(root)
    count = CREATIONS.get(str(root), 0)
    CREATIONS[str(root)] = count + 1
    assets = root / 'fixture-assets'
    command = [sys.executable, str(Path(__file__).with_name('stress_runtime_worker_fixture.py')),
               '--fixture-root', str(assets), '--capacity', str(capacity)]
    if count == 0 and scenario in ('cancel', 'interruption_recovery'):
        command.append('--gate-after-readable')
    return WorkerClient(capacity=capacity, command=command)


if __name__ == '__main__':
    raise SystemExit(main(worker_factory=worker_factory))
