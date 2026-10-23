"""Explicit CPU-only profiled shared-runtime worker; never a production backend."""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.worker_transport import serve_stdio
from tests.stage2_worker_fixture import Stage2FixtureEngine


class OutputTelemetryFixtureEngine(Stage2FixtureEngine):
    def render(self, model, job, config, writer, diagnostics):
        # The production profile path requires CUDA. This tiny CPU test uses
        # actual host/writer clocks and explicitly unavailable GPU measurements.
        import torch

        try:
            with patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=None), \
                 patch.object(torch.cuda, 'max_memory_reserved', return_value=None), \
                 patch.object(torch.cuda, 'get_device_name', return_value='CPU fixture placeholder'):
                super().render(model, job, config, writer, diagnostics)
        finally:
            if diagnostics.profile_stages:
                diagnostics.resources = {
                    'peak_allocated_vram_bytes': None, 'peak_reserved_vram_bytes': None,
                    'gpu': None, 'cuda_version': None,
                    'scope': 'NOT VERIFIED: explicit CPU fixture; no CUDA allocation measurement',
                }
                for stage in diagnostics.stages:
                    stage['method'] = 'host_wall_explicit_cpu_fixture'
            if isinstance(diagnostics.execution, dict):
                diagnostics.execution['profile_scope'] = 'Explicit CPU fixture host clocks; no CUDA timing or GPU acceptance'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=4)
    parser.add_argument('--fixture-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    raise SystemExit(serve_stdio(capacity=args.capacity, engine=OutputTelemetryFixtureEngine(args.fixture_root)))
