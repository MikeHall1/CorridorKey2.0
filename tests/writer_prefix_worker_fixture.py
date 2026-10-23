"""Explicit CPU shared-runtime worker; gate only after durable EXR-prefix seal."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.worker_transport import serve_stdio
from tests.output_telemetry_worker_fixture import OutputTelemetryFixtureEngine


class WriterPrefixFixtureEngine(OutputTelemetryFixtureEngine):
    def render(self, model, job, config, writer, diagnostics):
        instructions_path = Path(job.input_path).parent / '.writer-prefix-worker.json'
        instructions = json.loads(instructions_path.read_text()) if instructions_path.is_file() else {}
        original_progress = diagnostics.control.progress

        def progress(event):
            if original_progress is not None:
                original_progress(event)
            gate = Path(instructions['gate']) if instructions.get('gate') else None
            if (event.stage == 'writer_prefix_frames' and event.completed >= instructions.get('threshold', 2)
                    and gate is not None and not gate.exists()):
                adapter = diagnostics.durable_writer_prefix
                if adapter is None or adapter.snapshot()['stored_frames'] < instructions.get('threshold', 2):
                    raise RuntimeError('Fixture gate requires actual sealed writer-prefix receipt')
                if instructions.get('ready'):
                    Path(instructions['ready']).write_text(json.dumps(adapter.snapshot(), sort_keys=True))
                if instructions.get('hard_exit'):
                    os._exit(79)
                while not gate.exists():
                    diagnostics.control.check()
                    time.sleep(.005)

        diagnostics.control.progress = progress
        try:
            super().render(model, job, config, writer, diagnostics)
        finally:
            diagnostics.control.progress = original_progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=1)
    parser.add_argument('--fixture-root', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(serve_stdio(capacity=args.capacity, engine=WriterPrefixFixtureEngine(args.fixture_root)))
