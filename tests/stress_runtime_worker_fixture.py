"""Test-only shared CPU runtime with an out-of-input gated readable EXR prefix."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.worker_transport import serve_stdio
from tests.output_telemetry_worker_fixture import OutputTelemetryFixtureEngine


class StressFixtureEngine(OutputTelemetryFixtureEngine):
    def __init__(self, fixture_root, *, gate_after_readable=False):
        super().__init__(fixture_root)
        self.gate_after_readable = gate_after_readable

    def render(self, model, job, config, writer, diagnostics):
        import cv2
        original = diagnostics.control.progress

        def progress(event):
            if original is not None:
                original(event)
            if self.gate_after_readable and event.stage == 'output_frames' and event.completed >= 1:
                deadline = time.monotonic() + 5
                while not any(cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None
                              for path in Path(job.output_dir).rglob('*.exr')):
                    diagnostics.control.check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Injected fixture produced no readable EXR before gate')
                    time.sleep(.005)
                print('stress_prefix_readable_fixture', flush=True)
                while True:
                    diagnostics.control.check()
                    time.sleep(.005)

        diagnostics.control.progress = progress
        try:
            super().render(model, job, config, writer, diagnostics)
        finally:
            diagnostics.control.progress = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=1)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--gate-after-readable', action='store_true')
    args = parser.parse_args()
    raise SystemExit(serve_stdio(capacity=args.capacity, engine=StressFixtureEngine(args.fixture_root,
                                 gate_after_readable=args.gate_after_readable)))
