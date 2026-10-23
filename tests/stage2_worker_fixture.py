"""Explicit CPU-only worker fixture exercising the actual shared runtime."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.worker_transport import serve_stdio
from tests.worker_fixture import FixtureEngine


class Stage2FixtureEngine(FixtureEngine):
    def __init__(self, fixture_root):
        super().__init__()
        self.fixture_root = fixture_root

    def load(self):
        from tests.test_durable_stage2 import make_cpu_model

        self.loads += 1
        return make_cpu_model()

    def render(self, model, job, config, writer, diagnostics):
        from corridorkey.runtime import run_inference
        from tests.test_durable_stage2 import cpu_runtime_context

        control_file = Path(job.input_path).parent / '.stage2-worker.json'
        instructions = json.loads(control_file.read_text()) if control_file.is_file() else {}
        gate = Path(instructions['gate']) if instructions.get('gate') else None
        ready = Path(instructions['ready']) if instructions.get('ready') else None
        original_progress = diagnostics.control.progress

        def progress(event):
            if original_progress is not None:
                original_progress(event)
            if event.stage == 'output_frames' and event.completed >= 2 and gate is not None and not gate.exists():
                if ready is not None:
                    ready.write_text('Sealed predictions and submitted first output batch; waiting for explicit fixture release.')
                if instructions.get('hard_exit'):
                    import cv2

                    deadline = time.monotonic() + 5
                    while not any(cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None
                                  for path in Path(job.output_dir).rglob('*.exr')):
                        diagnostics.control.check()
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Fixture writer did not produce a readable EXR before hard exit')
                        time.sleep(0.01)
                    os._exit(79)
                while not gate.exists():
                    diagnostics.control.check()
                    time.sleep(0.01)

        diagnostics.control.progress = progress
        try:
            with cpu_runtime_context():
                run_inference(model, job.input_path, job.output_dir, config, writer,
                              initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
                              diagnostics=diagnostics)
        finally:
            if isinstance(diagnostics.execution, dict):
                diagnostics.execution.update(effective_device='cpu (explicit test fixture)',
                                             effective_autocast='disabled by explicit CPU test context',
                                             inference_scope='deterministic tiny model; no pretrained/GPU acceptance')
            diagnostics.control.progress = original_progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=4)
    parser.add_argument('--fixture-root', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(serve_stdio(capacity=args.capacity, engine=Stage2FixtureEngine(args.fixture_root)))
