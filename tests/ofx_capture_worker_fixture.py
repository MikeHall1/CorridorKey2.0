"""Test-only consumed-pixel worker. No pretrained model or neural inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.session import InferenceSession
from corridorkey.worker_core import SharedEngine
from corridorkey.worker_protocol import job_from_wire
from corridorkey.worker_transport import serve_stdio


class CaptureFixture(SharedEngine):
    def render(self, model, job, config, writer, diagnostics):
        import numpy as np

        from corridorkey.diagnostics import json_value
        from corridorkey.output_paths import output_layout
        from corridorkey.source_witness import SourceWitness
        from corridorkey.video_io import (
            ExrSequenceWriter,
            probe_frame_dir_meta,
            read_image_rgb,
        )
        meta = probe_frame_dir_meta(job.input_path, start_frame=config.start_frame, num_frames=config.num_frames,
                                    shape_quantum=config.shape_quantum, fps=config.frame_dir_fps,
                                    model_scale=config.model_scale,
                                    exr_linear=config.frame_dir_linear)
        diagnostics.source = {key: value for key, value in meta.items() if key != 'frame_paths'}
        diagnostics.effective = json_value(config)
        witness = SourceWitness(meta, diagnostics.control)
        height, width = meta['native_hw']
        count = meta['num_frames']
        layout = output_layout(job.input_path, config)['outputs']
        foreground = Path(job.output_dir) / layout['fg']['path']
        matte = Path(job.output_dir) / layout['alpha']['path']
        fg = ExrSequenceWriter(str(foreground), meta['native_fps'], (width, height), channels=3, dtype=config.exr_dtype, frame_name=layout['fg']['frame_name'])
        alpha = ExrSequenceWriter(str(matte), meta['native_fps'], (width, height), channels=1, dtype=config.exr_dtype, frame_name=layout['alpha']['frame_name'])
        try:
            for index, path in enumerate(meta['frame_paths']):
                diagnostics.control.checkpoint('capture_fixture_pixels', index, count)
                pixels = read_image_rgb(path, expected_channels=3)
                # This toy operator consumes the exact source it witnesses;
                # identity RGB / constant coverage is not neural inference.
                witness.observe_path(path, pixels)
                fg.write(pixels)
                alpha.write(np.full((height, width, 1), .5, np.float32))
        finally:
            fg.release()
            alpha.release()
        diagnostics.source_witness = witness.finish(Path(job.output_dir))
        diagnostics.outputs = {'fg': str(foreground), 'alpha': str(matte)}
        diagnostics.output_contracts = {
            'fg': {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear', 'foreground_mode': 'test_fixture'},
            'alpha': {'format': 'exr', 'channels': 1},
        }
        diagnostics.frames_submitted = count
        diagnostics.execution = {'neural_inference': False, 'fixture': 'actual consumed pixels; identity foreground / constant alpha'}

    def run(self, request, cancellation, progress):
        if self.session is None:
            self.session = InferenceSession(loader=lambda: object(), runner=self.render, disposer=lambda _: None)
        return self.session.run(job_from_wire(request.payload()['job']), cancellation=cancellation, progress=progress).to_dict()


if __name__ == '__main__':
    raise SystemExit(serve_stdio(engine=CaptureFixture()))
