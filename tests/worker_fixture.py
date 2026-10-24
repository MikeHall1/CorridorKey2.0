"""Explicit test-only child engine; production worker exposes no fixture backend."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridorkey.session import InferenceSession
from corridorkey.worker_core import SharedEngine
from corridorkey.worker_protocol import job_from_wire
from corridorkey.worker_transport import serve_stdio


class FixtureEngine(SharedEngine):
    def __init__(self):
        super().__init__()
        self.loads = 0
        self.selected_frame: int | None = None

    def load(self):
        self.loads += 1
        return object()

    def render(self, model, job, config, writer, diagnostics):
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter
        source = Path(job.input_path)
        data = json.loads((source / '.worker-control.json' if source.is_dir() else source).read_text())
        print('fixture Python engine log', flush=True)
        os.write(1, b'fixture native fd1 log\n')
        if data.get('exit'):
            os._exit(78)
        if data.get('descendant'):
            child_code = '''import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
count=0
while True:
    Path(sys.argv[2]).write_text(str(count))
    count+=1
    time.sleep(0.01)
'''
            self.descendant = subprocess.Popen([sys.executable, '-c', child_code, data['descendant'], data['heartbeat']])
            while not Path(data['descendant']).exists():
                time.sleep(0.01)
            diagnostics.control.checkpoint('fixture_descendant_ready', 0, 1)
            while True:
                time.sleep(0.05)  # Deliberately noncooperative test-only native-call substitute.

        while data.get('gate') and not Path(data['gate']).exists():
            diagnostics.control.checkpoint('fixture_wait', 0, 2)
            time.sleep(0.01)
        if data.get('fail'):
            raise ValueError('fixture render failed')
        directory = Path(job.output_dir) / 'alpha'
        rasters = None
        count, width, height, fps = 2, 2, 1, 24.
        source_start = 0
        if source.is_dir():
            from corridorkey.correction_rasters import CorrectionRasters
            from corridorkey.video_io import probe_frame_dir_meta
            meta = probe_frame_dir_meta(str(source), num_frames=config.num_frames, start_frame=config.start_frame, shape_quantum=config.shape_quantum,
                       fps=config.frame_dir_fps, exr_linear=config.frame_dir_linear, model_scale=config.model_scale,
                       alpha_mode=config.frame_dir_alpha)
            height, width = meta['native_hw']
            count, fps = meta['num_frames'], meta['native_fps']
            source_start = meta['start_frame']
            diagnostics.source = {k: v for k, v in meta.items() if k != 'frame_paths'}
            if diagnostics.correction_document is not None:
                rasters = CorrectionRasters(diagnostics.correction_document, meta)
        stream = ExrSequenceWriter(str(directory), fps, (width, height), channels=1, dtype='float', frame_name='alpha')
        try:
            for frame in range(count):
                diagnostics.control.checkpoint('fixture_frames', frame, count)
                alpha = (np.full((height, width), .5, np.float32) if source.is_dir()
                         else np.array([0., .5], dtype=np.float32).reshape(height, width))
                if rasters is not None:
                    alpha = rasters.apply(alpha, frame)
                if self.selected_frame is None or frame + source_start == self.selected_frame:
                    stream.write(alpha[..., None])
        finally:
            stream.release()
            if rasters is not None:
                rasters.close()
        diagnostics.outputs['alpha'] = str(directory)
        diagnostics.output_contracts['alpha'] = {'format': 'exr', 'channels': 1}
        if data.get('export_fg') and 'fg' in config.outputs:
            foreground = Path(job.output_dir) / 'fg'
            stream = ExrSequenceWriter(str(foreground), fps, (width, height), channels=3, dtype='float', frame_name='fg')
            try:
                for frame in range(count):
                    diagnostics.control.checkpoint('fixture_foreground', frame, count)
                    if self.selected_frame is None or frame + source_start == self.selected_frame:
                        stream.write(np.full((height, width, 3), .25, np.float32))
            finally:
                stream.release()
            diagnostics.outputs['fg'] = str(foreground)
            diagnostics.output_contracts['fg'] = {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear',
                                                 'foreground_mode': 'test_fixture'}
        if self.selected_frame is not None:
            relative = self.selected_frame - source_start
            if not 0 <= relative < count:
                raise ValueError('Fixture selected frame outside timeline')
            diagnostics.preview_selection = {'source_frame': self.selected_frame, 'relative_frame': relative,
                'context_frames_processed': count, 'captured_frames': 1, 'full_context_processed': True}
        diagnostics.frames_submitted = count if self.selected_frame is None else 1
        diagnostics.execution = {'neural_inference': False, 'fixture_loads': self.loads}

    def run(self, request, cancellation, progress):
        self.record = None
        self.selected_frame = request.payload().get('selected_frame')
        if request.action == 'recover':
            return self.recover(request, cancellation, progress)
        if self.session is None:
            self.session = InferenceSession(loader=self.load, runner=self.render, disposer=lambda _: None)
        try:
            record = self.session.run(job_from_wire(request.payload()['job']), cancellation=cancellation, progress=progress).to_dict()
            if self.selected_frame is not None:
                # Explicit non-neural test engine contract, never a production custom-runner opt-in.
                record['status'], record['end_to_end_fps'] = 'preview_complete', None
            return record
        except BaseException:
            if self.session.last_result is not None:
                self.record = self.session.last_result.to_dict()
            raise


    def recover(self, request, cancellation, progress):
        from corridorkey.batch import load_jobs
        from corridorkey.recovery import run_recoverable

        self.close()
        payload = request.payload()
        jobs, weights = load_jobs(payload['batch'])
        root = Path(payload['batch']).parent / 'fixture_assets'
        configured_root = getattr(self, 'fixture_root', None)
        if configured_root is not None:
            root = Path(configured_root)
        root.mkdir(exist_ok=True)
        assets = {}
        for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
            file = root / name
            file.write_bytes(b'explicit CPU fixture identity, no neural weights')
            assets[name] = file
        directory = root / 'cradio'
        directory.mkdir(exist_ok=True)
        (directory / 'config.json').write_text('{}')
        assets['cradio_dir'] = directory
        code = root / 'corridorkey'
        code.mkdir(exist_ok=True)
        (code / 'runtime.py').write_text('# explicit CPU worker fixture')

        def factory(**kwargs):
            return InferenceSession(loader=self.load, runner=self.render, disposer=lambda _: None, **kwargs)

        report = run_recoverable(
            jobs, workspace=payload['workspace'], max_cache_bytes=payload['max_cache_bytes'], assets=assets,
            environment={'backend': 'cpu-worker-fixture', 'precision': 'fixture-float32',
                         'dependencies': {'fixture': '1'}, 'runtime': {'synthetic': True}},
            implementation_root=root, weights=weights, session_factory=factory, cancellation=cancellation,
            progress=progress, resume=payload['resume'], durable_foundations=payload['durable_foundations'],
            durable_rvm=payload['durable_rvm'], durable_hann=payload['durable_hann'],
            durable_stage2=payload['durable_stage2'], durable_writer_prefix=payload['durable_writer_prefix'],
            reuse_upstream_finishing_edits=payload['reuse_upstream_finishing_edits'],
            reuse_stage2_finishing_edits=payload['reuse_stage2_finishing_edits'])
        return {'status': report['status'], 'report_path': report['report_path'], 'job_count': len(report['jobs']),
                'session_scope': 'fresh_recovery_batch', 'fixture_loads': self.loads}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=4)
    args = parser.parse_args()
    raise SystemExit(serve_stdio(capacity=args.capacity, engine=FixtureEngine()))
