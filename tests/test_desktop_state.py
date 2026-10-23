import json
import tempfile
import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.desktop_state import (
    load_project,
    pretty_settings,
    save_project,
    settings_job,
)
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


class DesktopStateTests(unittest.TestCase):
    def test_roundtrip_preserves_all_settings_without_running_a_job(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source.png'
            source.touch()
            job = JobRequest.create(source, Path(root) / 'out', InferenceConfig(output_format='exr',
                                    quick_key=True, quick_interval=3, foundation_cache_mb=11))
            path = Path(root) / 'project.json'
            save_project(path, job, 4)
            restored, frame = load_project(path)
            self.assertEqual(job_to_wire(restored), job_to_wire(job))
            self.assertEqual(frame, 4)
            config, writer = pretty_settings(job)
            self.assertEqual(job_to_wire(settings_job(str(source), job.output_dir, config, writer)), job_to_wire(job))
            self.assertFalse(Path(job.output_dir).exists())

    def test_protect_source_and_strict_project_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'source.json'
            path.write_text('original')
            job = JobRequest.create(path, Path(root) / 'out')
            with self.assertRaises(ValueError):
                save_project(path, job)
            self.assertEqual(path.read_text(), 'original')
            for raw in ['{"schema_version":1,"schema_version":1}',
                        json.dumps({'schema_version': True, 'type': 'corridorkey_desktop',
                                    'job': {}, 'preview_frame': 0})]:
                project = Path(root) / 'project.json'
                project.write_text(raw)
                with self.assertRaises(ValueError):
                    load_project(project)

    def test_duplicate_embedded_settings_reject(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'source'
            path.touch()
            config, writer = pretty_settings(JobRequest.create(path, Path(root) / 'out'))
            config = config.replace('"hann_chunk": 80', '"hann_chunk": 70, "hann_chunk": 80')
            with self.assertRaises(ValueError):
                settings_job(str(path), str(Path(root) / 'out'), config, writer)

    def test_oversized_project_rejects_even_with_valid_prefix_and_newlines(self):
        from corridorkey.worker_protocol import MAX_FRAME_BYTES
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'project.json'
            source = Path(root) / 'source.png'
            source.touch()
            job = JobRequest.create(source, Path(root) / 'output')
            save_project(path, job)
            with path.open('ab') as stream:
                stream.write(b'\n' * MAX_FRAME_BYTES)
            with self.assertRaisesRegex(ValueError, 'size limit'):
                load_project(path)

    def test_oversized_save_preserves_existing_project(self):
        from corridorkey.worker_protocol import MAX_FRAME_BYTES
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'project.json'
            path.write_text('previous project')
            job = JobRequest.create(Path(root) / 'source', Path(root) / 'output',
                                    writer=WriterConfig(codec='x' * MAX_FRAME_BYTES))
            with self.assertRaisesRegex(ValueError, 'size limit'):
                save_project(path, job)
            self.assertEqual(path.read_text(), 'previous project')
