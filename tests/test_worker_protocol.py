import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
    job_from_wire,
    job_to_wire,
    parse_event,
    parse_request,
    request_frame,
)


class ProtocolTests(unittest.TestCase):
    def test_job_and_request_snapshots_preserve_shared_settings(self):
        job = JobRequest.create('/tmp/source', '/tmp/output', InferenceConfig(outputs={'alpha'}, output_format='exr'))
        wire = job_to_wire(job)
        restored = job_from_wire(wire)
        self.assertEqual(job_to_wire(restored), job_to_wire(job))
        request = parse_request(decode_frame(request_frame('job-1', 'render', {'job': wire})))
        wire['output_dir'] = '/tmp/changed'
        self.assertEqual(request.payload()['job']['output_dir'], job.output_dir)
        payload = request.payload()
        payload.clear()
        self.assertIn('job', request.payload())

    def test_ambiguous_nonfinite_oversize_partial_and_invalid_envelopes_reject(self):
        for frame in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":1e309}\n', b'{"a":-1e309}\n', b'{}', b'[]\n', b'\xff\n', b'x' * (MAX_FRAME_BYTES + 1)):
            with self.subTest(frame=frame[:30]), self.assertRaises((ValueError, TypeError, UnicodeError)):
                decode_frame(frame)
        base = {'schema_version': 1, 'request_id': 'job', 'action': 'status', 'payload': {}}
        for changes in ({'schema_version': True}, {'request_id': '../bad'}, {'extra': 1}, {'action': 'eval'}, {'payload': []}):
            with self.assertRaises((ValueError, TypeError)):
                parse_request({**base, **changes})
        with self.assertRaises((ValueError, TypeError)):
            encode_frame({'value': float('inf')})

    def test_recovery_paths_budget_switches_and_controls_are_strict(self):
        payload = {'batch': '/tmp/batch.json', 'workspace': '/tmp/cache', 'max_cache_bytes': 0}
        request = parse_request(decode_frame(request_frame('r', 'recover', payload)))
        self.assertFalse(request.payload()['durable_hann'])
        for changes in ({'batch': 'relative'}, {'max_cache_bytes': True}, {'durable_hann': 1}, {'extra': 1}):
            with self.assertRaises((ValueError, TypeError)):
                request_frame('r', 'recover', {**payload, **changes})
        for action, value in (('cancel', {'target': ''}), ('shutdown', {'drain': 1}), ('status', {'extra': 1})):
            with self.assertRaises((ValueError, TypeError)):
                request_frame('c', action, value)
        self.assertTrue(Path(job_from_wire(job_to_wire(JobRequest.create('/tmp/s', '/tmp/o'))).input_path).is_absolute())

    def test_events_require_generation_sequence_and_version(self):
        event = {'schema_version': 1, 'generation': 'a' * 32, 'sequence': 0, 'kind': 'ready', 'request_id': None, 'payload': {}}
        self.assertEqual(parse_event(encode_frame(event)), event)
        for changes in ({'sequence': True}, {'generation': 'bad'}, {'kind': 'invented'}, {'schema_version': 2}):
            with self.assertRaises((ValueError, TypeError)):
                parse_event(encode_frame({**event, **changes}))
