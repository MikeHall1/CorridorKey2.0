"""Shared-session memory policy with injected measurements, no native/GPU claims."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from corridorkey.config import InferenceConfig
from corridorkey.memory_guard import (
    ProcessTreeRssGuard,
    ResourceLimitError,
    ResourceMeasurementError,
)
from corridorkey.session import InferenceSession, JobRequest
from tests.test_memory_guard import Observer, sample


class SessionMemoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.touch()
        self.config = InferenceConfig(output_format='exr', process_tree_rss_stop_bytes=100)

    def job(self, config=None):
        return JobRequest.create(self.source, self.root / 'output', config or self.config)

    def test_default_off_does_not_construct_provider_or_leak_previous_policy(self):
        observer = Observer(sample(1))
        guard = ProcessTreeRssGuard(100, observer, clock=lambda: 0.)
        with InferenceSession(loader=lambda: object(), runner=Mock(), disposer=Mock()) as session:
            measured = session.run(self.job(), resource_guard=guard)
            self.assertEqual(measured.to_dict()['diagnostics']['resource_policy']['observed_peak_rss_bytes'], 1)
            with patch.dict('sys.modules', {'corridorkey.process_memory': None}):
                unmeasured = session.run(self.job(replace(self.config, process_tree_rss_stop_bytes=None)))
            self.assertIsNone(unmeasured.to_dict()['diagnostics']['resource_policy'])

    def test_preflight_failure_does_not_load_or_discard_untouched_cached_model(self):
        loader, disposer = Mock(return_value=object()), Mock()
        with InferenceSession(loader=loader, runner=Mock(), disposer=disposer) as session:
            session.run(self.job(replace(self.config, process_tree_rss_stop_bytes=None)))
            guard = ProcessTreeRssGuard(100, Observer(sample(101)))
            with self.assertRaises(ResourceLimitError):
                session.run(self.job(), resource_guard=guard)
            loader.assert_called_once()
            disposer.assert_not_called()
            assert session.last_result is not None
            result = session.last_result.to_dict()
            self.assertEqual(result['status'], 'failed')
            self.assertIsNone(result['end_to_end_fps'])
            self.assertEqual(result['diagnostics']['resource_policy']['failure']['type'], 'ResourceLimitError')

    def test_unavailable_first_sample_fails_before_loader(self):
        loader = Mock()
        guard = ProcessTreeRssGuard(100, Observer({'status': 'unavailable', 'current_rss_bytes': None, 'reason': 'fixture access denied'}))
        with InferenceSession(loader=loader, runner=Mock(), disposer=Mock()) as session:
            with self.assertRaisesRegex(ResourceMeasurementError, 'fixture access denied'):
                session.run(self.job(), resource_guard=guard)
            loader.assert_not_called()
            assert session.last_result is not None
            self.assertEqual(session.last_result.to_dict()['status'], 'failed')

    def test_forced_postload_sample_discards_model_despite_throttle(self):
        observer = Observer(sample(1), sample(1), sample(101))
        guard = ProcessTreeRssGuard(100, observer, clock=lambda: 0.)
        runner, disposer = Mock(), Mock()
        with InferenceSession(loader=lambda: object(), runner=runner, disposer=disposer) as session:
            with self.assertRaises(ResourceLimitError):
                session.run(self.job(), resource_guard=guard)
            runner.assert_not_called()
            disposer.assert_called_once()
            self.assertEqual(observer.calls, 3)
            assert session.last_result is not None
            self.assertIsNone(session.last_result.to_dict()['end_to_end_fps'])

    def test_forced_completion_sample_fails_after_runner_without_success_fps(self):
        observer = Observer(sample(1), sample(1), sample(1), sample(101))
        guard = ProcessTreeRssGuard(100, observer, clock=lambda: 0.)

        def runner(model, job, config, writer, diagnostics):
            diagnostics.frames_submitted = 2

        disposer = Mock()
        with InferenceSession(loader=lambda: object(), runner=runner, disposer=disposer) as session:
            with self.assertRaises(ResourceLimitError):
                session.run(self.job(), resource_guard=guard)
            self.assertEqual(observer.calls, 4)
            disposer.assert_called_once()
            assert session.last_result is not None
            result = session.last_result.to_dict()
            self.assertEqual(result['diagnostics']['frames_submitted'], 2)
            self.assertIsNone(result['end_to_end_fps'])

    def test_primary_runner_failure_is_not_replaced_by_terminal_sampling(self):
        observer = Observer(sample(1), sample(2), sample(3), {'status': 'unavailable', 'current_rss_bytes': None})
        guard = ProcessTreeRssGuard(100, observer, clock=lambda: 0.)
        with InferenceSession(loader=lambda: object(), runner=Mock(side_effect=ValueError('primary processing failure')),
                              disposer=Mock()) as session:
            with self.assertRaisesRegex(ValueError, 'primary processing failure'):
                session.run(self.job(), resource_guard=guard)
            self.assertEqual(observer.calls, 3)
            assert session.last_result is not None
            result = session.last_result.to_dict()
            self.assertEqual(result['error']['type'], 'ValueError')
            self.assertEqual(result['diagnostics']['resource_policy']['last_sample']['current_rss_bytes'], 3)
            json.dumps(result, allow_nan=False)

    def test_supplied_guard_limit_must_match_request_before_model_work(self):
        loader = Mock()
        guard = ProcessTreeRssGuard(50, Observer(sample(1)))
        with InferenceSession(loader=loader, runner=Mock(), disposer=Mock()) as session:
            with self.assertRaisesRegex(ValueError, 'differs'):
                session.run(self.job(), resource_guard=guard)
            loader.assert_not_called()
