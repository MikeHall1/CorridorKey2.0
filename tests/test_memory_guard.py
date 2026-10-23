"""Deterministic guard policy; injected observers never import native providers."""
import logging
import threading
import unittest
from typing import Any

from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.memory_guard import (
    ProcessTreeRssGuard,
    ResourceLimitError,
    ResourceMeasurementError,
)


def sample(value):
    return {'status': 'available', 'current_rss_bytes': value, 'provider': 'injected fixture', 'reason': None}


class Observer:
    def __init__(self, *samples):
        self.samples = list(samples)
        self.calls = 0
        self.callbacks = []

    def sample(self, check=None):
        self.callbacks.append(check)
        self.calls += 1
        if check is not None:
            check()
        return self.samples[min(self.calls - 1, len(self.samples) - 1)]


class MemoryGuardTests(unittest.TestCase):
    def test_throttled_forced_equal_threshold_and_cached_independent_snapshot(self):
        now = [0.0]
        observer = Observer(sample(10), sample(20), sample(30))
        guard = ProcessTreeRssGuard(30, observer, clock=lambda: now[0])
        guard.check()
        guard.check()
        self.assertEqual(observer.calls, 1)
        guard.check(force=True)
        now[0] = .25
        guard.check()
        self.assertEqual(observer.calls, 3)
        report = guard.snapshot()
        self.assertEqual(report['observed_peak_rss_bytes'], 30)
        self.assertIsNone(report['failure'])
        report['last_sample']['current_rss_bytes'] = -1
        self.assertEqual(guard.snapshot()['last_sample']['current_rss_bytes'], 30)
        self.assertEqual(observer.calls, 3)

    def test_limit_failure_is_distinct_and_latched_without_more_os_reads(self):
        observer = Observer(sample(101), sample(1))
        guard = ProcessTreeRssGuard(100, observer)
        with self.assertRaises(ResourceLimitError):
            guard.check()
        with self.assertRaises(ResourceLimitError):
            guard.check(force=True)
        self.assertEqual(observer.calls, 1)
        self.assertEqual(guard.snapshot()['failure']['type'], 'ResourceLimitError')
        self.assertEqual(guard.snapshot()['observed_peak_rss_bytes'], 101)

    def test_partial_and_malformed_samples_fail_closed_without_zero_totals(self):
        for data in ({'status': 'unavailable', 'current_rss_bytes': None, 'observed_rss_bytes': 50, 'reason': 'denied child'},
                     sample(False), sample(-1), {**sample(1), 'duration_seconds': float('nan')}):
            with self.subTest(data=data):
                guard = ProcessTreeRssGuard(100, Observer(data))
                with self.assertRaises(ResourceMeasurementError):
                    guard.check()
                self.assertIsNone(guard.snapshot()['observed_peak_rss_bytes'])
                self.assertEqual(guard.snapshot()['failure']['type'], 'ResourceMeasurementError')

    def test_cancellation_precedes_hook_and_observer_receives_raw_token_callback(self):
        token = CancellationToken()
        observer = Observer(sample(1))
        guard = ProcessTreeRssGuard(100, observer, cancel_check=token.check)
        control = JobControl(token, check_hook=guard.check)
        control.check()
        self.assertEqual(observer.callbacks, [token.check])
        token.cancel('explicit user cancellation')
        with self.assertRaisesRegex(CancelledError, 'explicit user cancellation'):
            control.check()
        self.assertEqual(observer.calls, 1)
        self.assertIsNone(guard.snapshot()['failure'])

    def test_concurrent_writer_and_owner_checks_share_one_throttled_sample(self):
        entered, release = threading.Event(), threading.Event()
        calls, errors = [], []

        class Gated:
            def sample(self, check=None):
                calls.append(True)
                entered.set()
                release.wait(3)
                return sample(1)

        guard = ProcessTreeRssGuard(100, Gated(), clock=lambda: 0.)

        def run():
            try:
                guard.check()
            except BaseException as exc:
                logging.getLogger(__name__).exception('Concurrent guard fixture failed')
                errors.append(exc)

        first, second = threading.Thread(target=run), threading.Thread(target=run)
        try:
            first.start()
            self.assertTrue(entered.wait(2))
            second.start()
        finally:
            release.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)

    def test_cached_snapshot_does_not_wait_for_inflight_native_sample(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        snapshots = []

        class Gated:
            def sample(self, check=None):
                entered.set()
                release.wait(3)
                return sample(1)

        guard = ProcessTreeRssGuard(100, Gated())
        sampler = threading.Thread(target=guard.check)

        def read_snapshot():
            snapshots.append(guard.snapshot())
            finished.set()

        reader = threading.Thread(target=read_snapshot)
        try:
            sampler.start()
            self.assertTrue(entered.wait(2))
            reader.start()
            self.assertTrue(finished.wait(.5), 'cached diagnostics waited on a provider query')
            self.assertEqual(snapshots[0]['samples'], 0)
        finally:
            release.set()
            sampler.join(3)
            if reader.ident is not None:
                reader.join(3)
        self.assertFalse(sampler.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual(guard.snapshot()['samples'], 1)

    def test_invalid_policy_never_queries_an_observer(self):
        observer = Observer(sample(1))
        invalid_limits: list[Any] = [None, True, -1, 2**63]
        for limit in invalid_limits:
            with self.assertRaises(ValueError):
                ProcessTreeRssGuard(limit, observer)
        for interval in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ProcessTreeRssGuard(1, observer, interval=interval)
        self.assertEqual(observer.calls, 0)
