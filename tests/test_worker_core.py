import threading
import time
import unittest

from corridorkey.control import ProgressEvent
from corridorkey.worker_core import WorkerCore
from corridorkey.worker_protocol import parse_request


def request(name, action, payload=None):
    return parse_request({'schema_version': 1, 'request_id': name, 'action': action, 'payload': payload or {}})


def recover(name):
    return request(name, 'recover', {'batch': '/tmp/fixture.json', 'workspace': '/tmp/cache', 'max_cache_bytes': 0})


class FixtureEngine:
    def __init__(self):
        self.gate = threading.Event()
        self.closed = False
        self.calls = []

    def run(self, request, cancellation, progress):
        self.calls.append(request.request_id)
        progress(ProgressEvent('fixture', 0, 1))
        while not self.gate.wait(0.01):
            cancellation.check()
        cancellation.check()
        if request.request_id == 'bad':
            raise ValueError('fixture failure')
        return {'status': 'complete'}

    def failure_record(self):
        return {'fixture': True}

    def close(self):
        self.closed = True


class WorkerCoreTests(unittest.TestCase):
    def test_terminal_execution_interval_brackets_engine_not_event_delivery(self):
        core, engine, events, condition = self.setup_core(capacity=2)
        before = time.monotonic_ns()
        core.dispatch(recover('measured'))
        self.wait(events, condition, 'progress', 'measured')
        release = time.monotonic_ns()
        engine.gate.set()
        self.wait(events, condition, 'result', 'measured')
        terminal = next(payload for kind, name, payload in events if kind == 'result' and name == 'measured')
        interval = terminal['execution_interval']
        self.assertLessEqual(before, interval['started_monotonic_ns'])
        self.assertLessEqual(interval['started_monotonic_ns'], release)
        self.assertGreaterEqual(interval['finished_monotonic_ns'], release)
        self.assertEqual(interval['clock_implementation'], time.get_clock_info('monotonic').implementation)
        self.assertGreater(interval['pid'], 0)
        core.dispatch(recover('bad'))
        self.wait(events, condition, 'failed', 'bad')
        failed = next(payload for kind, name, payload in events if kind == 'failed' and name == 'bad')
        self.assertGreaterEqual(failed['execution_interval']['finished_monotonic_ns'],
                                failed['execution_interval']['started_monotonic_ns'])

    def setup_core(self, capacity=1, max_requests=4096):
        engine = FixtureEngine()
        events = []
        condition = threading.Condition()

        def emit(kind, name, payload):
            with condition:
                events.append((kind, name, payload))
                condition.notify_all()

        core = WorkerCore(engine, emit, capacity=capacity, max_requests=max_requests)
        self.addCleanup(lambda: (core.stop(), engine.gate.set(), core.finish()))
        return core, engine, events, condition

    def wait(self, events, condition, kind, name):
        with condition:
            self.assertTrue(condition.wait_for(lambda: any(k == kind and n == name for k, n, _ in events), timeout=2))

    def test_capacity_cancel_before_start_and_active_cancel_are_concurrent(self):
        core, engine, events, condition = self.setup_core()
        core.dispatch(recover('a'))
        self.wait(events, condition, 'started', 'a')
        core.dispatch(recover('b'))
        core.dispatch(recover('c'))
        self.wait(events, condition, 'rejected', 'c')
        core.dispatch(request('cancel-b', 'cancel', {'target': 'b'}))
        self.wait(events, condition, 'cancelled', 'b')
        core.dispatch(request('cancel-a', 'cancel', {'target': 'a'}))
        self.wait(events, condition, 'cancelled', 'a')
        self.assertEqual(engine.calls, ['a'])
        core.dispatch(recover('b'))
        self.assertEqual([p['code'] for k, n, p in events if k == 'rejected' and n == 'b'], ['duplicate_id'])
        core.stop()
        self.assertTrue(core.finish())
        self.assertTrue(engine.closed)

    def test_failure_continues_queue_and_draining_shutdown_preserves_order(self):
        core, engine, events, _condition = self.setup_core(capacity=3)
        core.dispatch(recover('bad'))
        core.dispatch(recover('good'))
        core.dispatch(request('stop', 'shutdown', {'drain': True}))
        engine.gate.set()
        self.assertTrue(core.finish())
        terminal = [(k, n) for k, n, _ in events if k in ('result', 'failed', 'cancelled')]
        self.assertEqual(terminal, [('failed', 'bad'), ('result', 'good')])
        core.dispatch(recover('later'))
        self.assertEqual(events[-1][2]['code'], 'stopping')

    def test_bounded_history_requires_explicit_restart(self):
        core, engine, events, condition = self.setup_core(max_requests=2)
        engine.gate.set()
        for name in ('a', 'b'):
            core.dispatch(recover(name))
            self.wait(events, condition, 'result', name)
        core.dispatch(recover('c'))
        self.assertEqual(events[-1][2]['code'], 'restart_required')
        core.dispatch(request('status', 'status'))
        self.assertTrue(events[-1][2]['restart_required'])
        self.assertEqual(len(core.seen), 2)

    def test_finish_timeout_does_not_claim_process_stopped(self):
        core, engine, events, condition = self.setup_core()
        core.dispatch(recover('a'))
        self.wait(events, condition, 'started', 'a')
        started = time.monotonic()
        self.assertFalse(core.finish(timeout=0.01))
        self.assertLess(time.monotonic() - started, 0.5)
        engine.gate.set()
        core.stop(drain=True)
        self.assertTrue(core.finish())

    def test_late_progress_cannot_follow_a_terminal_event(self):
        core, engine, events, condition = self.setup_core()
        saved = []
        original = engine.run

        def run(request, cancellation, progress):
            saved.append(progress)
            return original(request, cancellation, progress)

        engine.run = run
        engine.gate.set()
        core.dispatch(recover('done'))
        self.wait(events, condition, 'result', 'done')
        saved[0](ProgressEvent('late_encoder_callback', 1, 1))
        self.assertFalse(any(kind == 'progress' and payload['stage'] == 'late_encoder_callback'
                             for kind, _, payload in events))
