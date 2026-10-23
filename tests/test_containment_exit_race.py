"""Terminal sampling races preserve birth-pinned descendant/group checks."""
import os
import unittest
from unittest.mock import patch

from corridorkey.process_memory import MeasurementError, ProcessInfo, ProcessTreeMemory
from corridorkey.stress_containment import StressContainment


class Provider:
    name = 'deterministic-terminal-transition'

    def __init__(self):
        self.records = {100: ProcessInfo(100, 1, 'owned-root', 10)}
        self.mode = ''
        self.terminal = False
        self.root_reads = 0

    def begin(self, tick):
        tick()
        self.root_reads = 0
        if self.mode in ('exit', 'reuse', 'denied', 'deadline'):
            self.terminal = True
        if self.mode == 'exit':
            self.records.pop(100, None)
        elif self.mode == 'reuse':
            self.records[100] = ProcessInfo(100, 1, 'unrelated-reused-pid', 10)
        elif self.mode == 'denied':
            raise PermissionError('native observation denied')
        elif self.mode == 'deadline':
            raise MeasurementError('Cooperative sample deadline exceeded')

    def read(self, pid):
        if pid == 100:
            self.root_reads += 1
            if self.mode == 'final_exit' and self.root_reads == 2:
                self.records.pop(100, None)
                self.terminal = True
        return self.records.get(pid)

    def children(self, pid, tick):
        tick()
        return [row.pid for row in self.records.values() if row.parent_pid == pid]


@unittest.skipIf(os.name == 'nt', 'Native Windows containment remains NOT VERIFIED')
class TerminalRaceTests(unittest.TestCase):
    def make_owner(self):
        provider = Provider()
        observer = ProcessTreeMemory(100, provider=provider)
        owner = StressContainment(100, observer=observer, root_exited=lambda: provider.terminal)
        return provider, observer, owner

    def verify_no_group(self, owner):
        with patch('corridorkey.stress_containment.os.killpg', side_effect=ProcessLookupError) as groups:
            owner.verify(timeout=0)
        groups.assert_called_once_with(100, 0)

    def test_exit_between_owner_poll_and_sample_keeps_rss_unavailable_but_cleanup_valid(self):
        provider, observer, owner = self.make_owner()
        owner.sample(force=True)
        self.assertEqual(owner.complete_samples, 1)
        provider.mode = 'exit'
        owner.sample(force=True)
        self.assertIsNone(owner.failure)
        self.assertEqual(owner.terminal_exit_races, 1)
        result = observer.sample()
        self.assertEqual(result['reason_code'], 'root_exited')
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['current_rss_bytes'])
        self.verify_no_group(owner)

    def test_exit_during_final_identity_check_has_distinct_structured_reason(self):
        provider, _, owner = self.make_owner()
        owner.sample(force=True)
        provider.mode = 'final_exit'
        owner.sample(force=True)
        self.assertTrue(provider.terminal)
        self.assertEqual(owner.terminal_exit_races, 1)
        self.verify_no_group(owner)

    def test_initial_exit_without_complete_discovery_remains_unverified(self):
        provider, _, owner = self.make_owner()
        provider.mode = 'exit'
        owner.sample(force=True)
        self.assertEqual(owner.complete_samples, 0)
        self.assertEqual(owner.terminal_exit_races, 0)
        with self.assertRaisesRegex(RuntimeError, 'NOT VERIFIED'):
            self.verify_no_group(owner)

    def test_owner_must_confirm_terminal_not_just_native_absence(self):
        provider, _, owner = self.make_owner()
        owner.sample(force=True)
        owner.root_exited = lambda: False
        provider.mode = 'exit'
        owner.sample(force=True)
        self.assertIsNotNone(owner.failure)
        self.assertEqual(owner.terminal_exit_races, 0)

    def test_reuse_denial_and_deadline_remain_failed_even_if_owner_terminal(self):
        for mode, reason_code in (('reuse', 'root_identity_changed'), ('denied', 'measurement_unavailable'),
                                  ('deadline', 'measurement_unavailable')):
            with self.subTest(mode=mode):
                provider, observer, owner = self.make_owner()
                owner.sample(force=True)
                provider.mode = mode
                owner.sample(force=True)
                self.assertTrue(provider.terminal)
                self.assertEqual(owner.terminal_exit_races, 0)
                self.assertIsNotNone(owner.failure)
                self.assertEqual(observer.sample()['reason_code'], reason_code)
                with self.assertRaisesRegex(RuntimeError, 'NOT VERIFIED'):
                    self.verify_no_group(owner)

    def test_previous_partial_discovery_is_never_cleared_by_later_exit(self):
        provider, _, owner = self.make_owner()
        owner.sample(force=True)
        provider.mode = 'denied'
        owner.sample(force=True)
        failure = owner.failure
        provider.mode = 'exit'
        provider.terminal = False
        owner.sample(force=True)
        self.assertEqual(owner.failure, failure)
        with self.assertRaisesRegex(RuntimeError, 'NOT VERIFIED'):
            self.verify_no_group(owner)

    def test_terminal_race_never_skips_tracked_descendant_or_unknown_group_checks(self):
        provider, _, owner = self.make_owner()
        provider.records[101] = ProcessInfo(101, 100, 'owned-child', 10)
        owner.sample(force=True)
        provider.mode = 'exit'
        owner.sample(force=True)
        self.assertIn(101, owner.pins)
        with self.assertRaisesRegex(RuntimeError, 'remain live'):
            self.verify_no_group(owner)
        provider, _, owner = self.make_owner()
        owner.sample(force=True)
        provider.mode = 'exit'
        owner.sample(force=True)
        with patch('corridorkey.stress_containment.os.killpg') as groups, self.assertRaisesRegex(RuntimeError, 'group remains present'):
            owner.verify(timeout=0)
        groups.assert_called_once_with(100, 0)  # Query only; never destructive authority.

    def test_partial_child_discovery_during_parent_exit_is_not_benign_root_exit(self):
        provider, observer, owner = self.make_owner()
        owner.sample(force=True)
        provider.records[101] = ProcessInfo(101, 100, 'new-child', 10)
        original = provider.read

        def read(pid):
            if pid == 101:
                provider.records.pop(100, None)
                provider.terminal = True
            return original(pid)

        with patch.object(provider, 'read', side_effect=read):
            owner.sample(force=True)
        self.assertEqual(owner.terminal_exit_races, 0)
        self.assertIsNotNone(owner.failure)
        self.assertEqual(observer.sample()['reason_code'], 'root_exited')
        with self.assertRaisesRegex(RuntimeError, 'NOT VERIFIED'):
            self.verify_no_group(owner)


if __name__ == '__main__':
    unittest.main()
