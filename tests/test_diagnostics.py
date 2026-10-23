import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from corridorkey.diagnostics import json_value, write_json_atomic


@dataclass(frozen=True)
class Record:
    name: str
    path: Path


class SerializationTests(unittest.TestCase):
    def test_instances_and_recursive_containers_are_detached(self):
        value = {'record': Record('sample', Path('clip')), 'items': [
            (None, True, 3, 0.5), {3, 1, 2}, frozenset({'b', 'a'})]}
        result = json_value(value)
        self.assertEqual(result, {'record': {'name': 'sample', 'path': 'clip'},
                                 'items': [[None, True, 3, 0.5], [1, 2, 3], ['a', 'b']]})
        value['items'].append('later')
        self.assertEqual(len(result['items']), 3)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_classes_and_unsupported_values_are_explicitly_rejected(self):
        for value in (Record, object(), b'bytes', complex(1, 2), {'nested': [Record]}):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(
                    TypeError, 'Unsupported JSON record value'):
                json_value(value)

    def test_nested_class_in_dataclass_is_rejected(self):
        @dataclass
        class Wrapper:
            value: object
        with self.assertRaisesRegex(TypeError, 'Unsupported JSON record value'):
            json_value(Wrapper(Record))

    def test_mixed_and_structured_sets_are_deterministic(self):
        value = {None, 2, 'x', Path('clip'), (1, 'a'), Record('z', Path('p'))}
        result = json_value(value)
        self.assertEqual(len(result), 6)
        self.assertEqual(result, json_value(frozenset(value)))
        self.assertEqual(result, sorted(result, key=lambda v: json.dumps(
            v, sort_keys=True, allow_nan=False, separators=(',', ':'))))
        self.assertEqual(json_value({10, 2, -1, 0.5}), [-1, 0.5, 2, 10])
        self.assertEqual(json_value(set()), [])

    def test_set_order_is_stable_across_hash_seeds(self):
        code = ('import json; from pathlib import Path; '
                'from corridorkey.diagnostics import json_value; '
                'print(json.dumps(json_value({"x", None, 2, Path("clip"), (1, "a")})))')
        outputs = [subprocess.check_output([sys.executable, '-c', code], text=True,
                   env={**os.environ, 'PYTHONHASHSEED': seed}) for seed in ('1', '12', '999')]
        self.assertEqual(len(set(outputs)), 1)

    def test_keys_preserve_existing_conversion_and_reject_collisions(self):
        self.assertEqual(json_value({2: 'two', None: 0, Path('p'): 1}),
                         {'2': 'two', 'None': 0, 'p': 1})
        for value in ({1: 'first', '1': 'second'}, {Path('p'): 1, 'p': 2}):
            with self.assertRaisesRegex(ValueError, 'key collision'):
                json_value(value)
        with self.assertRaisesRegex(TypeError, 'Unsupported JSON record key'):
            json_value({object(): 1})

    def test_nonfinite_values_and_keys_are_rejected(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            for value in (invalid, {'nested': [invalid]}, {invalid}, {invalid: 1}):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite'):
                    json_value(value)

    def test_failed_serialization_preserves_atomic_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'report.json')
            write_json_atomic(path, {'ok': 1})
            for invalid in (Record, {1: 'a', '1': 'b'}, {'nested': object()}):
                with self.assertRaises((TypeError, ValueError)):
                    write_json_atomic(path, invalid)
                self.assertEqual(json.loads(path.read_text()), {'ok': 1})
                self.assertEqual(list(Path(folder).iterdir()), [path])
