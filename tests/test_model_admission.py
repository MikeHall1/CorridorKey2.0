"""CPU asset admission regressions; tiny local assets, no pretrained claims."""
import contextlib
import importlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from corridorkey.foundations import FrozenFoundations, _load_local_cradio_so400m
from corridorkey.runtime import load_model, load_runtime_checkpoint
from corridorkey.runtime_paths import RuntimePaths
from scripts.convert_to_safetensors import convert_checkpoint, convert_stats


class PrintPayload:
    def __reduce__(self):
        return print, ('MODEL_PAYLOAD_EXECUTED',)


class ModelAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def foundations(self):
        model = FrozenFoundations(device='cpu', use_moge=False, use_cradio=False)
        model.use_moge = True
        model.use_rvm = True
        model.use_cradio = True
        model.register_buffer("cradio_mean", torch.zeros(1))
        model.register_buffer("cradio_std", torch.ones(1))
        return model

    def stats(self):
        return {name: torch.tensor([2. if name.endswith('mean') else 4.])
                for name in ('moge_mean', 'moge_std', 'rvm_mean', 'rvm_std', 'cradio_mean', 'cradio_std')}

    def test_pickle_payload_refused_without_execution_at_all_legacy_boundaries(self):
        path = self.root / 'malicious.pt'
        torch.save({'payload': PrintPayload()}, path)
        operations = [lambda: load_runtime_checkpoint(path),
                      lambda: self.foundations().load_stats(str(path)),
                      lambda: convert_checkpoint(path, self.root / 'checkpoint.safetensors'),
                      lambda: convert_stats(path, self.root / 'stats.safetensors')]
        for operation in operations:
            with self.subTest(operation=operation):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises((pickle.UnpicklingError, RuntimeError)):
                    operation()
                self.assertNotIn('MODEL_PAYLOAD_EXECUTED', output.getvalue())
        self.assertFalse((self.root / 'checkpoint.safetensors').exists())
        self.assertFalse((self.root / 'stats.safetensors').exists())

    def test_valid_legacy_runtime_and_conversion(self):
        checkpoint = {'format': 'corridorkey_v2_runtime_checkpoint',
                      'model_config': {'channels': 2}, 'metadata': {},
                      'state_dict': {'weight': torch.tensor([3.])}}
        path = self.root / 'checkpoint.pt'
        torch.save(checkpoint, path)
        loaded = load_runtime_checkpoint(path)
        self.assertEqual(loaded['model_config'], checkpoint['model_config'])
        torch.testing.assert_close(loaded['state_dict']['weight'], checkpoint['state_dict']['weight'])
        target = self.root / 'checkpoint.safetensors'
        convert_checkpoint(path, target)
        torch.testing.assert_close(load_file(str(target))['weight'], torch.tensor([3.]))

    def test_runtime_cannot_override_separately_admitted_foundation_assets(self):
        stats_path = self.root / 'stats.pt'
        torch.save({'moge_mean': torch.zeros(1), 'moge_std': torch.ones(1)}, stats_path)

        class Tiny(torch.nn.Module):
            def __init__(self, **config):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.foundations = FrozenFoundations(device='cpu', use_moge=False)
                self.foundations.use_moge = True
                self.foundations.moge = torch.nn.Linear(1, 1, bias=False)
                self.foundations.load_stats(config['foundation_stats_path'])

        checkpoint_path = self.root / 'runtime.pt'
        paths = RuntimePaths(package_root=self.root, weights=checkpoint_path, foundation_stats=stats_path,
                             cradio_dir=self.root, moge_checkpoint=self.root, rvm_ckpt=self.root)
        checkpoint = {'format': 'corridorkey_v2_runtime_checkpoint', 'model_config': {},
                      'state_dict': {'weight': torch.tensor([3.])}}
        with patch('corridorkey.runtime.prepare_runtime_environment'), \
                patch('corridorkey.runtime.CorridorKeyV3', Tiny):
            torch.save(checkpoint, checkpoint_path)
            model = load_model(paths, device='cpu')
            torch.testing.assert_close(model.weight, torch.tensor([3.]))
            torch.testing.assert_close(model.foundations.moge_std, torch.ones(1))
            self.assertFalse(model.training)
            for key, value in [('foundations.moge_std', -torch.ones(1)),
                               ('foundations.moge.weight', torch.zeros(1, 1)),
                               ('foundations.unexpected', torch.ones(1))]:
                with self.subTest(key=key):
                    torch.save({**checkpoint, 'state_dict': {**checkpoint['state_dict'], key: value}},
                               checkpoint_path)
                    with self.assertRaisesRegex(RuntimeError, 'foundation'):
                        load_model(paths, device='cpu')

    def test_valid_stats_legacy_and_safetensors_are_exact(self):
        for suffix in ('.pt', '.safetensors'):
            path = self.root / ('stats' + suffix)
            stats = self.stats()
            if suffix == '.pt':
                torch.save(stats, path)
                convert_stats(path, self.root / 'converted.safetensors')
            else:
                save_file(stats, str(path))
            model = self.foundations()
            model.load_stats(str(path))
            for name, tensor in stats.items():
                torch.testing.assert_close(getattr(model, name), tensor, rtol=0, atol=0)

    def test_invalid_stats_fail_atomically(self):
        cases = []
        missing = self.stats()
        del missing['rvm_std']
        cases.append(missing)
        for key, value in [('rvm_std', torch.ones(2)), ('rvm_mean', torch.tensor([float('nan')])),
                           ('rvm_std', torch.tensor([float('inf')])), ('rvm_std', torch.zeros(1)),
                           ('rvm_std', -torch.ones(1)), ('rvm_std', 'not a tensor'),
                           ('cradio_mean', torch.tensor([float('inf')])),
                           ('cradio_std', torch.tensor([0.])),
                           ('moge_mean', torch.tensor([1e300], dtype=torch.float64))]:
            cases.append({**self.stats(), key: value})
        for index, stats in enumerate(cases):
            with self.subTest(index=index):
                path = self.root / 'stats.pt'
                torch.save(stats, path)
                model = self.foundations()
                original = {key: value.clone() for key, value in model.named_buffers()}
                with self.assertRaises((ValueError, RuntimeError)):
                    model.load_stats(str(path))
                for key, value in original.items():
                    torch.testing.assert_close(getattr(model, key), value, rtol=0, atol=0)

    def test_supplied_missing_stats_or_directory_refused(self):
        for path in (self.root / 'absent.pt', self.root):
            with self.subTest(path=path), self.assertRaises((FileNotFoundError, ValueError)):
                FrozenFoundations(device='cpu', use_moge=False, stats_path=str(path))

    def snapshot(self, name, fail=False):
        path = self.root / name
        path.mkdir()
        (path / 'helper.py').write_text(f'MARKER = {name!r}\n')
        (path / 'hf_model.py').write_text(
            'from .helper import MARKER\n' +
            ('raise RuntimeError("import failed")\n' if fail else '') +
            'class RADIOConfig:\n @classmethod\n def from_pretrained(cls, *a, **k): return None\n'
            'class RADIOModel:\n @classmethod\n def from_pretrained(cls, path, **k): return (MARKER, path)\n')
        return path

    def test_snapshot_roots_are_isolated_with_relative_imports_and_reuse(self):
        a, b = self.snapshot('A'), self.snapshot('B')
        for path in (a, b, b, a):
            self.assertEqual(_load_local_cradio_so400m(path), (path.name, str(path.resolve())))
        (a / 'hf_model.py').write_text('raise RuntimeError("must reuse same root")')
        self.assertEqual(_load_local_cradio_so400m(a), ('A', str(a.resolve())))

    def test_failed_snapshot_import_removes_its_modules_and_allows_retry(self):
        path = self.snapshot('failed', fail=True)
        before = set(sys.modules)
        with self.assertRaisesRegex(RuntimeError, 'import failed'):
            _load_local_cradio_so400m(path)
        self.assertEqual({name for name in set(sys.modules) - before
                          if name.startswith('_corridorkey_cradio_snapshot')}, set())
        source = path / 'hf_model.py'
        source.write_text(source.read_text().replace('raise RuntimeError("import failed")\n', ''))
        self.assertEqual(_load_local_cradio_so400m(path), ('failed', str(path.resolve())))

    def test_cradio_internal_typeerror_is_one_call(self):
        class Radio(torch.nn.Module):
            calls = 0

            def forward(self, value, **kwargs):
                self.calls += 1
                raise TypeError('internal failure')

        model = self.foundations()
        model.cradio = Radio()
        with self.assertRaisesRegex(TypeError, 'internal failure'):
            model(rgb_cradio=torch.ones(1, 3, 16, 16))
        self.assertEqual(model.cradio.calls, 1)

    def test_supported_cradio_contract_reshapes_tokens_and_normalizes(self):
        class Radio(torch.nn.Module):
            patch_size = 16

            def forward(self, value):
                return SimpleNamespace(features=torch.tensor([[[6.], [10.]]]))

        model = self.foundations()
        model.cradio = Radio()
        model.register_buffer('cradio_mean', torch.tensor([2.]))
        model.register_buffer('cradio_std', torch.tensor([4.]))
        result = model(rgb_cradio=torch.ones(1, 3, 16, 32))['cradio']
        torch.testing.assert_close(result, torch.tensor([[[[1., 2.]]]]))

    def test_moge_requires_complete_declared_architecture(self):
        with patch.object(sys, 'path', [str(Path('third_party/MoGe').resolve()), *sys.path]):
            MoGeModel = importlib.import_module("moge.model.v2").MoGeModel

        class Tiny(MoGeModel):
            def __init__(self, **kwargs):
                torch.nn.Module.__init__(self)
                self.encoder = torch.nn.Linear(1, 1, bias=False)
                self.normal_head = torch.nn.Linear(1, 1, bias=False)

        state = Tiny().state_dict()
        cases = [{}, {'normal_head.weight': state['normal_head.weight']},
                 {'encoder.weight': state['encoder.weight']},
                 {**state, 'encoder.unexpected': torch.ones(1)},
                 {**state, 'encoder.weight': torch.ones(2, 2)}]
        path = self.root / 'moge.pt'
        for index, bad_state in enumerate(cases):
            torch.save({'model_config': {}, 'model': bad_state}, path)
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                Tiny.from_pretrained(path)
        torch.save({'model_config': {}, 'model': state}, path)
        loaded = Tiny.from_pretrained(path)
        for key, value in state.items():
            torch.testing.assert_close(loaded.state_dict()[key], value)
