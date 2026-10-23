"""Loader setup contracts with CPU model substitutes; no pretrained execution."""
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires optional CPU PyTorch')
class FoundationSetupTests(unittest.TestCase):
    def model(self, error=None):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.setup_started = False

            def enable_pytorch_native_sdpa(self):
                self.setup_started = True
                if error is not None:
                    raise error

        return Model()

    def load(self, model):
        from corridorkey.foundations import build_moge_vitb

        module = types.ModuleType('moge.model.v2')
        with patch.object(module, 'MoGeModel', create=True) as factory, \
                patch.dict(sys.modules, {'moge.model.v2': module}):
            factory.from_pretrained.return_value = model
            result = build_moge_vitb(device='cpu', checkpoint='local-test-checkpoint')
            factory.from_pretrained.assert_called_once_with('local-test-checkpoint')
            return result

    def test_partial_native_setup_error_is_not_silently_accepted(self):
        for error in (RuntimeError('second block failed'), AttributeError('unexpected bundled setup error')):
            model = self.model(error)
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as caught:
                    self.load(model)
                self.assertIs(caught.exception, error)
                self.assertTrue(model.setup_started)

    def test_successful_native_setup_keeps_eval_and_parameter_freezing(self):
        model = self.model()
        self.assertIs(self.load(model), model)
        self.assertTrue(model.setup_started)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(model.weight.device.type, 'cpu')
