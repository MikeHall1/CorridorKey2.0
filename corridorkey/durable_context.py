"""Shared loaded-model binding for explicitly owned durable neural caches."""
from pathlib import Path

from .recovery_store import CacheInvalid


class BoundModelCache:
    def __init__(self, store, parent_identity, *, validate_context, check=None):
        self.store = store
        self.parent = parent_identity
        self.validate_context = validate_context
        self.check = check
        self.bound = False

    def bind(self, asset_paths):
        self.bound = False
        data = self.parent.to_dict()
        expected = {name: value['path'] for name, value in data['assets'].items()}
        synthetic = data['environment']['runtime'].get('synthetic') is True
        if asset_paths is None:
            if not synthetic:
                raise CacheInvalid('durable caches require actual loaded model asset paths')
        elif (not isinstance(asset_paths, dict) or set(asset_paths) != set(expected) or
              any(str(Path(asset_paths[name]).resolve()) != path for name, path in expected.items())):
            raise CacheInvalid('durable cache producer differs from fingerprinted model assets')
        self.validate_context()
        self.bound = True

    def _require_bound(self):
        if not self.bound:
            raise CacheInvalid('durable cache access requires a bound shared session')
        self.store._require_owner()
        if self.check is not None:
            self.check()
