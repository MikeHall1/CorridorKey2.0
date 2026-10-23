"""Explicit ownership for temporary mappings and prefetched source generators."""
from functools import wraps
from .lifecycle import cleanup_all


def close_mapping(mapping):
    callbacks = []
    if mapping.mode != "r":
        callbacks.append(mapping.flush)
    callbacks.append(mapping._mmap.close)
    cleanup_all(callbacks)


class ResourceScope:
    def __init__(self):
        self.callbacks = []

    def own(self, resource, close=None):
        self.callbacks.append(close if close is not None else resource.close)
        return resource

    def mapping(self, factory, *args, **kwargs):
        mapping = factory(*args, **kwargs)
        return self.own(mapping, lambda: close_mapping(mapping))

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        callbacks, self.callbacks = self.callbacks, []
        cleanup_all(reversed(callbacks), primary_error=error)


def scoped_resources(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ResourceScope() as resources:
            return function(*args, _resources=resources, **kwargs)
    return wrapped
