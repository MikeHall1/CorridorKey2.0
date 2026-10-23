__all__ = ["CorridorKeyV2"]


def __getattr__(name):
    # Configuration, diagnostics and CPU correctness checks need no model stack.
    if name == "CorridorKeyV2":
        from .model import CorridorKeyV3

        class CorridorKeyV2(CorridorKeyV3):
            """Public CorridorKey v2 runtime alias for the exported model."""

        CorridorKeyV2.__qualname__ = "CorridorKeyV2"
        globals()[name] = CorridorKeyV2
        return CorridorKeyV2
    raise AttributeError(name)
