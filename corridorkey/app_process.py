"""Explicit child entry points for source and frozen desktop applications."""
from __future__ import annotations

import sys

MODULES = frozenset({'corridorkey.desktop', 'corridorkey.worker', 'corridorkey.batch',
                     'corridorkey.benchmark', 'corridorkey.benchmark_pair', 'corridorkey.backend_probe',
                     'corridorkey.ofx_cache', 'corridorkey.cache_maintenance', 'corridorkey.cache_backup',
                     'corridorkey.queue_transfer', 'corridorkey.asset_audit', 'corridorkey.asset_download',
                     'corridorkey.evaluation_dossier', 'corridorkey.stress_suite'})


def module_command(module: str, *arguments: str) -> list[str]:
    if module not in MODULES and module != 'unittest':
        raise ValueError('Unsupported application child module.')
    if getattr(sys, 'frozen', False):
        if module == 'unittest':
            raise ValueError('CPU source correctness suites require a source checkout and Python; unavailable in this bundle.')
        return [sys.executable, '--corridorkey-module', module, *arguments]
    return [sys.executable, '-m', module, *arguments]


def main(argv: list[str] | None = None) -> None:
    import runpy

    arguments = list(sys.argv[1:] if argv is None else argv)
    module = 'corridorkey.desktop'
    if arguments[:1] == ['--corridorkey-module']:
        if len(arguments) < 2 or arguments[1] not in MODULES:
            raise SystemExit('Unsupported CorridorKey application module.')
        module, arguments = arguments[1], arguments[2:]
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
