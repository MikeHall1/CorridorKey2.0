"""Optional Qt Widgets frontend using the shared supervised engine worker."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def first_run_readiness(window, preferences, schedule, *, force=False, enabled=True) -> bool:
    """Present the introduction only; never check, load, download or render."""
    if not enabled or (not force and preferences.value('readiness/introduction_seen', False, type=bool)):
        return False

    def present():
        if not window.closing:
            window.readiness.show()
            preferences.setValue('readiness/introduction_seen', True)

    schedule(0, present)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='CorridorKey desktop (optional Qt Widgets interface)')
    parser.add_argument('--project', type=Path, help='Open a saved local desktop project; never auto-render.')
    introduction = parser.add_mutually_exclusive_group()
    introduction.add_argument('--readiness', action='store_true', help='Open the readiness introduction on startup without running checks.')
    introduction.add_argument('--no-first-run', action='store_true', help='Suppress the first-launch introduction for this invocation.')
    args = parser.parse_args(argv)
    try:
        from PySide6.QtCore import QSettings, QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        parser.exit(2, 'Desktop requires requirements-desktop.txt in the selected Python environment.\n')
    from .desktop_window import DesktopWindow

    app = QApplication(sys.argv[:1])
    window = DesktopWindow()
    if args.project:
        window.open_project(args.project)
    window.show()
    first_run_readiness(window, QSettings('MikeHall1', 'CorridorKey2'), QTimer.singleShot,
                        force=args.readiness, enabled=not args.no_first_run)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
