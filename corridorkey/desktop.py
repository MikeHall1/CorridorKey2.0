"""Optional Qt Widgets frontend using the shared supervised engine worker."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='CorridorKey desktop (optional Qt Widgets interface)')
    parser.add_argument('--project', type=Path, help='Open a saved local desktop project; never auto-render.')
    args = parser.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        parser.exit(2, 'Desktop requires requirements-desktop.txt in the selected Python environment.\n')
    from .desktop_window import DesktopWindow

    app = QApplication(sys.argv[:1])
    window = DesktopWindow()
    if args.project:
        window.open_project(args.project)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
