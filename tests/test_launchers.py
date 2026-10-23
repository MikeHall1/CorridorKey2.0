"""Source shell entrypoint behavior; no Windows execution or engine inference."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')


@unittest.skipUnless(BASH, 'requires Bash to execute the source shell launcher')
class LauncherTests(unittest.TestCase):
    def test_shell_preserves_child_status_prompt_and_exact_arguments_from_other_cwd(self):
        assert BASH is not None
        for child_status, terminal_input in ((37, '\n'), (0, '\n'), (0, '')):
            with self.subTest(child_status=child_status, terminal_input=terminal_input), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                launcher = root / 'CorridorKey_DRAG_CLIPS_HERE.sh'
                launcher.write_bytes((ROOT / launcher.name).read_bytes())
                fake_python = root / 'venv/bin/python'
                fake_python.parent.mkdir(parents=True)
                fake_python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > invocation.txt\n'
                                       f'exit {child_status}\n', encoding='utf-8')
                fake_python.chmod(0o700)
                unrelated = root / 'unrelated working directory'
                unrelated.mkdir()
                selected = ['clip with spaces.mov', 'second clip.mov']
                result = subprocess.run([BASH, str(launcher), *selected], cwd=unrelated,
                                        input=terminal_input, text=True, capture_output=True, timeout=5,
                                        check=False)
                self.assertEqual(result.returncode, child_status, result.stdout + result.stderr)
                self.assertEqual((root / 'invocation.txt').read_text().splitlines(),
                                 ['run_interactive.py', *selected])
                self.assertFalse((unrelated / 'invocation.txt').exists())
