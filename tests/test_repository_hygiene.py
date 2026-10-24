"""Publication policy keeps build inputs while excluding local work state."""
import posixpath
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PREFIXES = ('.codex/', '.serena/', '.omx/', '.omx-state-locks/', '.ralph/',
                  'docs/audits/', 'docs/checkpoints/', 'docs/overnight/', 'docs/environments/')
LOCAL_FILES = {'AGENTS.md', '.DS_Store', '.omx-state-locks.identity.json', 'scripts/ralph_loop.py', 'tests/test_ralph.py',
               'docs/acceptance.md', 'docs/audit-20261023.md', 'docs/baseline.md',
               'docs/checkpoint.md', 'docs/status.md', 'docs/development-brief.md',
               'docs/product-improvements-brief.md', 'docs/product-performance-brief.md',
               'docs/product-improvements.md', 'docs/ralph-loop.md',
               'docs/review-2026-10-23.md', 'docs/review-objective.md'}


def local_only(path):
    return (path.startswith(LOCAL_PREFIXES) or path in LOCAL_FILES
            or path.startswith('docs/') and path.endswith('-review.md'))


class RepositoryHygieneTests(unittest.TestCase):
    def ignored(self, paths):
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin', '-z'],
                                cwd=ROOT, input=b''.join(p.encode() + b'\0' for p in paths),
                                capture_output=True, check=False)
        self.assertIn(result.returncode, (0, 1), result.stderr.decode(errors='replace'))
        return set(result.stdout.decode().strip('\0').split('\0')) - {''}

    def test_local_tools_reports_models_and_environments_are_ignored(self):
        paths = ['AGENTS.md', '.codex/agents/executor.toml', '.codex/prompts/example.md',
                 '.serena/project.yml', '.omx/state/session.json', '.ralph/plan.json',
                 'docs/overnight/morning-handoff.md', 'docs/audits/evidence/raw.json',
                 'docs/checkpoints/milestone1.md', 'docs/progressive-previews-review.md',
                 'scripts/ralph_loop.py', 'tests/test_ralph.py', '.venv/pyvenv.cfg',
                 '.ruff_cache/result', '.DS_Store', '.vscode/settings.json',
                 'weights/downloaded-assets/model.safetensors',
                 'weights/downloaded-assets/publisher/model.py', 'vfms/rvm/model.npz']
        self.assertEqual(self.ignored(paths), set(paths))

    def test_program_plugin_qa_and_licences_remain_publishable(self):
        paths = ['corridorkey/runtime.py', 'native/ofx/CMakeLists.txt', 'tests/test_verify.py',
                 'tests/static-baseline.json', 'scripts/verify.py', '.github/workflows/cpu-checks.yml',
                 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/architecture.md',
                 'docs/sessions.md', 'third_party/OpenFX/LICENSE.md',
                 'weights/.gitkeep', 'vfms/rvm/.gitkeep']
        self.assertEqual(self.ignored(paths), set())

    def test_public_index_excludes_local_only_state(self):
        paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        self.assertEqual([p for p in paths if p and local_only(p)], [])

    def test_maintained_markdown_links_resolve_in_public_tree(self):
        indexed = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        public = {p for p in indexed if p and not local_only(p)}
        failures = []
        for file in sorted(p for p in public if p == 'README.md' or p.startswith('docs/') and p.endswith('.md')):
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', (ROOT / file).read_text()):
                link = link.split(' "')[0]
                if link.startswith(('http:', 'https:', 'mailto:', '#')):
                    continue
                target = posixpath.normpath(posixpath.join(posixpath.dirname(file), link.split('#')[0]))
                if target not in public and not any(p.startswith(target.rstrip('/') + '/') for p in public):
                    failures.append((file, link))
        self.assertEqual(failures, [])
