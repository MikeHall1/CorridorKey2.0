"""Legacy download flags retain one immutable revision for paired model assets."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import download_weights


class DownloadRevisionTests(unittest.TestCase):
    def test_mutable_revision_resolved_once_for_both_assets(self):
        for revision in (None, 'main', 'release'):
            with (self.subTest(revision=revision), tempfile.TemporaryDirectory() as root,
                  patch.object(download_weights, 'HfApi', create=True) as api,
                  patch.object(download_weights, 'hf_hub_download') as download):
                api.return_value.model_info.return_value = SimpleNamespace(sha='a' * 40)
                download_weights.download_corridorkey(Path(root), 'publisher/model', revision)
                api.return_value.model_info.assert_called_once_with(
                    'publisher/model', revision=revision, timeout=30, token=False)
                self.assertEqual(download.call_count, 2)
                self.assertEqual({call.kwargs['revision'] for call in download.call_args_list}, {'a' * 40})

    def test_explicit_immutable_revision_does_not_resolve_head(self):
        with (tempfile.TemporaryDirectory() as root,
              patch.object(download_weights, 'HfApi', create=True) as api,
              patch.object(download_weights, 'hf_hub_download') as download):
            download_weights.download_corridorkey(Path(root), 'publisher/model', 'b' * 40)
            api.assert_not_called()
            self.assertEqual({call.kwargs['revision'] for call in download.call_args_list}, {'b' * 40})

    def test_unpinned_metadata_refused_before_any_download(self):
        with (tempfile.TemporaryDirectory() as root,
              patch.object(download_weights, 'HfApi', create=True) as api,
              patch.object(download_weights, 'hf_hub_download') as download):
            api.return_value.model_info.return_value = SimpleNamespace(sha='main')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                download_weights.download_corridorkey(Path(root), 'publisher/model', None)
            download.assert_not_called()
