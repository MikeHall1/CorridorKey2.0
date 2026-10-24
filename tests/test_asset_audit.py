import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from corridorkey import asset_audit as audit
from corridorkey.asset_inventory import AssetSelection
from corridorkey.control import CancelledError


def safetensors(path, tensors=None, *, metadata=None):
    tensors = tensors if tensors is not None else {'weight': ('F32', [1])}
    header = {}
    payload = b''
    for name, (dtype, shape) in tensors.items():
        length = audit.DTYPES[dtype]
        for value in shape:
            length *= value
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(payload), len(payload) + length]}
        payload += b'\0' * length
    if metadata is not None:
        header['__metadata__'] = metadata
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


def npy(shape=(1,), dtype='<f4'):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode()
    header += b' ' * ((-(10 + len(header) + 1)) % 64) + b'\n'
    length = int(dtype[2:]) if len(dtype) > 2 and dtype[2:].isdigit() else 1
    for value in shape:
        length *= value
    return b'\x93NUMPY\1\0' + struct.pack('<H', len(header)) + header + b'\0' * length


def valid_assets(root):
    safetensors(root / 'weights/corridorkey_v2.safetensors', metadata={
        'format': 'corridorkey_v2_runtime_checkpoint', 'model_config': '{}', 'metadata': '{}'})
    safetensors(root / 'weights/foundation_stats.safetensors', {name: ('F32', [2]) for name in audit.STATS})
    rvm = root / 'vfms/rvm/pretrain_rvm_small16_256_204031069.npz'
    rvm.parent.mkdir(parents=True)
    with zipfile.ZipFile(rvm, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('tokenizer/patch_embedding/Conv_0/kernel.npy', npy((1, 2)))
        archive.writestr('tokenizer/patch_embedding/Conv_0/bias.npy', npy((2,)))
    moge = root / 'vfms/moge/model.pt'
    moge.parent.mkdir(parents=True)
    moge.write_bytes(b'\x80\x04opaque pickle must never execute')
    cradio = root / 'vfms/cradio/C-RADIOv4-SO400M'
    cradio.mkdir(parents=True)
    (cradio / 'hf_model.py').write_text("raise RuntimeError('AUDIT MUST NEVER EXECUTE LOCAL MODEL CODE')\n")
    (cradio / 'config.json').write_text('{"model_type":"radio"}')
    safetensors(cradio / 'model.safetensors')
    return AssetSelection(root)


class AssetAuditTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.selection = valid_assets(self.root / 'assets')

    def entries(self, report):
        return {entry['asset']: entry for entry in report['entries']}

    def test_actual_small_formats_are_checked_without_models_or_tensor_loading(self):
        before = {str(path): (path.stat().st_mtime_ns, path.read_bytes())
                  for path in self.selection.root.rglob('*') if path.is_file()}
        progress = []
        report = audit.audit_assets(self.selection, progress=progress.append)
        self.assertEqual(report['status'], 'complete')
        self.assertTrue(report['assets_present'])
        entries = self.entries(report)
        for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'cradio_dir'):
            self.assertEqual(entries[name]['format_status'], 'PASS', entries[name])
            self.assertEqual(len(entries[name]['sha256']), 64)
        self.assertEqual(entries['moge_checkpoint']['format_status'], 'UNVERIFIED')
        self.assertEqual(report['model_compatibility'], 'NOT VERIFIED')
        self.assertIn('NOT VERIFIED', entries['rvm_ckpt']['details']['payload_crc_and_values'])
        self.assertEqual(entries['rvm_ckpt']['details']['tensor_count'], 2)
        self.assertTrue(any(event.stage == 'asset_hash' for event in progress))
        self.assertEqual({str(path): (path.stat().st_mtime_ns, path.read_bytes())
                          for path in self.selection.root.rglob('*') if path.is_file()}, before)
        code = "import sys; from pathlib import Path; from corridorkey.asset_audit import audit_assets,AssetSelection; r=audit_assets(AssetSelection(Path(sys.argv[1]))); print(r['status'],'torch' in sys.modules,'numpy' in sys.modules)"
        result = subprocess.run([sys.executable, '-c', code, str(self.selection.root)], capture_output=True,
                                text=True, check=True, timeout=10)
        self.assertEqual(result.stdout.strip(), 'complete False False')

    def test_invalid_runtime_metadata_offsets_headers_and_stats_are_honest(self):
        path = self.selection.root / 'weights/corridorkey_v2.safetensors'
        safetensors(path, metadata={'format': 'wrong'})
        report = audit.audit_assets(self.selection)
        self.assertEqual(self.entries(report)['weights']['format_status'], 'FAIL')
        self.assertIsNotNone(self.entries(report)['weights']['identity'])
        safetensors(path, metadata={'format': 'corridorkey_v2_runtime_checkpoint'})
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['weights']['format_status'], 'UNVERIFIED')
        safetensors(self.selection.root / 'weights/foundation_stats.safetensors', {'moge_mean': ('F32', [1, 2])})
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['foundation_stats']['format_status'], 'UNVERIFIED')
        path.write_bytes(struct.pack('<Q', audit.MAX_HEADER_BYTES + 1))
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['weights']['format_status'], 'FAIL')

    def test_legacy_pt_fallback_is_hashed_and_unverified_without_deserialization(self):
        weights = self.selection.root / 'weights/corridorkey_v2.safetensors'
        weights.unlink()
        legacy = weights.with_suffix('.pt')
        legacy.write_bytes(b'\x80\x04dangerous pickle payload must stay opaque')
        entry = self.entries(audit.audit_assets(self.selection))['weights']
        self.assertEqual(entry['path'], str(legacy.absolute()))
        self.assertEqual(entry['format_status'], 'UNVERIFIED')
        self.assertEqual(len(entry['sha256']), 64)

    def test_npz_object_traversal_duplicate_extent_and_zipbomb_are_rejected(self):
        path = self.selection.root / 'vfms/rvm/pretrain_rvm_small16_256_204031069.npz'
        for case in ('object', 'traversal', 'duplicate', 'extent', 'bomb'):
            with self.subTest(case=case):
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                    name = '../outside.npy' if case == 'traversal' else 'encoder/kernel.npy'
                    raw = npy(dtype='|O1') if case == 'object' else npy()
                    if case == 'extent':
                        raw += b'wrong-extra-bytes'
                    if case == 'bomb':
                        raw = npy((1000000,), '|u1')
                    archive.writestr(name, raw)
                    if case == 'duplicate':
                        archive.writestr('encoder/kernel/child.npy', npy())
                with patch.object(audit, 'MAX_ZIP_EXPANDED', 1000000):
                    entry = self.entries(audit.audit_assets(self.selection))['rvm_ckpt']
                self.assertEqual(entry['format_status'], 'FAIL', entry)

    def test_symlinked_hf_blob_is_supported_with_lexical_and_resolved_identity(self):
        path = self.selection.root / 'weights/corridorkey_v2.safetensors'
        blob = self.root / 'hf-cache-blob'
        path.rename(blob)
        path.symlink_to(blob)
        entry = self.entries(audit.audit_assets(self.selection))['weights']
        self.assertEqual(entry['format_status'], 'PASS')
        self.assertEqual(entry['identity']['path'], str(path.absolute()))
        self.assertEqual(entry['identity']['resolved_path'], str(blob.resolve()))
        self.assertNotEqual(entry['identity']['lexical_stat']['mode'], entry['identity']['target_stat']['mode'])

    def test_maximum_comment_zip64_locator_is_rejected_before_zipfile_allocation(self):
        path = self.selection.root / 'vfms/rvm/pretrain_rvm_small16_256_204031069.npz'
        raw = path.read_bytes()
        ending = bytearray(raw[-22:])
        ending[-2:] = struct.pack('<H', 65535)
        path.write_bytes(raw[:-22] + b'PK\x06\x07' + b'\0' * 16 + ending + b'x' * 65535)
        with (patch.object(audit.zipfile, 'ZipFile', side_effect=AssertionError('Unbounded parser must not open')) as constructor,
              self.assertRaisesRegex(ValueError, 'ZIP64 central')):
            audit._npz(path, path.stat().st_size, None)
        constructor.assert_not_called()

    def test_npz_constructor_uses_same_capped_descriptor_despite_path_replacement(self):
        self._check_npz_constructor_path_replacement(refuse_replacement=False)

    def test_npz_constructor_keeps_bounds_when_open_file_replacement_is_refused(self):
        self._check_npz_constructor_path_replacement(refuse_replacement=True)

    def _check_npz_constructor_path_replacement(self, *, refuse_replacement):
        path = self.selection.root / 'vfms/rvm/pretrain_rvm_small16_256_204031069.npz'
        original_bytes = path.read_bytes()
        constructor = audit.zipfile.ZipFile
        observed = []
        replaced = []

        def opening(stream):
            self.assertIsInstance(stream, audit._BoundedAssetFile)
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(b'invalid changed path after pre-admission')
            try:
                replacement.replace(path)
            except PermissionError:
                # Windows can refuse replacement while the admitted FD is open.
                self.assertEqual(path.read_bytes(), original_bytes)
                replacement.unlink()
                replaced.append(False)
            else:
                replaced.append(True)
            with self.assertRaisesRegex(ValueError, 'allocation'):
                stream.read(audit.MAX_HEADER_BYTES + 1)
            with self.assertRaisesRegex(ValueError, 'range'):
                stream.seek(len(original_bytes) + 1)
            observed.append(True)
            return constructor(stream)

        with patch.object(audit.zipfile, 'ZipFile', opening):
            if refuse_replacement:
                with patch.object(Path, 'replace', side_effect=PermissionError('controlled open-file sharing refusal')):
                    status, _ = audit._npz(path, len(original_bytes), None)
            else:
                status, _ = audit._npz(path, len(original_bytes), None)
        self.assertEqual(status, 'PASS')
        self.assertEqual(observed, [True])
        self.assertEqual(len(replaced), 1)
        if refuse_replacement:
            self.assertEqual(replaced, [False])
        # A successful replacement is rejected; OS refusal preserves valid input.
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['rvm_ckpt']['format_status'],
                         'FAIL' if replaced[0] else 'PASS')

    def test_tree_frontier_is_capped_before_enqueuing_branches(self):
        tree = self.root / 'branching-tree'
        tree.mkdir()
        for first in range(3):
            for second in range(3):
                (tree / str(first) / str(second)).mkdir(parents=True)
        original = os.scandir
        visited = []

        def scanning(path):
            visited.append(Path(path))
            return original(path)

        with (patch.object(audit, 'MAX_TREE_FILES', 4), patch.object(audit.os, 'scandir', scanning),
              self.assertRaisesRegex(ValueError, 'frontier')):
            audit._tree(tree, None, None)
        self.assertEqual(len(visited), 2)

    def test_tree_cycle_missing_loader_and_shard_inventory_refuse(self):
        tree = self.selection.root / 'vfms/cradio/C-RADIOv4-SO400M'
        (tree / 'cycle').symlink_to(tree, target_is_directory=True)
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['cradio_dir']['format_status'], 'FAIL')
        (tree / 'cycle').unlink()
        (tree / 'model.safetensors.index.json').write_text('{"weight_map":{"x":"missing.safetensors"}}')
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['cradio_dir']['format_status'], 'FAIL')
        (tree / 'model.safetensors.index.json').unlink()
        (tree / 'hf_model.py').unlink()
        self.assertEqual(self.entries(audit.audit_assets(self.selection))['cradio_dir']['format_status'], 'FAIL')

    def test_cancel_and_changed_or_growing_assets_never_report_verified_hash(self):
        def check():
            raise CancelledError('fixture cancelled')

        self.assertEqual(audit.audit_assets(self.selection, check=check)['status'], 'cancelled')
        path = self.selection.root / 'weights/corridorkey_v2.safetensors'
        identity = audit._identity(path)
        changed = []

        def grow():
            if not changed:
                with path.open('ab') as stream:
                    stream.write(b'extra bytes')
                changed.append(True)

        with self.assertRaisesRegex(ValueError, 'grew beyond'):
            audit._hash(path, identity, grow, None)

    def test_export_exact_fresh_report_stale_and_overwrite_refusal(self):
        report = audit.audit_assets(self.selection)
        output = self.root / 'audit.json'
        exported = audit.export_audit(report, output)
        self.assertEqual(exported['status'], 'complete', exported)
        self.assertEqual(output.read_bytes(), audit._canonical(report))
        refused = audit.export_audit(report, output)
        self.assertEqual(refused['status'], 'failed')
        self.assertFalse(refused['usable'])
        (self.selection.root / 'vfms/moge/model.pt').write_bytes(b'changed opaque checkpoint')
        stale = audit.export_audit(report, self.root / 'stale.json')
        self.assertEqual(stale['status'], 'failed')
        self.assertIn('stale', stale['error'])
        self.assertFalse((self.root / 'stale.json').exists())

    def test_export_late_alias_cleanup_failure_stays_usable_without_asset_changes(self):
        report = audit.audit_assets(self.selection)
        output = self.root / 'audit.json'
        original = Path.unlink

        def failed(path, *args, **kwargs):
            if path.name.startswith('.asset-audit-'):
                raise PermissionError('fixture alias cleanup denied')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'unlink', failed):
            result = audit.export_audit(report, output)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['usable'])
        self.assertIsNotNone(result['partial_path'])
        self.assertEqual(output.read_bytes(), audit._canonical(report))
        self.assertEqual(audit.audit_assets(self.selection)['token'], report['token'])

    def test_readable_malformed_tree_config_keeps_identity_and_exports_diagnostic(self):
        config = self.selection.root / 'vfms/cradio/C-RADIOv4-SO400M/config.json'
        config.write_bytes(b'{malformed JSON but readable complete file')
        report = audit.audit_assets(self.selection)
        entry = self.entries(report)['cradio_dir']
        self.assertEqual(entry['format_status'], 'FAIL')
        self.assertIsNotNone(entry['identity'])
        self.assertEqual(len(entry['sha256']), 64)
        output = self.root / 'malformed-tree-report.json'
        self.assertTrue(audit.export_audit(report, output)['usable'])
        self.assertEqual(output.read_bytes(), audit._canonical(report))

    def test_readable_empty_checkpoint_keeps_empty_hash_and_exports_failed_diagnostic(self):
        checkpoint = self.selection.root / 'weights/corridorkey_v2.safetensors'
        checkpoint.write_bytes(b'')
        report = audit.audit_assets(self.selection)
        entry = self.entries(report)['weights']
        self.assertTrue(entry['present'])
        self.assertEqual(entry['bytes'], 0)
        self.assertEqual(entry['format_status'], 'FAIL')
        self.assertEqual(entry['sha256'], hashlib.sha256(b'').hexdigest())
        self.assertIsNotNone(entry['identity'])
        output = self.root / 'empty-checkpoint-report.json'
        self.assertTrue(audit.export_audit(report, output)['usable'])
        self.assertEqual(output.read_bytes(), audit._canonical(report))

    def test_export_revalidates_file_tree_missing_and_preference_after_stage_fsync(self):
        original = os.fsync
        for kind in ('file', 'tree-file', 'missing', 'preference'):
            with self.subTest(kind=kind):
                case_root = self.root / kind
                selection = valid_assets(case_root)
                checkpoint = case_root / 'weights/corridorkey_v2.safetensors'
                if kind == 'missing':
                    (case_root / 'vfms/moge/model.pt').unlink()
                if kind == 'preference':
                    checkpoint.unlink()
                    checkpoint.with_suffix('.pt').write_bytes(b'opaque legacy')
                report = audit.audit_assets(selection)
                changed = []

                def fsync(descriptor, case=kind, root=case_root, weights=checkpoint, changed=changed):
                    original(descriptor)
                    if not changed:
                        if case == 'file':
                            weights.write_bytes(weights.read_bytes() + b'changed')
                        elif case == 'tree-file':
                            (root / 'vfms/cradio/C-RADIOv4-SO400M/hf_model.py').write_text('# changed code, never executed')
                        elif case == 'missing':
                            (root / 'vfms/moge/model.pt').write_bytes(b'newly appeared')
                        else:
                            safetensors(weights, metadata={'format': 'corridorkey_v2_runtime_checkpoint'})
                        changed.append(True)

                output = self.root / f'{kind}-report.json'
                with patch.object(audit.os, 'fsync', fsync):
                    result = audit.export_audit(report, output)
                self.assertFalse(result['usable'], result)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(output.exists())
                self.assertIsNotNone(result['partial_path'])

    def test_export_into_audited_tree_refuses_and_cli_exports(self):
        report = audit.audit_assets(self.selection)
        tree = self.selection.root / 'vfms/cradio/C-RADIOv4-SO400M'
        self.assertFalse(audit.export_audit(report, tree / 'new-report.json')['usable'])
        self.assertFalse((tree / 'new-report.json').exists())
        output = self.root / 'cli-audit.json'
        result = subprocess.run([sys.executable, '-m', 'corridorkey.asset_audit', 'audit', '--asset-root', str(self.selection.root),
                                 '--output', str(output)], capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), json.loads(output.read_text()))


if __name__ == '__main__':
    unittest.main()
