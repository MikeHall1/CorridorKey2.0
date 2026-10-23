from pathlib import Path
import tempfile
import unittest

from corridorkey.corrections import (CorrectionDocument, CorrectionHistory, invalidated_ranges,
                                     load_corrections, save_corrections)
from corridorkey.identities import file_sha256


def document_data():
    return {"schema_version": 1, "revision": 0,
            "source": {"sha256": "a" * 64, "start_frame": 0, "frame_count": 10,
                       "fps": 24, "width": 2, "height": 2},
            "shots": [[0, 5], [5, 10]], "entries": []}


def raster_entry(frame=3, kind="matte", identifier="first"):
    return {"id": identifier, "revision": 0, "frame": frame, "kind": kind,
            "asset": {"path": "mask.png", "sha256": "b" * 64},
            "strength": 1, "direction": "both", "range": [frame, frame + 1]}


class CorrectionFormatTests(unittest.TestCase):
    def test_canonical_snapshot_identity_and_order_are_independent_of_path_location(self):
        data = document_data()
        data["entries"] = [raster_entry(), raster_entry(kind="remove", identifier="second")]
        document = CorrectionDocument.parse(data)
        data["entries"][0]["strength"] = 0
        self.assertEqual(document.to_dict()["entries"][0]["strength"], 1)
        relocated = document.to_dict()
        relocated["entries"][0]["asset"]["path"] = "/other/location.png"
        self.assertEqual(document.identity, CorrectionDocument.parse(relocated).identity)
        relocated["entries"].reverse()
        self.assertNotEqual(document.identity, CorrectionDocument.parse(relocated).identity)

    def test_malformed_timeline_unknown_semantics_and_unsafe_values_are_rejected(self):
        mutations = [lambda d: d.update(schema_version=True), lambda d: d.update(extra=True),
                     lambda d: d.update(shots=[[0, 3], [4, 10]]),
                     lambda d: d["source"].update(width=True),
                     lambda d: d["entries"][0].update(kind="trimap"),
                     lambda d: d["entries"][0].update(strength=float("nan")),
                     lambda d: d["entries"][0].update(range=[3, 7]),
                     lambda d: d["entries"][0].update(direction="forward", range=[2, 4]),
                     lambda d: d["entries"][0]["asset"].update(path="https://example.com/mask.png")]
        for mutate in mutations:
            data = document_data()
            data["entries"] = [raster_entry()]
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                CorrectionDocument.parse(data)

    def test_reopen_verify_assets_and_detect_changed_source_or_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, mask = root / "source.mp4", root / "mask.png"
            source.write_bytes(b"source")
            mask.write_bytes(b"asset")
            data = document_data()
            data["source"]["sha256"] = file_sha256(source)
            data["entries"] = [raster_entry()]
            data["entries"][0]["asset"]["sha256"] = file_sha256(mask)
            document = CorrectionDocument.parse(data, root)
            destination = root / "project" / "corrections.json"
            save_corrections(destination, document)
            loaded = load_corrections(destination, source=source)
            self.assertEqual(loaded.identity, document.identity)
            mask.write_bytes(b"changed asset")
            with self.assertRaisesRegex(ValueError, "asset hash"):
                load_corrections(destination, source=source)
            mask.write_bytes(b"asset")
            source.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "source content"):
                load_corrections(destination, source=source)

    def test_revision_history_undo_redo_and_whole_shot_invalidation(self):
        original = document_data()
        first = CorrectionDocument.parse(original)
        original.update(revision=1, entries=[raster_entry()])
        second = CorrectionDocument.parse(original)
        history = CorrectionHistory(first)
        self.assertEqual(history.apply(second), [(0, 5)])
        self.assertEqual(history.undo(), [(0, 5)])
        self.assertEqual(history.current.identity, first.identity)
        self.assertEqual(history.redo(), [(0, 5)])
        with self.assertRaises(ValueError):
            history.apply(first)
        third = second.to_dict()
        third.update(revision=2)
        third["entries"].append(raster_entry(7, "keep", "third"))
        self.assertEqual(history.apply(CorrectionDocument.parse(third)), [(5, 10)])

    def test_removal_reordering_reset_and_timeline_changes_invalidate_dependencies(self):
        data = document_data()
        data["entries"] = [raster_entry(), raster_entry(7, "keep", "second")]
        before = CorrectionDocument.parse(data)
        removed = before.to_dict()
        removed["entries"].pop(0)
        self.assertEqual(invalidated_ranges(before, CorrectionDocument.parse(removed)), [(0, 5)])
        reordered = before.to_dict()
        reordered["entries"].reverse()
        self.assertEqual(invalidated_ranges(before, CorrectionDocument.parse(reordered)), [(0, 5), (5, 10)])
        reset = document_data()
        reset["entries"] = [{"id": "reset", "revision": 0, "frame": 3, "kind": "reset"}]
        document = CorrectionDocument.parse(reset)
        self.assertEqual(document.resets, (3,))
        self.assertEqual(document.frames, (3,))
        changed = document.to_dict()
        changed["source"]["fps"] = 25
        self.assertEqual(invalidated_ranges(document, CorrectionDocument.parse(changed)), [(0, 10)])
