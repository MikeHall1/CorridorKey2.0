"""Synthetic CPU heuristic mechanics; no production quality/model evidence."""
import hashlib
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Any

import cv2
import numpy as np

from corridorkey.clean_plate import (
    INPUT_CONTRACT,
    CleanPlateProvenance,
    CleanPlateSettings,
    PlateMapping,
    assist_clean_plate,
)
from corridorkey.control import CancellationToken, CancelledError


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class CleanPlateTests(unittest.TestCase):
    def provenance(self, frame=10):
        return CleanPlateProvenance(sha('source'), sha('plate'), sha('source-decode'), sha('plate-decode'), frame, 0,
                                    PlateMapping('single', 10, 3, 0, 1), INPUT_CONTRACT)

    def inputs(self, *, textured=False):
        height, width = 96, 128
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        plate = np.empty((height, width, 3), np.float32)
        plate[..., 0] = .06 + xx / width * .03
        plate[..., 1] = .35 + xx / width * .2 + yy / height * .05
        plate[..., 2] = .09 + yy / height * .03
        if textured:
            rng = np.random.default_rng(881)
            noise = cv2.GaussianBlur(rng.normal(0, .15, (height, width)).astype(np.float32), (3, 3), 0)
            plate += noise[..., None]
        samples = np.zeros((height, width), np.uint8)
        samples[12:-12, 12:-12] = 1
        samples[30:66, 42:86] = 0
        return plate.copy(), plate, samples

    def test_declared_aligned_plate_spatial_field_and_foreground_are_rough_native_classes(self):
        source, plate, samples = self.inputs()
        source[38:58, 50:78] = [.8, .03, .02]
        before = (source.tobytes(), plate.tobytes(), samples.tobytes())
        result = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        self.assertTrue(result.usable)
        self.assertEqual(int(result.labels()[48, 64]), 255)
        self.assertEqual(int(result.labels()[24, 24]), 0)
        self.assertEqual(int(result.labels()[38, 50]), 128)  # eroded seed boundary
        self.assertGreater(float(result.screen_field()[48, 110, 1]), float(result.screen_field()[48, 15, 1]))
        self.assertTrue(set(np.unique(result.labels())).issubset({0, 128, 255}))
        self.assertEqual(result.guidance_bytes(), result.trimap_bytes)
        self.assertEqual((source.tobytes(), plate.tobytes(), samples.tobytes()), before)
        report = result.report()
        self.assertEqual(report['diagnostics']['actual_inference'], 'NOT PERFORMED')
        self.assertEqual(report['diagnostics']['quality_benefit'], 'NOT VERIFIED')
        self.assertIn('not confidence', report['diagnostics']['residual'])

    def test_estimated_actual_translation_reports_direction_and_preserves_unknown_borders(self):
        _, plate, samples = self.inputs(textured=True)
        source = cv2.warpAffine(plate, np.array([[1., 0., 4.], [0., 1., -3.]], np.float32),
                                (plate.shape[1], plate.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32, copy=False)
        source[38:58, 50:78] = [.8, .03, .02]
        result = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('estimate_translation'), provenance=self.provenance())
        self.assertTrue(result.usable, result.report())
        shift = result.report()['diagnostics']['alignment']['translation_reference_to_source_xy']
        self.assertAlmostEqual(shift[0], 4., delta=.4)
        self.assertAlmostEqual(shift[1], -3., delta=.4)
        self.assertEqual(int(result.labels()[48, 64]), 255)
        self.assertEqual(int(result.validity()[0, 0]), 0)
        self.assertEqual(int(result.labels()[0, 0]), 128)
        self.assertIn('mask-edge', result.report()['diagnostics']['alignment']['estimator_limits'])
        refused = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('estimate_translation', max_translation_pixels=2.), provenance=self.provenance())
        self.assertFalse(refused.usable)
        self.assertIn('exceeds', refused.report()['diagnostics']['reason'])

    def test_exposure_compensation_is_explicit_scalar_bounded_and_analysis_only(self):
        _, plate, samples = self.inputs()
        source = plate * np.float32(1.5)
        before = plate.tobytes()
        off = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        self.assertFalse(off.usable)
        on = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared', compensate_exposure=True), provenance=self.provenance())
        self.assertTrue(on.usable)
        self.assertAlmostEqual(on.report()['diagnostics']['exposure']['gain'], 1.5, places=5)
        self.assertEqual(int(on.labels()[24, 24]), 0)
        self.assertEqual(plate.tobytes(), before)
        refused = assist_clean_plate(source, plate, samples,
                                     settings=CleanPlateSettings('declared', compensate_exposure=True, max_gain=1.2), provenance=self.provenance())
        self.assertFalse(refused.usable)
        self.assertIn('not clamped', refused.report()['diagnostics']['reason'])

    def test_shadows_and_similar_screen_chroma_remain_unknown_not_foreground_claims(self):
        source, plate, samples = self.inputs()
        source[38:58, 50:62] *= .45
        source[38:58, 64:78] *= .88
        result = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        self.assertTrue(result.usable)
        self.assertEqual(int(result.labels()[48, 55]), 128)
        self.assertEqual(int(result.labels()[48, 70]), 128)
        self.assertGreater(result.report()['diagnostics']['shadow_or_similar_unknown_pixels'], 0)
        self.assertIn('indistinguishable', result.report()['diagnostics']['warnings'][0])

    def test_misaligned_low_texture_and_excess_shift_produce_no_acceptable_guidance(self):
        _, plate, samples = self.inputs(textured=True)
        source = cv2.warpAffine(plate, np.array([[1., 0., 15.], [0., 1., 7.]], np.float32),
                                (plate.shape[1], plate.shape[0]), flags=cv2.INTER_LINEAR).astype(np.float32, copy=False)
        for settings, image, reference in (
            (CleanPlateSettings('declared', sample_error_threshold=.04, min_sample_match_fraction=.95), source, plate),
            (CleanPlateSettings('estimate_translation'), np.full_like(plate, .3), np.full_like(plate, .3)),
            (CleanPlateSettings('declared', translation_xy=(20., 0.), min_valid_fraction=.99), source, plate),
        ):
            with self.subTest(settings=settings):
                result = assist_clean_plate(image, reference, samples, settings=settings, provenance=self.provenance())
                self.assertFalse(result.usable)
                self.assertTrue(np.all(result.labels() == 128))
                self.assertFalse(result.validity().any())
                with self.assertRaisesRegex(ValueError, 'cannot be accepted'):
                    result.guidance_bytes()

    def test_immutable_output_bytes_and_versioned_content_identity_include_all_dependencies(self):
        source, plate, samples = self.inputs()
        settings = CleanPlateSettings('declared')
        first = assist_clean_plate(source, plate, samples, settings=settings, provenance=self.provenance())
        again = assist_clean_plate(source, plate, samples, settings=settings, provenance=self.provenance())
        self.assertEqual(first.identity, again.identity)
        for view in (first.labels(), first.screen_field(), first.residual(), first.validity()):
            self.assertFalse(view.flags.writeable)
            with self.assertRaises(ValueError):
                view.setflags(write=True)
        invalid_mutation: Any = first  # exercise runtime immutability deliberately
        with self.assertRaises(FrozenInstanceError):
            invalid_mutation.usable = False
        changed = plate.copy()
        changed[20, 20, 0] += .01
        other = assist_clean_plate(source, changed, samples, settings=settings, provenance=self.provenance())
        self.assertNotEqual(first.identity, other.identity)
        other = assist_clean_plate(source, plate, samples, settings=replace(settings, screen_radius=0), provenance=self.provenance())
        self.assertNotEqual(first.identity, other.identity)
        other = assist_clean_plate(source, plate, samples, settings=settings, provenance=replace(self.provenance(), reference_decode_identity=sha('changed-decode')))
        self.assertNotEqual(first.identity, other.identity)
        first.report()['diagnostics']['warnings'].append('outside mutation')
        self.assertEqual(len(first.report()['diagnostics']['warnings']), 2)

    def test_mapping_requires_explicit_complete_correspondence_and_exact_frame_indices(self):
        mapping = PlateMapping('sequence', 10, 3, 12, 30, -2)
        self.assertEqual([mapping.reference_frame(f) for f in (10, 11, 12)], [10, 11, 12])
        single = PlateMapping('single', 10, 3, 4, 20)
        self.assertEqual(single.reference_frame(12), 4)
        for frame in (9, 13, True):
            with self.assertRaises(ValueError):
                mapping.reference_frame(frame)
        for invalid in (PlateMapping('sequence', 0, 4, 0, 3), PlateMapping('sequence', 0, 301, 0, 500), PlateMapping('single', 0, 1, 0, 1, -1), PlateMapping('single', 2**31 - 1, 2, 0, 1)):
            with self.assertRaises(ValueError):
                invalid.validate()
        source, plate, samples = self.inputs()
        with self.assertRaisesRegex(ValueError, 'reference frame differs'):
            assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=replace(self.provenance(), reference_frame=1))

    def test_native_declaration_budgets_samples_and_nonfinite_data_are_strict(self):
        source, plate, samples = self.inputs()
        for settings in (CleanPlateSettings('declared', max_pixels=10), CleanPlateSettings('declared', max_working_bytes=1000)):
            with self.assertRaisesRegex(ValueError, 'budget'):
                assist_clean_plate(source, plate, samples, settings=settings, provenance=self.provenance())
        with self.assertRaisesRegex(ValueError, 'explicit straight'):
            assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=replace(self.provenance(), input_contract='display_srgb'))
        bad = samples.copy()
        bad[5, 5] = 255
        with self.assertRaisesRegex(ValueError, 'exact binary'):
            assist_clean_plate(source, plate, bad, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        bad_rgb = source.copy()
        bad_rgb[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            assist_clean_plate(bad_rgb, plate, samples, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        invalid_instance: Any = CleanPlateSettings
        with self.assertRaises(TypeError):
            assist_clean_plate(source, plate, samples, settings=invalid_instance, provenance=self.provenance())

    def test_cancellation_before_and_during_analysis_does_not_modify_inputs(self):
        source, plate, samples = self.inputs(textured=True)
        before = source.tobytes(), plate.tobytes(), samples.tobytes()
        for cancel_at in (1, 5, 8):
            token = CancellationToken()
            calls = [0]

            def check(token=token, calls=calls, cancel_at=cancel_at):
                calls[0] += 1
                if calls[0] == cancel_at:
                    token.cancel('test-only clean-plate cancellation')
                token.check()

            with self.assertRaises(CancelledError):
                assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=self.provenance(), check=check)
        self.assertEqual((source.tobytes(), plate.tobytes(), samples.tobytes()), before)

    def test_signed_linear_input_is_preserved_and_analysis_overflow_is_unusable(self):
        source, plate, samples = self.inputs()
        source[..., 0] = plate[..., 0] = -.05
        source_before, plate_before = source.tobytes(), plate.tobytes()
        result = assist_clean_plate(source, plate, samples, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        self.assertTrue(result.usable)
        self.assertLess(float(result.screen_field()[24, 24, 0]), 0.)
        self.assertEqual((source.tobytes(), plate.tobytes()), (source_before, plate_before))
        huge = np.full_like(source, 3e38)
        refused = assist_clean_plate(huge, np.full_like(source, .3), samples, settings=CleanPlateSettings('declared'), provenance=self.provenance())
        self.assertFalse(refused.usable)
        self.assertIn('overflowed', refused.report()['diagnostics']['reason'])
        with self.assertRaises(ValueError):
            refused.guidance_bytes()

    def test_filtered_support_and_empty_eroded_seeds_cannot_be_accepted(self):
        for side, settings, reason in (
            (32, CleanPlateSettings('declared', translation_xy=(2., 0.), screen_radius=15), 'screen-field support'),
            (4, CleanPlateSettings('declared', min_sample_pixels=4, seed_erosion=2), 'No conservative'),
        ):
            image = np.full((side, side, 3), .3, np.float32)
            result = assist_clean_plate(image, image, np.ones((side, side), np.uint8), settings=settings, provenance=self.provenance())
            self.assertFalse(result.usable)
            self.assertIn(reason, result.report()['diagnostics']['reason'])
            with self.assertRaises(ValueError):
                result.guidance_bytes()

    def test_estimated_texture_overflow_is_an_unusable_json_safe_report(self):
        image = np.full((16, 16, 3), .3, np.float32)
        image[::2] = 3e38
        result = assist_clean_plate(image, image, np.ones((16, 16), np.uint8), settings=CleanPlateSettings('estimate_translation'), provenance=self.provenance())
        self.assertFalse(result.usable)
        self.assertIn('overflowed', result.report()['diagnostics']['reason'])
        self.assertIsNone(result.report()['diagnostics']['alignment']['reference_texture_std'])

    def test_exposure_overflow_remains_an_unusable_json_safe_report(self):
        source = np.full((16, 16, 3), 3e38, np.float32)
        reference = np.full_like(source, .00002)
        result = assist_clean_plate(source, reference, np.ones((16, 16), np.uint8), settings=CleanPlateSettings('declared', compensate_exposure=True), provenance=self.provenance())
        self.assertFalse(result.usable)
        self.assertIsNone(result.report()['diagnostics']['exposure']['gain'])
        self.assertIn('not clamped', result.report()['diagnostics']['reason'])

    def test_matching_extreme_values_cannot_hide_norm_overflow_as_zero_residual(self):
        image = np.full((16, 16, 3), 2e38, np.float32)
        result = assist_clean_plate(image, image, np.ones((16, 16), np.uint8), settings=CleanPlateSettings('declared', screen_radius=0), provenance=self.provenance())
        self.assertFalse(result.usable)
        self.assertIn('overflowed', result.report()['diagnostics']['reason'])
        self.assertTrue(np.all(result.labels() == 128))
