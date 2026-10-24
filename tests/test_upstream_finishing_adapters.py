"""Existing corruption/state/budget/cleanup regressions in the new typed namespace."""
import unittest
from typing import Any

from tests import test_durable_foundations as iid
from tests import test_durable_hann as hann
from tests import test_durable_rvm as rvm
from tests import test_rvm_prepass as prepass
from tests import test_stage2_finishing_identity as identities


def parent():
    from corridorkey.recovery_identity import upstream_finishing_context
    fixture = identities.Stage2FinishingIdentityTests()
    fixture.setUp()
    return upstream_finishing_context(fixture.identity(fixture.data))


@unittest.skipUnless(iid.HAS_STACK, "requires optional CPU adapters")
class UpstreamIidTests(unittest.TestCase):
    def setUp(self: Any):
        iid.DurableFeatureTests.setUp(self)
        self.parent = parent()

    adapter = iid.DurableFeatureTests.adapter
    extract = iid.DurableFeatureTests.extract
    window = iid.DurableFeatureTests.window
    test_partial_run_frames_survive_new_owner_and_only_missing_frames_evaluate = iid.DurableFeatureTests.test_partial_run_frames_survive_new_owner_and_only_missing_frames_evaluate
    test_bfloat16_proxy_bits_and_input_changes_have_distinct_keys = iid.DurableFeatureTests.test_bfloat16_proxy_bits_and_input_changes_have_distinct_keys
    test_corrupt_and_wrong_dtype_entries_are_recomputed_without_stale_values = iid.DurableFeatureTests.test_corrupt_and_wrong_dtype_entries_are_recomputed_without_stale_values
    test_loaded_tensors_are_detached_from_closed_maps_and_sealed_storage = iid.DurableFeatureTests.test_loaded_tensors_are_detached_from_closed_maps_and_sealed_storage
    test_binding_budget_and_invalid_producer_are_explicit = iid.DurableFeatureTests.test_binding_budget_and_invalid_producer_are_explicit
    test_shared_session_binds_before_runner_and_rejects_mismatched_assets = iid.DurableFeatureTests.test_shared_session_binds_before_runner_and_rejects_mismatched_assets
    test_cancelled_private_stage_never_seals_and_is_cleaned = iid.DurableFeatureTests.test_cancelled_private_stage_never_seals_and_is_cleaned
    test_tensor_write_disk_failure_preserves_primary_and_no_partial_entry = iid.DurableFeatureTests.test_tensor_write_disk_failure_preserves_primary_and_no_partial_entry


@unittest.skipUnless(rvm.HAS_STACK, "requires optional CPU adapters")
class UpstreamRvmTests(unittest.TestCase):
    def setUp(self: Any):
        rvm.DurableRvmTests.setUp(self)
        self.parent = parent()

    adapter = rvm.DurableRvmTests.adapter
    identity = rvm.DurableRvmTests.identity
    def test_new_context_zero_budget_drift_and_wrong_parent(self: Any):
        from corridorkey.durable_rvm import DurableRvmChunks
        from corridorkey.recovery_identity import UpstreamFinishingContext, canonical
        from corridorkey.recovery_store import BundleStore, CacheInvalid
        with BundleStore(self.root / 'zero', 0) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            self.assertFalse(adapter.put(identity, self.features, self.state))
            self.assertEqual(adapter.budget_rejected, 1)
            other_parent = UpstreamFinishingContext(canonical({**self.parent.to_dict(), 'changed': True}))
            other = DurableRvmChunks(store, other_parent, validate_context=lambda: None)
            other.bind(None)
            with self.assertRaises(CacheInvalid):
                other.get(identity)
            def changed():
                raise CacheInvalid('full current job changed')
            with self.assertRaisesRegex(CacheInvalid, 'full current job'):
                DurableRvmChunks(store, self.parent, validate_context=changed).bind(None)

    test_exact_state_dtype_and_feature_bits_survive_reopen = rvm.DurableRvmTests.test_exact_state_dtype_and_feature_bits_survive_reopen
    test_incoming_state_precision_and_bits_select_different_chains = rvm.DurableRvmTests.test_incoming_state_precision_and_bits_select_different_chains
    test_corrupt_or_incomplete_state_bundle_is_rejected_atomically = rvm.DurableRvmTests.test_corrupt_or_incomplete_state_bundle_is_rejected_atomically
    test_valid_tensors_with_invalid_provenance_are_recomputed = rvm.DurableRvmTests.test_valid_tensors_with_invalid_provenance_are_recomputed
    test_capacity_cancel_and_disk_failure_never_leave_partial_state = rvm.DurableRvmTests.test_capacity_cancel_and_disk_failure_never_leave_partial_state
    test_loaded_model_mismatch_or_changed_context_prevents_runner_access = rvm.DurableRvmTests.test_loaded_model_mismatch_or_changed_context_prevents_runner_access
    test_mutation_after_lookup_is_rejected_before_tensor_deserialization = rvm.DurableRvmTests.test_mutation_after_lookup_is_rejected_before_tensor_deserialization


@unittest.skipUnless(hann.HAS_STACK, "requires optional CPU adapters")
class UpstreamHannTests(unittest.TestCase):
    def setUp(self: Any):
        hann.DurableHannTests.setUp(self)
        self.parent = parent()

    adapter = hann.DurableHannTests.adapter
    identity = hann.DurableHannTests.identity
    test_exact_baton_aux_alpha_bits_and_dtypes_survive_reopen = hann.DurableHannTests.test_exact_baton_aux_alpha_bits_and_dtypes_survive_reopen
    test_all_live_replay_state_and_exact_model_inputs_invalidate = hann.DurableHannTests.test_all_live_replay_state_and_exact_model_inputs_invalidate
    test_incomplete_or_invalid_provenance_event_is_rejected = hann.DurableHannTests.test_incomplete_or_invalid_provenance_event_is_rejected
    test_binding_zero_budget_no_carry_and_recurrent_state_guard = hann.DurableHannTests.test_binding_zero_budget_no_carry_and_recurrent_state_guard
    test_capacity_disk_full_and_cancellation_never_seal_partial_event = hann.DurableHannTests.test_capacity_disk_full_and_cancellation_never_seal_partial_event
    test_signed_large_raw_values_and_post_lookup_mutation = hann.DurableHannTests.test_signed_large_raw_values_and_post_lookup_mutation
    test_shared_session_rejects_loaded_model_mismatch_before_runner = hann.DurableHannTests.test_shared_session_rejects_loaded_model_mismatch_before_runner


@unittest.skipUnless(prepass.HAS_STACK, "requires optional CPU adapters")
class UpstreamPrepassTests(unittest.TestCase):
    def setUp(self: Any):
        prepass.RvmPrepassTests.setUp(self)
        self.parent = parent()

    adapter = prepass.RvmPrepassTests.adapter
    run_prepass = prepass.RvmPrepassTests.run_prepass
    test_interrupted_prepass_restores_state_and_only_evaluates_remaining_chunks = prepass.RvmPrepassTests.test_interrupted_prepass_restores_state_and_only_evaluates_remaining_chunks
    test_missing_middle_chunk_recomputes_from_restored_state_then_reuses_suffix = prepass.RvmPrepassTests.test_missing_middle_chunk_recomputes_from_restored_state_then_reuses_suffix
    test_added_reset_invalidates_affected_chain_without_previous_shot_state = prepass.RvmPrepassTests.test_added_reset_invalidates_affected_chain_without_previous_shot_state
    test_native_rvm_code_with_tiny_random_weights_restores_exact_state = prepass.RvmPrepassTests.test_native_rvm_code_with_tiny_random_weights_restores_exact_state
