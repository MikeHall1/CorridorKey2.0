"""Explicit whole-plan gate admission; never infer per-shot memory or optimality."""
from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import replace
from pathlib import Path

from .batch import jobs_from_data
from .benchmark_report import read_report
from .calibration_task import _metrics
from .project_collection import admit_file, assert_pin, check_now
from .session import JobRequest
from .shot_profiles import ProfilePlan, base_recipe_key, validate_live


def candidate_plan(job: JobRequest, report: dict) -> ProfilePlan:
    """Restrict this recommendation workflow, not the general paired benchmark."""
    plans = []
    for side in ('baseline', 'candidate'):
        batch = report['provenance'][side]['normalized_batch']
        captured, _ = jobs_from_data({'schema_version': 1, 'kind': 'inference_batch', **batch}, Path.cwd())
        if len(captured) != 1:
            raise ValueError('Shot recommendation requires one complete selected-timeline job per side')
        captured_job = captured[0]
        config, _ = captured_job.configurations()
        if config.auto_vram or captured_job.profile_stages or config.shot_profiles is None:
            raise ValueError('Shot recommendation requires explicit unprofiled complete manual plans')
        plan = ProfilePlan.parse(config.shot_profiles)
        plan.validate_recipe(captured_job)
        if not plan.to_dict()['enabled'] or base_recipe_key(captured_job) != base_recipe_key(job):
            raise ValueError('Measured shot plan is disabled or differs from the current complete base recipe')
        plans.append(plan)
    left, right = (p.to_dict() for p in plans)
    normalized = copy.deepcopy(right)
    if len(left['rows']) != len(right['rows']):
        raise ValueError('Shot recommendations cannot change row boundaries/counts')
    changed = False
    for a, b in zip(left['rows'], normalized['rows'], strict=True):
        changed |= a['stage2_batch'] != b['stage2_batch']
        b['stage2_batch'] = a['stage2_batch']
    if left != normalized or not changed:
        raise ValueError('Only shot-row Stage2 batch caps may differ; complete candidate must actually differ')
    return plans[1]


def recommend(job: JobRequest, *, report_path, policy_path, policy_sha256,
              numerical_evidence_path, numerical_evidence_sha256,
              resource_evidence_path, resource_evidence_sha256,
              visual_evidence_path=None, visual_evidence_sha256=None, check=None):
    """Inspect existing local receipts, return candidate + nonpersistent provenance.

    A passing screen qualifies only this candidate versus its one tested baseline,
    under the supplied contracts. It is not an authenticated hardware measurement
    or a globally optimal/per-shot resource recommendation. No files/jobs are written.
    """
    from .regression_gate import evaluate
    from .regression_numerical import fence

    references = [(report_path, None), (policy_path, policy_sha256),
        (numerical_evidence_path, numerical_evidence_sha256),
        (resource_evidence_path, resource_evidence_sha256)]
    if (visual_evidence_path is None) != (visual_evidence_sha256 is None):
        raise ValueError('Visual evidence requires both explicit path and SHA256')
    if visual_evidence_path is not None:
        references.append((visual_evidence_path, visual_evidence_sha256))
    pins = []
    for index, (path, sha) in enumerate(references):
        check_now(check)
        if path is None or (index > 0 and (not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None)):
            raise ValueError('Explicit evidence paths and exact SHA256 receipts are required')
        pins.append(admit_file(path))
    report, raw = read_report(Path(report_path), regular_file=True)
    plan = candidate_plan(job, report)
    result = evaluate(report_path, policy_path, policy_sha256, check=check,
        numerical_evidence_path=numerical_evidence_path, numerical_evidence_sha256=numerical_evidence_sha256,
        resource_evidence_path=resource_evidence_path, resource_evidence_sha256=resource_evidence_sha256,
        visual_evidence_path=visual_evidence_path, visual_evidence_sha256=visual_evidence_sha256)
    policy = result.get('policy', {}).get('value', {})
    if (result['status'] != 'pass' or 'numerical' not in policy or 'resources' not in policy
            or not {'numerical_parity', 'observed_memory'} <= set(policy.get('required_evidence', []))
            or result.get('source', {}).get('sha256') != hashlib.sha256(raw).hexdigest()):
        raise ValueError('Whole-plan recommendation requires a passing bound numerical/resource gate: '
                         + str(result.get('error') or result['status']))
    if not any(c['job'] == 0 for c in policy['resources']['constraints']):
        raise ValueError('Whole-plan recommendation requires an explicit observed job0 allocator constraint')
    # Reuse actual-record admission, never turn a CPU/fixture provider into GPU evidence.
    for pair in report['pairs']:
        for side in ('baseline', 'candidate'):
            sample = pair['samples'][side]
            record = sample['runs'][0]['jobs'][0]
            captured, _ = jobs_from_data({'schema_version': 1, 'kind': 'inference_batch',
                **report['provenance'][side]['normalized_batch']}, Path(report_path))
            actual_job = replace(captured[0], output_dir=record['output_dir'])
            source = record['diagnostics']['source']
            span = plan.to_dict()['source_range']
            if [source['start_frame'], source['start_frame'] + source['num_frames']] != span:
                raise ValueError('Actual measured source coverage differs from complete shot plan')
            measured = _metrics(record, actual_job, span[1] - span[0], sample['model_state'], source['native_hw'])
            if not measured['neural_gpu_verified']:
                raise ValueError('CPU/unknown execution cannot qualify a measured shot recommendation')
    validate_live(plan, job, check=check)
    # Same publication boundary as the gate CLI: nested declaration, numerical
    # export and resource inputs all remain live, not merely their top receipts.
    for name in ('numerical_inputs', 'visual_inputs', 'resource_inputs'):
        if name in result:
            fence(result[name], check=check)
    for pin in pins:
        check_now(check)
        assert_pin(pin)
    checks = {r['name']: r for r in result['checks']}
    summary = {'pair_path': str(Path(report_path).resolve()),
        'policy_path': str(Path(policy_path).resolve()),
        'numerical_evidence_path': str(Path(numerical_evidence_path).resolve()),
        'resource_evidence_path': str(Path(resource_evidence_path).resolve()),
        'source_range': plan.to_dict()['source_range'],
        'pair_sha256': result['source']['sha256'], 'policy_sha256': policy_sha256,
        'context_sha256': result['context_sha256'], 'numerical_evidence_sha256': numerical_evidence_sha256,
        'resource_evidence_sha256': resource_evidence_sha256,
        'candidate_over_baseline': checks['elapsed_seconds']['candidate_over_baseline'],
        'observed_memory': checks['observed_memory']['observations'],
        'accepted_tradeoffs': policy['accepted_tradeoffs'],
        'scope': 'Gate-eligible COMPLETE measured candidate versus one baseline, not fastest/best or independently optimized rows. '
                 'Whole-selected-timeline allocator observations only; resource_fit remains unavailable. '
                 'No hard cap, authenticated hardware/human inspection or future performance guarantee. Explicit Apply required.'}
    return plan, summary
