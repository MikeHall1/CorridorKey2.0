"""Source-bound opt-in shot settings; one unchanged temporal job, no model planner."""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .diagnostics import finite_json_float, json_value
from .local_json import _depth, _pairs, read_object
from .project_collection import admit_file, assert_pin, read_file

if TYPE_CHECKING:
    from .config import InferenceConfig
    from .session import JobRequest

MAX_BYTES = 1024 * 1024
MAX_ROWS = 256
MAX_FRAMES = 100_000
READ_BUDGET = 2 * 1024**3
ANALYSIS_MAX_PIXELS = 16 * 1024**2
ANALYSIS_MAX_ENCODED_BYTES = 256 * 1024**2
CONTROLS = ('stage2_batch', 'alpha_expand_radius', 'alpha_feather_radius')


def _canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError('Shot profile metadata exceeds 1 MiB')
    return raw


def _integer(value, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError('Shot profile integer is outside its exact bounds')
    return value


def base_recipe_key(job: JobRequest):
    config, writer = job.configurations()
    settings = json_value(replace(config, shot_profiles=None, temp_dir=Path(config.temp_dir).resolve()))
    return hashlib.sha256(_canonical({'input': job.input_path, 'settings': settings,
        'writer': json_value(writer), 'initial_hint': job.initial_hint_path,
        'full_hint': job.hint_video_path, 'profile_stages': job.profile_stages}).encode()).hexdigest()


@dataclass(frozen=True)
class ProfilePlan:
    payload: str
    _rows: tuple = field(init=False, repr=False)
    _starts: tuple[int, ...] = field(init=False, repr=False)
    _enabled: bool = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.payload, str) or len(self.payload.encode()) > MAX_BYTES:
            raise ValueError('Shot profile payload exceeds its byte/type bound')
        value = json.loads(self.payload)
        if self._validate(value) != self.payload:
            raise ValueError('Shot profile payload must be canonical')
        rows = tuple((row['range'][0], row['range'][1], row['enabled'],
                      *(row[name] for name in CONTROLS)) for row in value['rows'])
        object.__setattr__(self, '_rows', rows)
        object.__setattr__(self, '_starts', tuple(row[0] for row in rows))
        object.__setattr__(self, '_enabled', value['enabled'])

    def to_dict(self):
        return json.loads(self.payload)

    @staticmethod
    def _validate(value):
        if (not isinstance(value, dict) or set(value) != {'schema_version', 'kind', 'enabled',
                'recipe_sha256', 'source_sha256', 'source_range', 'read_budget', 'rows', 'analysis'}
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['kind'] != 'shot_processing_profiles' or type(value['enabled']) is not bool):
            raise ValueError('Expected version1 shot_processing_profiles document')
        for key in ('recipe_sha256', 'source_sha256'):
            if not isinstance(value[key], str) or re.fullmatch('[0-9a-f]{64}', value[key]) is None:
                raise ValueError('Shot profiles require exact source/recipe SHA256 bindings')
        span = value['source_range']
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError('Shot source range requires an absolute half-open pair')
        lo, hi = _integer(span[0], 0, 2**63 - 1), _integer(span[1], 1, 2**63 - 1)
        if not 0 < hi - lo <= MAX_FRAMES:
            raise ValueError('Shot selection exceeds the 100000-frame bound')
        _integer(value['read_budget'], 1, READ_BUDGET)
        if (not isinstance(value['analysis'], dict) or set(value['analysis']) != {'calibrated', 'resource_fit'}
                or value['analysis']['calibrated'] is not False or value['analysis']['resource_fit'] != 'unavailable'):
            raise ValueError('Shot source heuristics do not establish calibrated quality or resource fit')
        rows = value['rows']
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
            raise ValueError('Shot profiles require 1..256 rows; merge or select a shorter range')
        previous = lo
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'range', 'enabled', *CONTROLS, 'signals'}
                    or type(row['enabled']) is not bool or not isinstance(row['range'], list)
                    or len(row['range']) != 2):
                raise ValueError('Shot profile row fields/types differ')
            start, end = (_integer(row['range'][0], lo, hi - 1), _integer(row['range'][1], lo + 1, hi))
            if not previous <= start < end:
                raise ValueError('Shot profile rows overlap or are out of order')
            previous = end
            _integer(row['stage2_batch'], 1, MAX_FRAMES)
            _integer(row['alpha_expand_radius'], -32, 32)
            _integer(row['alpha_feather_radius'], 0, 32)
            signals = row['signals']
            if not isinstance(signals, dict) or set(signals) != {'mean_rgb_change', 'mean_edge_detail'}:
                raise ValueError('Shot source signal inventory differs')
            for number in signals.values():
                if type(number) not in (int, float) or not 0 <= number <= 1:
                    raise ValueError('Shot heuristic scores must be finite in [0,1]')
        return _canonical(value)

    @classmethod
    def parse(cls, value):
        return cls(cls._validate(value))

    @classmethod
    def from_bytes(cls, raw: bytes):
        if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
            raise ValueError('Shot profile editor exceeds 1 MiB')
        _depth(raw, 24)
        return cls.parse(json.loads(raw, object_pairs_hook=_pairs, parse_float=finite_json_float,
                                   parse_constant=finite_json_float))

    def validate_recipe(self, job: JobRequest):
        data = self.to_dict()
        config, _ = job.configurations()
        if base_recipe_key(job) != data['recipe_sha256']:
            raise ValueError('Shot profile base recipe changed; disable or analyze again')
        if any(row['stage2_batch'] > config.stage2_batch for row in data['rows']):
            raise ValueError('Shot batch caps cannot exceed the explicitly selected global batch size')

    def settings(self, frame: int, config: InferenceConfig):
        index = bisect_right(self._starts, frame) - 1
        if self._enabled and index >= 0:
            row = self._rows[index]
            if row[2] and frame < row[1]:
                return row[3:]
        return tuple(getattr(config, name) for name in CONTROLS)

    def batches(self, items, config: InferenceConfig, start: int):
        cursor = 0
        while cursor < len(items):
            cap = self.settings(start + items[cursor][0], config)[0]
            size = 1
            while size < min(cap, len(items) - cursor):
                following_cap = self.settings(start + items[cursor + size][0], config)[0]
                if size + 1 > following_cap:
                    break
                cap = min(cap, following_cap)
                size += 1
            yield items[cursor:cursor + size]
            cursor += size


def reject_derived_profiles(config: InferenceConfig, workflow: str):
    if config.shot_profiles is not None and config.shot_profiles['enabled']:
        raise ValueError(f'{workflow} cannot preserve source/base-recipe-bound shot profiles; disable profiles explicitly first')


def _source_pin(job, span, read_budget, check):
    """Pin full membership/coordinates and selected bytes, without loading a model."""
    from .preview_sequence import scan_sequence

    source = Path(job.input_path)
    if source.is_dir():
        index = scan_sequence(source, check=check)
        start, end = span
        if not 0 <= start < end <= len(index.paths):
            raise ValueError('Shot range lies outside current source membership')
        paths = index.paths[start:end]
        membership = index.identity
    else:
        paths, membership = (str(source),), None
    pins = [admit_file(path) for path in paths]
    if source.is_dir() and any(pin.size > ANALYSIS_MAX_ENCODED_BYTES for pin in pins):
        raise ValueError('Shot source image exceeds the 256 MiB encoded-byte limit')
    # Capture + decode/header admission + final verification; runtime uses fewer reads.
    if sum(pin.size for pin in pins) * 4 + (pins[0].size * 2 if pins else 0) > read_budget:
        raise ValueError('Shot source reads exceed the explicit aggregate encoded-byte budget')
    digest = hashlib.sha256(json.dumps({'membership': membership, 'source_range': span}, sort_keys=True).encode())
    contents = {}
    for pin in pins:
        content, _ = read_file(pin, check=check)
        contents[pin.path] = content
        digest.update(json.dumps([pin.path, content], separators=(',', ':')).encode())
    for pin in pins:
        assert_pin(pin)
    return digest.hexdigest(), contents, membership


def validate_live(plan: ProfilePlan, job: JobRequest, *, check=None):
    plan.validate_recipe(job)
    data = plan.to_dict()
    current, _, _ = _source_pin(job, data['source_range'], data['read_budget'], check)
    if current != data['source_sha256']:
        raise ValueError('Shot profile source changed; disable or analyze again')


def apply_profiles(job: JobRequest, plan: ProfilePlan, *, check=None):
    from .session import JobRequest

    if plan.to_dict()['enabled']:
        validate_live(plan, job, check=check)
    config, writer = job.configurations()
    return JobRequest.create(job.input_path, job.output_dir, replace(config, shot_profiles=plan.to_dict()), writer,
        initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
        profile_stages=job.profile_stages)


def execution_clone(job: JobRequest, *, owned_root: Path, temp_dir: Path, output_dir: Path,
                    profile_stages: bool) -> JobRequest:
    """Explicit owned measurement clone; never project a timeline or relax stale admission.

    The caller owns attempt allocation/retirement and source-content fences. This
    operation performs no writes; runtime still validates the unchanged source pin.
    """
    from .project_collection import real_path
    from .session import JobRequest

    if not isinstance(job, JobRequest) or type(profile_stages) is not bool:
        raise ValueError('Measurement cloning requires an immutable job and boolean profiling flag')
    root = real_path(owned_root, directory=True)
    temp, output = Path(temp_dir).resolve(), Path(output_dir).resolve()
    if any(path == root or not path.is_relative_to(root) for path in (temp, output)):
        raise ValueError('Measurement scratch/output must remain inside the owned attempt')
    if temp == output or temp.is_relative_to(output) or output.is_relative_to(temp):
        raise ValueError('Measurement scratch/output must be separate owned locations')
    config, writer = job.configurations()
    plan = None
    if config.shot_profiles is not None and config.shot_profiles['enabled']:
        plan = ProfilePlan.parse(config.shot_profiles)
        plan.validate_recipe(job)
    clone = JobRequest.create(job.input_path, output, replace(config, temp_dir=temp), writer,
        initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
        profile_stages=profile_stages)
    if plan is not None:
        data = plan.to_dict()
        data['recipe_sha256'] = base_recipe_key(clone)
        rebound = ProfilePlan.parse(data)
        config, writer = clone.configurations()
        clone = JobRequest.create(clone.input_path, clone.output_dir,
            replace(config, shot_profiles=rebound.to_dict()), writer,
            initial_hint_path=clone.initial_hint_path, hint_video_path=clone.hint_video_path,
            profile_stages=clone.profile_stages)
        rebound.validate_recipe(clone)
    return clone


def analyze_job(job: JobRequest, *, check=None, progress=None, read_budget=READ_BUDGET):
    from .control import JobControl
    from .native_image_admission import admit_image
    from .shot_analysis import analyze_shots
    from .video_io import DraftSourceReader, _read_frame_dir_views, probe_video_meta

    check = check or (lambda: None)
    _integer(read_budget, 1, READ_BUDGET)
    config, _ = job.configurations()
    job.validate_source(config)
    if config.start_mode != 'begin':
        raise ValueError('Shot analysis requires an explicit begin/source range; choose In/Out first')
    source = Path(job.input_path)
    is_dir = source.is_dir()
    sequence_source = None
    start = config.start_frame or 0
    if is_dir:
        from .preview_sequence import scan_sequence
        index = scan_sequence(source, check=check)
        count = min(config.num_frames, len(index.paths) - start) if config.num_frames > 0 else len(index.paths) - start
        span = [start, start + count]
        before, contents, membership = _source_pin(job, span, read_budget, check)
        if membership != index.identity:
            raise ValueError('Shot source membership changed during capture')
        def admit_frame(path, expected_hw):
            admitted = admit_image(path, expected_hw=expected_hw, max_pixels=ANALYSIS_MAX_PIXELS,
                max_bytes=ANALYSIS_MAX_ENCODED_BYTES, check=check)
            if admitted.sha256 != contents[str(admitted.path)]:
                raise ValueError('Shot source changed since the bound source fingerprint (encoded bytes differ)')
            return admitted
        expected = admit_frame(index.paths[start], None).shape
        def read_sequence(frame):
            from .draft import SourceFrame

            check()
            admitted = admit_frame(index.paths[start + frame], expected)
            proxy, linear = _read_frame_dir_views(str(admitted.path), exr_linear=config.frame_dir_linear,
                alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile,
                encoded_bytes=admitted.encoded_bytes)
            if linear.shape != (*expected, 3) or proxy.shape != (*expected, 3):
                raise ValueError('Shot source decoded geometry differs from its admitted native header')
            check()
            return SourceFrame(linear, proxy)
        sequence_source = read_sequence
        meta = {'num_frames': count}
    else:
        pin = admit_file(source)
        if pin.size * 4 > read_budget:
            raise ValueError('Shot video exceeds the explicit encoded-byte read budget')
        meta = probe_video_meta(str(source), num_frames=config.num_frames, start_frame=config.start_frame,
            shape_quantum=config.shape_quantum, native_resolution=True, start_mode='begin', seed=config.seed,
            model_scale=config.model_scale, linear_brightness=config.linear_brightness,
            linear_contrast=config.linear_contrast, linear_contrast_pivot=config.linear_contrast_pivot)
        span = [start, start + meta['num_frames']]
        assert_pin(pin)
        before, _, _ = _source_pin(job, span, read_budget, check)
    if not 0 < meta['num_frames'] <= MAX_FRAMES:
        raise ValueError('Shot analysis requires 1..100000 selected frames')
    rows = []
    current = None
    def emit(record):
        nonlocal current
        check()
        if current is None or record['cut']:
            if len(rows) >= MAX_ROWS:
                raise ValueError('Shot analysis exceeds 256 rows; choose a shorter source range')
            current = {'start': start + record['frame'], 'count': 0, 'motion': 0., 'detail': 0.}
            rows.append(current)
        current['count'] += 1
        current['motion'] += 0. if record['cut'] else record['mean_rgb_change']
        current['detail'] += record['edge_detail']
    control = JobControl(progress=progress)
    control.check_hook = check
    with_reader = None if is_dir else DraftSourceReader(str(source), meta, is_dir=False)
    try:
        read_source = sequence_source if with_reader is None else with_reader.read
        assert read_source is not None
        analyze_shots(meta['num_frames'], read_source, control=control, emit=emit)
    finally:
        if with_reader is not None:
            with_reader.close()
    proposals = []
    for row in rows:
        motion, detail = row['motion'] / row['count'], row['detail'] / row['count']
        # Source-only heuristic: suggest modest feather only on smooth/stable shots.
        # It is not a subject-edge, motion-flow, quality or VRAM measurement.
        feather = 1 if detail < .02 and motion < .01 and config.alpha_feather_radius == 0 else config.alpha_feather_radius
        proposals.append({'range': [row['start'], row['start'] + row['count']], 'enabled': True,
            'stage2_batch': config.stage2_batch, 'alpha_expand_radius': config.alpha_expand_radius,
            'alpha_feather_radius': feather, 'signals': {'mean_rgb_change': motion, 'mean_edge_detail': detail}})
    result = ProfilePlan.parse({'schema_version': 1, 'kind': 'shot_processing_profiles', 'enabled': True,
        'recipe_sha256': base_recipe_key(job), 'source_sha256': before, 'source_range': span,
        'read_budget': read_budget, 'rows': proposals,
        'analysis': {'calibrated': False, 'resource_fit': 'unavailable'}})
    validate_live(result, job, check=check)
    return result


def save_profiles(path: Path, plan: ProfilePlan, job: JobRequest, *, check=None):
    from .readiness import export_report
    from .worker_protocol import job_to_wire

    validate_live(plan, job, check=check)
    from .session import JobRequest

    config, writer = job.configurations()
    base = JobRequest.create(job.input_path, job.output_dir, replace(config, shot_profiles=None), writer,
        initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
        profile_stages=job.profile_stages)
    protected = [job.input_path, job.output_dir, job.initial_hint_path, job.hint_video_path, config.corrections_path]
    if config.corrections_path is not None:
        from .corrections import load_corrections
        protected.extend(str(p) for p in load_corrections(config.corrections_path).asset_paths())
    record = {'schema_version': 1, 'kind': 'shot_profile_document', 'job': job_to_wire(base),
        'plan': plan.to_dict(), 'protected_paths': [str(p) for p in protected if p is not None]}
    _canonical(record)
    if path.resolve().is_relative_to(Path(job.output_dir).resolve()):
        raise ValueError('Save shot profiles outside original output destinations')
    export_report(record, path)


def load_profiles(path: Path):
    from .worker_protocol import job_from_wire

    data, _ = read_object(path, max_bytes=MAX_BYTES, max_depth=24, regular_file=True)
    if (set(data) != {'schema_version', 'kind', 'job', 'plan', 'protected_paths'}
            or type(data['schema_version']) is not int or data['schema_version'] != 1
            or data['kind'] != 'shot_profile_document'):
        raise ValueError('Expected version1 shot profile document')
    job = job_from_wire(data['job'])
    plan = ProfilePlan.parse(data['plan'])
    plan.validate_recipe(job)
    return job, plan
