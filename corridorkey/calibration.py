"""Explicit local calibration plans and measured-only summaries; no model import.

A plan authenticates bounded local source/model/code bytes before execution.
Cold means a fresh worker process, never a cold filesystem cache. Samples restart
temporal state; quality and long-clip resource feasibility remain unverified.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .asset_inventory import AssetSelection
from .native_image_admission import MAX_PIXELS, admit_image
from .portable_project import _locations_overlap
from .preview_sequence import scan_sequence
from .project_collection import (
    FilePin,
    admit_file,
    assert_pin,
    check_now,
    read_file,
    real_path,
)
from .readiness import prepare_short_render
from .runtime_paths import RuntimePaths
from .session import JobRequest
from .worker_protocol import job_from_wire, job_to_wire

MAX_PLAN_BYTES = 32 * 1024**2
MAX_PROFILES = 8
OVERRIDES = {'model_scale', 'hann_chunk', 'hann_stride', 'stage2_batch', 'low_vram', 'foundation_cache_mb',
             'quick_key', 'quick_interval', 'quick_context_radius', 'quick_fallback', 'quick_min_reliability',
             'quick_max_disagreement', 'quick_flow_max_side', 'low_vram_foundation_chunk',
             'low_vram_mlp_chunk_tokens', 'low_vram_enc_frame_chunk', 'low_vram_swin_window_batch'}


def canonical(value: Any, maximum: int = MAX_PLAN_BYTES) -> bytes:
    # json.dumps deliberately rejects dataclass classes/instances and all sets.
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > maximum:
        raise ValueError('Calibration metadata byte bound exceeded')
    return raw


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate calibration JSON field')
        result[key] = value
    return result


def parse(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_PLAN_BYTES:
        raise ValueError('Calibration metadata must be bounded immutable bytes')
    data = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(data, dict):
        raise TypeError('Calibration metadata requires an object')
    canonical(data)
    return data


def _int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} requires an exact integer in [{low},{high}]')


@dataclass(frozen=True)
class CalibrationProfile:
    identifier: str
    overrides_json: bytes
    purpose: str = 'final'

    def __post_init__(self):
        if not isinstance(self.identifier, str) or re.fullmatch('[A-Za-z0-9_-]{1,64}', self.identifier) is None:
            raise ValueError('Calibration profile ID requires 1..64 ASCII letters/digits/underscore/hyphen')
        if self.purpose not in ('preview', 'final'):
            raise ValueError('Profile purpose must be preview or final; quality is not thereby verified')
        data = parse(self.overrides_json)
        if set(data) - OVERRIDES:
            raise ValueError('Calibration overrides cannot change source/range/output/colour/precision or unapproved settings')
        object.__setattr__(self, 'overrides_json', canonical(data, 16 * 1024))

    @classmethod
    def capture(cls, identifier: str, overrides: dict, purpose: str = 'final') -> CalibrationProfile:
        return cls(identifier, canonical(overrides, 16 * 1024), purpose)


@dataclass(frozen=True)
class CalibrationLimits:
    repetitions: int = 3
    max_jobs: int = 72
    max_total_frames: int = 8640
    max_wall_seconds: float = 900.
    max_output_bytes: int = 512 * 1024**2
    read_budget: int = 4 * 1024**3

    def __post_init__(self):
        for name, low, high in (('repetitions',3,7),('max_jobs',1,168),('max_total_frames',1,20160),
                                ('max_output_bytes',1,2**40),('read_budget',0,2**63-1)):
            _int(getattr(self,name),low,high,name)
        if type(self.max_wall_seconds) not in (int,float) or not math.isfinite(self.max_wall_seconds) or not 0 < self.max_wall_seconds <= 86400:
            raise ValueError('Calibration wall budget must be finite seconds in (0,86400]')


@dataclass(frozen=True)
class CalibrationPlan:
    payload_json: bytes
    pins: tuple[FilePin, ...]
    source_names: tuple[str, ...]

    def to_dict(self):
        return parse(self.payload_json)

    @property
    def identity(self):
        return hashlib.sha256(self.payload_json).hexdigest()

    @property
    def summary(self):
        return self.to_dict()['summary']


def _model_files(selection: AssetSelection, check):
    paths = RuntimePaths.from_package(Path(__file__).resolve().parents[1], selection.weights, asset_root=selection.root)
    roots = [getattr(paths,key) for key in ('weights','foundation_stats','rvm_ckpt','moge_checkpoint','cradio_dir')]
    files = []
    pending = list(roots)
    while pending:
        check_now(check)
        if len(files)+len(pending) > 4096:
            raise ValueError('Calibration model entry budget exceeded')
        path = real_path(pending.pop())
        if path.is_dir():
            with os.scandir(path) as directory:
                found = False
                for entry in directory:
                    found = True
                    check_now(check)
                    pending.append(path/entry.name)
                    if len(files)+len(pending) > 4096:
                        raise ValueError('Calibration model entry budget exceeded')
                if not found:
                    raise ValueError('Calibration local model directory is empty')
        else:
            files.append(admit_file(path))
    return tuple(sorted(files,key=lambda pin:pin.path)), roots


def _environment():
    packages = {}
    for name in ('torch','numpy','opencv-python','transformers','torchvision'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'packages':packages,
            'driver':'NOT VERIFIED until execution observations; no GPU initialized during planning'}


def _inventory(job, sample_start, frame_count, profiles, asset_selection, scratch_parent, report_parent, limits, check):
    if not isinstance(job,JobRequest) or not isinstance(asset_selection,AssetSelection) or not isinstance(limits,CalibrationLimits):
        raise TypeError('Calibration job/assets/limits require dataclass instances')
    job = job_from_wire(asdict(job))
    if not isinstance(profiles,tuple) or not 1 <= len(profiles) <= MAX_PROFILES or any(not isinstance(profile,CalibrationProfile) for profile in profiles):
        raise TypeError('Calibration profiles require a tuple of one to eight profile instances')
    if len({profile.identifier for profile in profiles}) != len(profiles):
        raise ValueError('Calibration profile IDs must be unique')
    config,writer = job.configurations()
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(config, 'Calibration sample derivation')
    if config.auto_vram or job.profile_stages:
        raise ValueError('Controlled calibration requires explicitly configured auto_vram=false and profile_stages=false')
    source = real_path(job.input_path,directory=True)
    if job.initial_hint_path is not None or job.hint_video_path is not None:
        raise ValueError("Calibration conditioning hints are unsupported until their complete frame identity can be pinned; no hint is stripped")
    # Shared short-render validation rejects ambiguous correction/hint rebasing.
    sample = prepare_short_render(job,sample_start,frame_count)
    index = scan_sequence(source,check=check)
    if sample_start+frame_count > len(index.paths):
        raise ValueError('Calibration sample extends beyond the source sequence')
    original_start = config.start_frame or 0
    available = len(index.paths)-original_start
    if available <= 0:
        raise ValueError('Original requested source In has no available frames for a duration estimate')
    requested_frames = min(config.num_frames,available) if config.num_frames > 0 else available
    scratch = real_path(scratch_parent,directory=True)
    reports = real_path(report_parent,directory=True)
    protected = [source,Path(job.output_dir).resolve(),Path(asset_selection.root).resolve()]
    if asset_selection.weights is not None:
        protected.append(Path(asset_selection.weights).resolve())
    if _locations_overlap(scratch,reports) or any(_locations_overlap(path,root) for path in (scratch,reports) for root in protected):
        raise ValueError('Calibration report/scratch parents must be separate from each other, source, original outputs and models')
    jobs = len(profiles)*limits.repetitions*3
    frames = jobs*frame_count
    if jobs > limits.max_jobs or frames > limits.max_total_frames:
        raise ValueError('Calibration declared total job/frame workload budget exceeded (including full warmups)')
    rows = []
    for profile in profiles:
        settings = replace(config,**parse(profile.overrides_json))
        settings.validate()
        if profile.purpose == 'final' and (settings.model_scale != 1 or settings.quick_key):
            raise ValueError('Final-eligible calibration profiles require scale1 and Quick Key disabled; quality remains unverified')
        candidate = JobRequest.create(job.input_path,job.output_dir,settings,writer,initial_hint_path=job.initial_hint_path,hint_video_path=job.hint_video_path)
        candidate = prepare_short_render(candidate,sample_start,frame_count)
        # Ephemeral output/scratch are supplied by the task, never identity material.
        wire = job_to_wire(candidate)
        wire['output_dir'] = job.output_dir
        rows.append({'id':profile.identifier,'purpose':profile.purpose,'overrides':parse(profile.overrides_json),'job':wire})
    pins = tuple(admit_file(path) for path in index.paths)
    missing = []
    try:
        models,model_roots = _model_files(asset_selection,check)
    except (OSError,ValueError) as exc:
        models,model_roots = (),[]
        missing.append(f'{type(exc).__name__}: {exc}')
    # Hash implementation bytes, not arbitrary sidecars, credentials or caches.
    code_root = Path(__file__).resolve().parent
    code = tuple(admit_file(path) for path in sorted(code_root.glob('*.py')))
    if len(code) > 1024:
        raise ValueError('Calibration implementation file budget exceeded')
    first = pins[sample_start]
    needed = sum(pin.size for pin in (*pins,*models,*code)) + first.size  # bounded native-header admission reread
    summary = {'preview_only':False,'source':str(source),'sample_start':sample_start,'frame_count':frame_count,
               'declared_fps':config.frame_dir_fps,'requested_available_frames':requested_frames,'source_frame_count':len(index.paths),'profiles':rows,'repetitions':limits.repetitions,
               'total_jobs':jobs,'total_frames':frames,'warmup_jobs':len(profiles)*limits.repetitions,
               'read_bytes_required':needed,'missing':missing,'limits':asdict(limits),
               'asset_root':str(Path(asset_selection.root).resolve()),'weights':str(Path(asset_selection.weights).resolve()) if asset_selection.weights is not None else None,
               'scratch_parent':str(scratch),'report_parent':str(reports),
               'scope':'Cold process, filesystem cache uncontrolled. Warm trial has one full unmeasured warmup. Temporal context restarts. No quality, long-clip RAM-fit or real-time guarantee.'}
    return job,sample,rows,pins,models,code,model_roots,index,summary


def _quality_capture(path, rows, summary, limits, check):
    """Bound policy/template metadata and inventory every referenced file before hashing."""
    from .evaluation_dossier import METRICS
    from .local_json import read_object

    pins = []
    consumed = 0
    def metadata(path, maximum):
        nonlocal consumed
        pin = admit_file(path)
        if pin.size > maximum or consumed + pin.size > limits.read_budget:
            raise ValueError('Quality metadata exceeds its byte bound or explicit read budget')
        check_now(check)
        value, raw = read_object(Path(pin.path), max_bytes=maximum, max_depth=32, regular_file=True)
        assert_pin(pin)
        pins.append(replace(pin, sha256=hashlib.sha256(raw).hexdigest()))
        consumed += len(raw)
        return value, Path(pin.path)

    policy, policy_path = metadata(path, 64 * 1024)
    if (set(policy) != {'schema_version', 'kind', 'objective', 'manifest', 'limits', 'accepted_tradeoffs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or policy['kind'] != 'calibration_quality_policy'
            or policy['objective'] != 'fastest_warm_reference_agreement'
            or not isinstance(policy['manifest'], str) or not policy['manifest'].strip()):
        raise ValueError('Expected explicit version1 fastest_warm_reference_agreement policy; memory objectives are unavailable')
    tradeoffs = policy['accepted_tradeoffs']
    if (not isinstance(tradeoffs, list) or len(tradeoffs) > 16
            or any(not isinstance(v, str) or not v.strip() or len(v) > 2048 for v in tradeoffs)):
        raise ValueError('Quality policy requires a bounded explicit tradeoff list')
    template, template_path = metadata(policy_path.parent / policy['manifest'], 1024 * 1024)
    family = template.get('kind')
    prefixes = {'alpha_sequence_evaluation': ('alpha.',), 'colour_sequence_evaluation': ('alpha.', 'colour.'),
                'temporal_sequence_evaluation': ('alpha.', 'temporal.')}.get(family) if isinstance(family, str) else None
    constraints = policy['limits']
    if prefixes is None or not isinstance(constraints, list) or not 1 <= len(constraints) <= 12:
        raise ValueError('Quality requires a supported native reference manifest and1..12 constraints')
    names = set()
    for row in constraints:
        if (not isinstance(row, dict) or set(row) != {'metric', 'units', 'maximum'}
                or row['metric'] not in METRICS or not row['metric'].startswith(prefixes)
                or row['metric'] in names or row['units'] != METRICS[row['metric']]
                or type(row['maximum']) not in (int, float)):
            raise ValueError('Quality metric must be distinct, supported by the reference family and use exact units')
        try:
            valid = math.isfinite(row['maximum']) and row['maximum'] >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('Quality tolerance must be finite nonnegative absolute reference error')
        names.add(row['metric'])
    frames = template.get('frames')
    if (not isinstance(frames, list) or not 1 <= len(frames) <= summary['frame_count']
            or not isinstance(template.get('source'), str)
            or (template_path.parent / template['source']).resolve() != Path(summary['source'])):
        raise ValueError('Quality manifest must name the captured source and a bounded sample-relative frame selection')
    if template.get('fps') != summary['declared_fps']:
        raise ValueError('Quality reference FPS differs from captured source FPS')
    seen = set()
    for frame in frames:
        if (not isinstance(frame, dict) or type(frame.get('frame')) is not int
                or not 0 <= frame['frame'] < summary['frame_count'] or frame['frame'] in seen):
            raise ValueError('Reference frame IDs must be distinct and inside the output-relative sample')
        seen.add(frame['frame'])
        for role, value in frame.items():
            if role in ('frame', 'candidate', 'candidate_rgb', 'candidate_alpha', 'cut', 'flow_previous_frame'):
                continue
            if not isinstance(value, str) or not value:
                raise ValueError('Quality reference/mask/flow assets require explicit local paths')
            pin = admit_file(template_path.parent / value)
            if pin.path not in {p.path for p in pins}:
                pins.append(pin)
    for pin in pins:
        if any(_locations_overlap(Path(pin.path), Path(summary[k])) for k in ('scratch_parent', 'report_parent')):
            raise ValueError('Quality policy/reference assets must be outside calibration artifact parents')
    for row in rows:
        config, _ = job_from_wire(row['job']).configurations()
        if config.output_format != 'exr' or 'alpha' not in config.outputs:
            raise ValueError('Quality-constrained calibration requires existing explicit native EXR alpha outputs')
    summary['quality_objective'] = policy['objective']
    summary['quality_constraints'] = [f"{r['metric']} <= {r['maximum']} {r['units']}" for r in constraints]
    summary['read_bytes_required'] += sum(p.size for p in pins) + consumed + 2 * pins[1].size + sum(p.size for p in pins[2:])
    # Shared template preparation rehashes the selected source sequence.
    return {'policy': policy, 'policy_path': str(policy_path), 'template_path': str(template_path)}, tuple(pins)


def inventory_calibration(job:JobRequest,*,sample_start:int,frame_count:int,profiles:tuple[CalibrationProfile,...],
                          asset_selection:AssetSelection,scratch_parent:Path,report_parent:Path,
                          limits:CalibrationLimits|None=None,check=None,quality_policy_path:Path|None=None)->dict:
    limits = limits or CalibrationLimits()
    values = _inventory(job,sample_start,frame_count,profiles,asset_selection,scratch_parent,report_parent,limits,check)
    summary = values[-1]
    if quality_policy_path is not None:
        _quality_capture(quality_policy_path, values[2], summary, limits, check)
        summary['read_bytes_required'] += sum(pin.size for pin in values[3])
    else:
        summary['quality_objective'] = 'timing_only; quality NOT VERIFIED'
    return summary


def plan_calibration(job:JobRequest,*,sample_start:int,frame_count:int,profiles:tuple[CalibrationProfile,...],
                     asset_selection:AssetSelection,scratch_parent:Path,report_parent:Path,
                     limits:CalibrationLimits|None=None,check=None,quality_policy_path:Path|None=None)->CalibrationPlan:
    limits = limits if limits is not None else CalibrationLimits()
    job,_sample,rows,sources,models,code,model_roots,index,summary = _inventory(job,sample_start,frame_count,profiles,asset_selection,scratch_parent,report_parent,limits,check)
    quality, quality_pins = None, ()
    if quality_policy_path is not None:
        quality, quality_pins = _quality_capture(quality_policy_path, rows, summary, limits, check)
        summary['read_bytes_required'] += sum(pin.size for pin in sources)
    else:
        summary['quality_objective'] = 'timing_only; quality NOT VERIFIED'
    if summary['missing']:
        raise ValueError('Calibration local model assets unavailable: '+str(summary['missing']))
    if summary['read_bytes_required'] > limits.read_budget:
        raise ValueError(f"Calibration verification requires {summary['read_bytes_required']} bytes; explicit read budget is {limits.read_budget}")
    verified = []
    categories = {}
    for name,pins in (('source',sources),('models',models),('implementation',code),('quality',quality_pins)):
        hashes = []
        for pin in pins:
            digest,_ = read_file(pin,check=check)
            if pin.sha256 and pin.sha256 != digest:
                raise ValueError('Captured quality metadata bytes changed before verification')
            verified.append(replace(pin,sha256=digest))
            hashes.append({'path':pin.path,'bytes':pin.size,'sha256':digest})
        categories[name] = hashes
    if quality is not None:
        from .benchmark_quality import prepare_quality_templates
        templates = prepare_quality_templates([quality['template_path']], [job_from_wire(rows[0]['job'])],
                                              manifest_dir=Path(quality['template_path']).parent)
        if not templates or templates[0] is None:
            raise ValueError('Requested quality template is unavailable')
        template = templates[0]
        expected = {p.path: p.sha256 for p in verified}
        if template['sha256'] != expected[quality['template_path']] or any(expected.get(p) != h for p, h in template['asset_hashes'].items()):
            raise ValueError('Quality template/reference bytes changed during preparation')
        quality['template'] = template
    image = admit_image(sources[sample_start].path,expected_hw=None,max_pixels=MAX_PIXELS,check=check)
    if image.sha256 != verified[sample_start].sha256:
        raise ValueError('Sample changed during native header admission')
    for pin in verified:
        assert_pin(pin)
    if scan_sequence(job.input_path,check=check).paths != index.paths:
        raise ValueError('Source sequence membership changed while planning')
    payload = {'schema_version':1,'type':'corridorkey_calibration_plan','requested_job':job_to_wire(job),
               'summary':summary,'profiles':rows,'limits':asdict(limits),'environment':_environment(),
               'identities':categories,'source_index_identity':index.identity,'native_hw':list(image.shape),
               'model_roots':[str(path) for path in model_roots],
               'implementation_scope':'CorridorKey package Python modules and local model assets; external installed package versions recorded, package binary bytes not hashed.',
               'created_utc':datetime.now(UTC).isoformat()}
    if quality is not None:
        payload['quality'] = quality
    check_now(check)
    return CalibrationPlan(canonical(payload),tuple(verified),tuple(Path(path).name for path in index.paths))


def revalidate_plan(plan:CalibrationPlan,*,check=None)->None:
    if not isinstance(plan,CalibrationPlan):
        raise TypeError('Calibration requires a verified plan instance')
    data = plan.to_dict()
    for pin in plan.pins:
        check_now(check)
        assert_pin(pin)
    index = scan_sequence(data['requested_job']['input_path'],check=check)
    if tuple(Path(path).name for path in index.paths) != plan.source_names:
        raise ValueError('Calibration source membership changed')
    selection = AssetSelection(Path(data['summary']['asset_root']),Path(data['summary']['weights']) if data['summary']['weights'] is not None else None)
    live,_ = _model_files(selection,check)
    current_code = [str(path) for path in sorted(Path(__file__).resolve().parent.glob('*.py'))]
    if current_code != [row['path'] for row in data['identities']['implementation']]:
        raise ValueError('Calibration implementation membership changed')
    expected = [row['path'] for row in data['identities']['models']]
    if [pin.path for pin in live] != expected:
        raise ValueError('Calibration model membership changed')


DURATION_SCOPE = ('Sample-limited processing-only estimate; excludes startup, model load and preflight. '
                  'Observed sample range is not a confidence interval. Cuts, temporal context, thermal conditions, '
                  'source complexity and memory growth may change actual duration; no universal guarantee.')


def _duration_estimate(data, rows, state, sample_count, eligible):
    unavailable = {'status':'NOT VERIFIED','frames':None,'min_seconds':None,'central_seconds':None,
                   'max_seconds':None,'min_fps':None,'median_fps':None,'max_fps':None,'scope':DURATION_SCOPE}
    if state != 'warm_session' or not eligible or len(rows)<3:
        return unavailable
    try:
        plan = data['plan']
        source_count = len(plan['identities']['source'])
        _int(source_count,1,100000,'Source identity frame count')
        settings = parse(plan['requested_job']['config_json'].encode())
        start = settings.get('start_frame')
        start = 0 if start is None else start
        _int(start,0,source_count-1,'Original requested source In')
        requested = settings['num_frames']
        _int(requested,-2**63,2**63-1,'Original requested frame count')
        available = source_count-start
        frames = min(requested,available) if requested>0 else available
        seconds = []
        for row in rows:
            measured = row['measurement']
            if measured.get('neural_gpu_verified') is not True or measured.get('frames') != sample_count:
                return unavailable
            value = measured['processing_seconds']
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                return unavailable
            seconds.append(value)
        rates = [sample_count/value for value in seconds]
        median = statistics.median(rates)
        return {'status':'sample-limited estimate','frames':frames,'min_seconds':frames/max(rates),
                'central_seconds':frames/median,'max_seconds':frames/min(rates),
                'min_fps':min(rates),'median_fps':median,'max_fps':max(rates),'scope':DURATION_SCOPE}
    except (KeyError,TypeError,ValueError,OverflowError):
        return unavailable


def _quality_profile_passed(data, profile, repetitions):
    expected = {(state, repetition) for state in ('cold_process', 'warm_session') for repetition in range(repetitions)}
    trials = [row for row in data['trials'] if row['profile'] == profile]
    return (len(trials) == len(expected)
            and {(row.get('state'), row.get('repetition')) for row in trials} == expected
            and all(row.get('status') == 'complete' and row.get('quality', {}).get('status') == 'pass'
                    and row.get('measurement', {}).get('neural_gpu_verified') is True for row in trials))


def summarize_trials(data:dict)->dict:
    count = data['plan']['summary']['frame_count']
    _int(count,1,120,'Measured sample frame count')
    repetitions = data['plan']['limits']['repetitions']
    _int(repetitions,3,7,'Measured repetitions')
    groups = []
    for profile in data['plan']['profiles']:
        for state in ('cold_process','warm_session'):
            rows = [row for row in data['trials'] if row['profile']==profile['id'] and row['state']==state and row.get('eligible') is True
                    and ('quality' not in data['plan'] or row.get('quality', {}).get('status') == 'pass')]
            times = [row['elapsed_seconds'] for row in rows]
            groups.append({'profile':profile['id'],'purpose':profile['purpose'],'state':state,'successful_repetitions':len(rows),
                           'eligible':len(rows)==repetitions,
                           'estimated_render_duration':_duration_estimate(data,rows,state,count,len(rows)==repetitions),
                           'median_seconds':statistics.median(times) if times else None,
                           'min_seconds':min(times) if times else None,'max_seconds':max(times) if times else None,
                           'median_fps':count/statistics.median(times) if times else None,
                           'scope':'Actual full trial wall time; warmup excluded from warm timing, process startup/shutdown included in cold timing.'})
    recommendations = []
    for purpose in ('preview','final'):
        choices = [row for row in groups if row['purpose']==purpose and row['eligible'] and row['state']=='warm_session']
        if 'quality' in data['plan']:
            choices = [row for row in choices if _quality_profile_passed(data, row['profile'], repetitions)]
        if choices:
            selected = min(choices,key=lambda row:row['median_seconds'])
            recommendations.append({'purpose':purpose,'profile':selected['profile'],'basis':('Fastest measured comparable warm sample satisfying all declared reference-agreement tolerances; visual quality/full-clip performance NOT VERIFIED' if 'quality' in data['plan'] else 'Fastest measured comparable warm sample; quality and full-clip performance NOT VERIFIED'),
                                    'median_fps':selected['median_fps'],'estimated_render_duration':selected['estimated_render_duration'],'sample_seconds_range':[selected['min_seconds'],selected['max_seconds']]})
    return {'groups':groups,'recommendations':recommendations,'status':data['status'],
            'quality_objective':data['plan'].get('quality', {}).get('policy', {}).get('objective', 'timing_only; quality NOT VERIFIED'),
            'quality_constraints':data['plan']['summary'].get('quality_constraints', []),
            'excluded_trials':sum(row.get('eligible') is not True for row in data['trials']),
            'duration_scope':'N/sample FPS is sample-based only. Startup, scene complexity, cuts, thermal conditions and memory growth can invalidate extrapolation.'}


def calibration_validity(report:dict,current_identity:str,*,now:datetime|None=None,max_age_seconds:float=7*86400)->dict:
    now = now if now is not None else datetime.now(UTC)
    created = datetime.fromisoformat(report['created_utc'])
    reasons = []
    age = (now-created).total_seconds()
    if age < 0 or age > max_age_seconds:
        reasons.append('Calibration expired or clock moved backwards')
    if current_identity != report['binding_identity']:
        reasons.append('Hardware/model/sample/settings/implementation binding changed or is not verified')
    return {'valid':not reasons,'reasons':reasons,'age_seconds':age,'scope':'Validity does not establish quality or long-clip applicability.'}
