"""Sequential calibration through owned production WorkerClient processes.

No concurrent model owner, implicit backend fallback or export queue mutation.
Outputs and scratch remain local for audit. Output limits are sampled/postchecked
admission bounds, not a hard disk cap. Unknown process cleanup is sticky: retain
that client and start no further worker. Artifact errors are a separate outcome.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .calibration import (
    CalibrationPlan,
    canonical,
    parse,
    revalidate_plan,
    summarize_trials,
)
from .control import CancellationToken, CancelledError
from .project_collection import fresh_directory, real_path
from .worker_client import WorkerClient
from .worker_protocol import MAX_FRAME_BYTES, job_from_wire


@dataclass(frozen=True)
class CalibrationReport:
    payload_json: bytes
    report_path: Path
    retained_owner: Any = None

    def to_dict(self):
        return parse(self.payload_json)

    @property
    def summary(self):
        return self.to_dict()['summary']


class ProcessCleanupError(RuntimeError):
    def __init__(self, message):
        self.cleanup_error = message
        super().__init__(message)


class GlobalPrerequisiteUnavailable(RuntimeError):
    """No retry can change a missing shared prerequisite in this frozen run."""

    def __init__(self,message,record_ref=None):
        self.record_ref = record_ref
        super().__init__(message)


class ArtifactBudgetError(RuntimeError):
    pass


def _write(root:Path,name:str,data:dict)->dict:
    raw = canonical(data,MAX_FRAME_BYTES)
    path = root/name
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def _size(root:Path,maximum:int)->int:
    root = real_path(root,directory=True)
    original = root.lstat()

    def check_root():
        current = real_path(root,directory=True).lstat()
        if (current.st_dev,current.st_ino)!=(original.st_dev,original.st_ino):
            raise ValueError('Calibration owned artifact root changed during sampled scan')

    total = 0
    pending = [root]
    count = 0
    while pending:
        path = pending.pop()
        try:
            real_path(path,directory=True)
            with os.scandir(path) as directory:
                for item in directory:
                    count += 1
                    if count > 100000:
                        raise ArtifactBudgetError('Calibration retained artifact file-count budget exceeded')
                    try:
                        child = real_path(path/item.name)
                        info = child.lstat()
                    except FileNotFoundError:
                        # Native runtime cleanup can retire a scanned descendant.
                        # Root disappearance/replacement is never ordinary churn.
                        check_root()
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(child)
                    else:
                        if info.st_nlink != 1 or not stat.S_ISREG(info.st_mode):
                            raise ArtifactBudgetError('Calibration artifacts contain an alias/multiply linked file')
                        total += info.st_size
                        if total > maximum:
                            raise ArtifactBudgetError('Calibration retained output/scratch byte budget exceeded; limit is not a hard disk cap')
        except FileNotFoundError:
            if path==root:
                raise
            check_root()
    check_root()
    return total


def _metrics(record,job,frame_count,state,native_hw):
    if not isinstance(record,dict) or record.get('status') != 'complete':
        raise ValueError('Measurement requires a complete actual engine record')
    config,_writer = job.configurations()
    requested = json.loads(job.config_json)
    if record.get('input') != job.input_path or record.get('output_dir') != job.output_dir:
        raise ValueError('Actual record source/output differs from the owned measured job')
    if canonical(record.get('writer_settings')) != canonical(json.loads(job.writer_json)):
        raise ValueError('Actual writer settings differ from controlled requested settings')
    diagnostics = record.get('diagnostics')
    if not isinstance(diagnostics,dict) or canonical(record.get('requested_settings')) != canonical(requested):
        raise ValueError('Actual requested settings differ from the immutable measured job')
    effective = diagnostics.get('effective_settings')
    if not isinstance(effective,dict) or canonical(effective) != canonical(requested) or config.auto_vram:
        raise ValueError('Actual effective settings are missing or differ from controlled requested settings')
    source = diagnostics.get('source')
    if not isinstance(source,dict):
        raise TypeError('Actual source observations are missing')
    observed_start,observed_count,fps = source.get('start_frame'),source.get('num_frames'),source.get('native_fps',0)
    shape = source.get('native_hw')
    if (type(observed_start) is not int or observed_start != (config.start_frame or 0)
            or type(observed_count) is not int or observed_count != frame_count
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0 or fps != config.frame_dir_fps
            or not isinstance(shape,(list,tuple)) or len(shape)!=2
            or any(type(side) is not int or side<=0 for side in shape)
            or tuple(shape) != tuple(native_hw)):
        raise ValueError('Actual source In/count/FPS/native dimensions differ from the verified sample')
    outputs,contracts = diagnostics.get('outputs'),diagnostics.get('output_contracts')
    if not isinstance(outputs,dict) or not isinstance(contracts,dict) or set(outputs)!=config.outputs or set(contracts)!=config.outputs:
        raise ValueError('Actual output kinds differ from the owned measured job')
    for kind,path in outputs.items():
        channels = 4 if kind=='cutout' else 1 if kind=='alpha' and config.output_format=='exr' else 3
        contract = contracts[kind]
        if not isinstance(contract,dict) or contract.get('format')!=config.output_format or type(contract.get('channels')) is not int or contract['channels']!=channels:
            raise ValueError('Actual output format/channels differ from requested settings')
        if not isinstance(path,str) or not Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('Actual output path is not an owned absolute location')
        output = Path(path).resolve()
        if str(output)!=path or output==Path(job.output_dir) or Path(job.output_dir) not in output.parents:
            raise ValueError('Actual output path escapes the owned trial output directory')
    frames = diagnostics.get('frames_submitted')
    if type(frames) is not int or frames != frame_count:
        raise ValueError('Measured actual frame coverage differs from the fixed sample workload')
    if record.get('model_reused') is not (state=='warm_session'):
        raise ValueError('Actual model reuse differs from the declared cold/warm trial policy')
    execution = diagnostics.get('execution') or {}
    if not isinstance(execution,dict):
        raise TypeError('Execution observations must be an object')
    neural = False
    if execution.get('backend')=='pytorch' and execution.get('neural_inference') is not False:
        completed,evaluated,calls = (execution.get(key,0) for key in ('completed_prediction_frames','stage2_frame_evaluations','stage2_calls'))
        strategy = ('sparse_native_context_with_full_rate_reset_aware_rvm' if config.quick_key
                    else 'full_rate_with_chronological_rvm_prepass')
        if (execution.get('requested_device')!='cuda' or not isinstance(execution.get('gpu'),str) or not execution['gpu'].strip()
                or type(execution.get('device_index')) is not int or execution['device_index']<0
                or execution.get('autocast_dtype')!='bfloat16'
                or any(type(value) is not int or value<=0 for value in (completed,evaluated,calls))
                or completed>frame_count or evaluated>completed or calls>evaluated or execution.get('temporal_strategy')!=strategy
                or (not config.quick_key and completed!=frame_count)):
            raise ValueError('Actual CUDA device/neural counters/temporal strategy are missing or inconsistent')
        # Sparse Quick Key predicts anchors/fallbacks, while frames_submitted above
        # proves full delivered coverage. A declared neural flag alone is no evidence.
        neural = True
    result = {'frames':frames,'requested_settings':requested,'effective_settings':effective,
              'execution':execution,'resources':diagnostics.get('resources'),'neural_gpu_verified':neural,
              'resource_scope':'Runtime-reported allocator observations. Peaks may include warmup/model lifetime when profile_stages=false; not whole-machine usage or memory-fit proof.'}
    for key in ('processing_seconds','elapsed_seconds','model_load_seconds','model_initialization_seconds'):
        value = record.get(key)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
            raise ValueError('Engine measurement contains invalid timing')
        result[key] = value
    first = diagnostics.get('first_rendered_batch_seconds')
    if first is not None and (type(first) not in (int,float) or not math.isfinite(first) or first<0):
        raise ValueError('First rendered batch latency is invalid')
    result['first_rendered_batch_seconds'] = first
    result['latency_scope'] = 'First rendered batch excludes model load/preflight; not per-frame UI latency.'
    return result


def evaluate_trial_quality(quality, record, *, check=None):
    """Posthoc native-export reference agreement; never included in inference timing."""
    from .benchmark_quality import evaluate_job_quality
    from .evaluation_dossier import _compatibility
    from .project_collection import check_now

    check_now(check)
    if not isinstance(record, dict):
        raise TypeError('Quality evaluation requires the actual measured job record')
    report = evaluate_job_quality(quality['template'], record)
    check_now(check)
    rows = []
    for constraint in quality['policy']['limits']:
        row = {**constraint, 'status': 'not_verified'}
        try:
            metric, _ = _compatibility(report, report, constraint['metric'])
            row.update(value=metric['value'], population=metric['population'],
                       status='pass' if metric['value'] <= constraint['maximum'] else 'fail')
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            row['reason'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    statuses = {r['status'] for r in rows}
    return {'status': 'fail' if 'fail' in statuses else 'not_verified' if 'not_verified' in statuses else 'pass', 'metrics': rows,
            'scope': 'Absolute error against declared references, not pixel parity or absolute/visual ground-truth quality.'}, report



def run_calibration(plan:CalibrationPlan,*,worker_factory=None,cancellation:CancellationToken|None=None,
                    progress=None,clock=time.perf_counter,_owner=None)->CalibrationReport:
    if not isinstance(plan,CalibrationPlan):
        raise TypeError('Calibration requires a verified plan instance')
    revalidate_plan(plan)
    data = plan.to_dict()
    limits,summary = data['limits'],data['summary']
    token = cancellation if cancellation is not None else CancellationToken()
    factory = worker_factory if worker_factory is not None else WorkerClient
    token.check()
    report_root = fresh_directory(Path(summary['report_parent']),'calibration-')
    if progress is not None:
        progress({'completed_jobs':0,'total_jobs':summary['total_jobs'],'phase':'artifact_allocation',
                  'retained_paths':{'report_root':str(report_root),'scratch_root':None}})
    scratch_root = fresh_directory(Path(summary['scratch_parent']),'calibration-')
    started = clock()
    current = None
    report = {'schema_version':1,'type':'corridorkey_calibration_report','created_utc':datetime.now(UTC).isoformat(),
              'plan_identity':plan.identity,'binding_identity':plan.identity,'plan':data,'trials':[],
              'status':'running','cleanup_error':None,'artifact_error':None,'error':None,
              'retained_paths':{'scratch_root':str(scratch_root),'report_root':str(report_root)},
              'scope':'Cold process; filesystem cache uncontrolled. Actual sample timings, not a quality or long-clip guarantee.'}
    completed = 0
    record_number = 0
    effective_seen = {}
    execution_identity = None

    def notify(**values):
        if progress is not None:
            progress(dict(values,completed_jobs=completed,total_jobs=summary['total_jobs'],retained_paths=report['retained_paths']))

    def check():
        token.check()
        if clock()-started > limits['max_wall_seconds']:
            raise TimeoutError('Calibration declared wall-time budget exceeded')
        retained = _size(scratch_root,limits['max_output_bytes'])
        if 'quality' in data:
            _size(report_root, limits['max_output_bytes'] - retained)

    def run_job(client,wire,profile,state,repetition,phase):
        nonlocal completed,record_number
        check()
        target = scratch_root/f'{record_number:04d}-{profile}-{state}-{phase}'
        target.mkdir(mode=0o700)
        actual = dict(wire,output_dir=str(target/'outputs'))
        settings = json.loads(actual['config_json'])
        settings['temp_dir'] = str(target/'scratch')
        actual['config_json'] = json.dumps(settings,sort_keys=True,allow_nan=False)
        job = job_from_wire(actual)
        notify(profile=profile,state=state,repetition=repetition,phase=phase)
        request_id = client.send('preview',{'job':actual})
        while True:
            try:
                check()
            except BaseException:
                try:
                    client.send('cancel',{'target':request_id,'reason':'Calibration cancelled or budget exceeded'})
                except (OSError,RuntimeError,ValueError,BufferError):
                    logging.getLogger(__name__).exception('Calibration cancellation dispatch failed; close still owns retirement')
                raise
            try:
                event = client.next_event(timeout=.1)
            except TimeoutError:
                continue
            if event.get('request_id') != request_id:
                continue
            if event['kind'] not in ('result','failed','cancelled','rejected','lost'):
                continue
            completed += 1
            payload = event.get('payload') or {}
            record = payload.get('record')
            ref = None
            if isinstance(record,dict):
                ref = _write(report_root,f'record-{record_number:04d}.json',record)
            record_number += 1
            if event['kind'] != 'result':
                diagnostic = record.get('diagnostics') if isinstance(record,dict) else None
                checkpoint = diagnostic.get('last_checkpoint') if isinstance(diagnostic,dict) else None
                if (isinstance(checkpoint,dict) and checkpoint.get('stage')=='model_load'
                        and type(checkpoint.get('completed')) is int and checkpoint['completed']==0):
                    raise GlobalPrerequisiteUnavailable(f"Model loading unavailable: {payload.get('error')}",ref)
                raise ValueError(f"Engine {event['kind']}: {payload.get('error')}; record={ref}")
            return job,record,ref

    try:
        for repetition in range(limits['repetitions']):
            profiles = data['profiles'] if repetition%2==0 else list(reversed(data['profiles']))
            for profile in profiles:
                for state in ('cold_process','warm_session'):
                    check()
                    revalidate_plan(plan,check=check)
                    trial = {'profile':profile['id'],'purpose':profile['purpose'],'state':state,'repetition':repetition,
                             'eligible':False,'status':'pending','warmup_record':None,'record':None}
                    report['trials'].append(trial)
                    current = None
                    trial_start = clock()
                    primary = None
                    record = None
                    try:
                        try:
                            # WorkerClient retires failed startup before raising,
                            # or carries cleanup_error, which remains fail closed.
                            # Injected factories must obey the same owner contract.
                            current = factory(capacity=1,weights=summary['weights'],asset_root=summary['asset_root'],
                                              startup_timeout=min(10.,max(.1,limits['max_wall_seconds']-(clock()-started))),shutdown_timeout=5.)
                        except Exception as startup:
                            if getattr(startup,'cleanup_error',None):
                                raise
                            raise GlobalPrerequisiteUnavailable(f'Worker startup unavailable: {type(startup).__name__}: {startup}') from startup
                        if _owner is not None:
                            _owner(current)
                        if state=='warm_session':
                            warmjob,warm,warmref = run_job(current,profile['job'],profile['id'],state,repetition,'warmup')
                            _metrics(warm,warmjob,summary['frame_count'],'cold_process',data['native_hw'])
                            trial['warmup_record'] = warmref
                            trial_start = clock()
                        job,record,ref = run_job(current,profile['job'],profile['id'],state,repetition,'measured')
                        trial['record'] = ref
                        trial['measurement'] = _metrics(record,job,summary['frame_count'],state,data['native_hw'])
                        if state=='warm_session':
                            trial['elapsed_seconds'] = clock()-trial_start
                        trial['status'] = 'complete'
                    except (Exception,KeyboardInterrupt) as exc:
                        logging.getLogger(__name__).exception('Calibration trial failed')
                        primary = exc
                        trial['status'] = 'cancelled' if isinstance(exc,(CancelledError,KeyboardInterrupt)) else 'failed'
                        trial['error'] = f'{type(exc).__name__}: {exc}'[:4096]
                        if isinstance(exc,GlobalPrerequisiteUnavailable):
                            trial['failed_record'] = exc.record_ref
                    finally:
                        if current is not None:
                            try:
                                retired = current.close()
                                if not retired or getattr(current,'cleanup_error',None):
                                    raise ProcessCleanupError(str(getattr(current,'cleanup_error',None) or 'Worker close could not verify process/descendant retirement'))
                            except BaseException as cleanup:
                                logging.getLogger(__name__).exception('Calibration process cleanup failed')
                                report['cleanup_error'] = f'{type(cleanup).__name__}: {cleanup}'[:4096]
                                raise ProcessCleanupError(report['cleanup_error']) from primary
                            if _owner is not None:
                                _owner(None)
                            current = None
                    startup_cleanup = getattr(primary,'cleanup_error',None)
                    if startup_cleanup:
                        report['cleanup_error'] = str(startup_cleanup)[:4096]
                        raise ProcessCleanupError(report['cleanup_error']) from primary
                    if state=='cold_process':
                        trial['elapsed_seconds'] = clock()-trial_start
                    if trial['status'] == 'complete' and 'quality' in data:
                        quality_start = clock()
                        trial['quality'] = {'status': 'not_verified', 'reason': 'Evaluation not completed'}
                        try:
                            revalidate_plan(plan, check=check)
                            outcome, quality_report = evaluate_trial_quality(data['quality'], record, check=check)
                            trial['quality'] = outcome
                            trial['quality']['report'] = _write(report_root, f'quality-{record_number:04d}.json', quality_report)
                            revalidate_plan(plan, check=check)
                        except (CancelledError, KeyboardInterrupt):
                            trial['quality'].update(status='not_verified', reason='Quality evaluation cancelled')
                            raise
                        except (ValueError, TypeError, KeyError) as exc:
                            trial['quality'].update(status='not_verified', reason=f'{type(exc).__name__}: {exc}')
                        finally:
                            trial['quality_evaluation_seconds'] = clock() - quality_start
                            trial['quality_timing_scope'] = 'Posthoc evaluation after inference timer and process retirement; excluded from cold/warm trial throughput.'
                    elapsed = trial.get('elapsed_seconds')
                    if trial['status']=='complete':
                        if not isinstance(elapsed,(int,float)) or isinstance(elapsed,bool) or not math.isfinite(elapsed) or elapsed<=0:
                            raise ValueError('Actual trial wall time must be finite and positive')
                        measured = trial['measurement']
                        comparison = dict(measured['effective_settings'])
                        comparison.pop('temp_dir',None)
                        prior = effective_seen.setdefault(profile['id'],comparison)
                        if comparison != prior:
                            raise ValueError('Actual effective settings changed between comparable repetitions')
                        execution = measured['execution']
                        binding = {key:execution.get(key) for key in ('backend','requested_device','gpu','device_index','cuda_version','autocast_dtype')}
                        if execution_identity is not None and binding != execution_identity:
                            raise ValueError('Actual hardware/backend/precision changed between calibration trials')
                        execution_identity = binding
                        trial['eligible'] = measured['neural_gpu_verified'] and ('quality' not in data or trial['quality']['status'] == 'pass')
                        trial['eligibility_note'] = ('Actual production CUDA execution observations' if trial['eligible'] else
                            'Declared reference-agreement constraints failed or are unavailable' if 'quality' in data and trial['quality']['status'] != 'pass' else
                            'Fixture/unknown execution is not a GPU inference recommendation')
                        if not trial['eligible']:
                            trial['exclusion_reason'] = trial['eligibility_note']
                    revalidate_plan(plan,check=check)
                    notify(profile=profile['id'],state=state,repetition=repetition,phase='retired')
                    if isinstance(primary,(CancelledError,KeyboardInterrupt,TimeoutError,ArtifactBudgetError,OSError,GlobalPrerequisiteUnavailable)):
                        raise primary
        report['status'] = 'complete' if any(row['status']=='complete' for row in report['trials']) else 'failed'
    except (Exception,KeyboardInterrupt) as exc:
        logging.getLogger(__name__).exception('Calibration stopped')
        report['status'] = 'cancelled' if isinstance(exc,(CancelledError,KeyboardInterrupt)) else 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'[:4096]
        if isinstance(exc,(ArtifactBudgetError,OSError)):
            report['artifact_error'] = report['error']
    if report['status'] != 'complete':
        present = {(row['profile'],row['state'],row['repetition']) for row in report['trials']}
        for repetition in range(limits['repetitions']):
            for profile in data['profiles']:
                for state in ('cold_process','warm_session'):
                    if (profile['id'],state,repetition) not in present:
                        report['trials'].append({'profile':profile['id'],'purpose':profile['purpose'],'state':state,
                                                 'repetition':repetition,'status':'not_run','eligible':False,
                                                 'exclusion_reason':report['error'] or report['status'],'record':None,'warmup_record':None})
        for trial in report['trials']:
            trial['eligible'] = False
    report['execution_identity'] = execution_identity
    report['binding_identity'] = hashlib.sha256(canonical({'plan':plan.identity,'execution':execution_identity})).hexdigest()
    report['summary'] = summarize_trials(report)
    report['elapsed_seconds'] = clock()-started
    report_path = report_root/'report.json'
    raw = canonical(report)
    pending = report_root/'.report.pending'
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if report_path.exists():
        raise FileExistsError(report_path)
    pending.rename(report_path)
    return CalibrationReport(raw,report_path,current)


class CalibrationTask:
    def __init__(self,plan:CalibrationPlan,*,worker_factory=None):
        if not isinstance(plan,CalibrationPlan):
            raise TypeError('CalibrationTask requires a verified plan instance')
        self.plan = plan
        self.worker_factory = worker_factory
        self.cancellation = CancellationToken()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.client = None
        self.cleanup_error = None
        self.artifact_error = None
        self.report_path:Path|None = None
        self.report:CalibrationReport|None = None
        self.state = {'status':'starting','revision':0,'progress':{},'case_summaries':[], 'report_path':None,
                      'cleanup_error':None,'artifact_error':None,'error':None,'retained_paths':{},'summary':{}}
        self.thread = threading.Thread(target=self._run,name='calibration-owner',daemon=False)
        self.thread.start()

    def cancel(self):
        self.cancellation.cancel('Calibration cancelled by user')

    def _update(self,**values):
        with self.lock:
            self.state.update(values)
            self.state['revision'] += 1

    def _owner(self,client):
        self.client = client

    def _progress(self,progress):
        self._update(status='running',progress={key:value for key,value in progress.items() if key!='retained_paths'},retained_paths=progress['retained_paths'])

    def snapshot(self):
        with self.lock:
            raw = canonical(self.state,64*1024)
        return parse(raw)

    def _run(self):
        try:
            report = run_calibration(self.plan,worker_factory=self.worker_factory,cancellation=self.cancellation,
                                     progress=self._progress,_owner=self._owner)
            self.report = report
            self.report_path = report.report_path
            data = report.to_dict()
            self.cleanup_error = data['cleanup_error']
            self.artifact_error = data['artifact_error']
            self._update(status=data['status'],summary=data['summary'],case_summaries=data['summary']['groups'],
                         report_path=str(self.report_path),cleanup_error=self.cleanup_error,artifact_error=self.artifact_error,
                         error=data['error'],retained_paths=data['retained_paths'])
        except BaseException as exc:
            logging.getLogger(__name__).exception('Calibration task failed')
            self.cleanup_error = (str(getattr(self.client,'cleanup_error',None) or 'Retained worker ownership has not been verified retired')[:4096] if self.client is not None else getattr(exc,'cleanup_error',None))
            error = f'{type(exc).__name__}: {exc}'[:4096]
            self.artifact_error = error if isinstance(exc,(OSError,ArtifactBudgetError)) else None
            self._update(status='failed',error=error,artifact_error=self.artifact_error,cleanup_error=self.cleanup_error)
        finally:
            self.done.set()
