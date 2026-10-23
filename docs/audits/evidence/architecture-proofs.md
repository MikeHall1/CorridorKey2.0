# Architecture reproducers

Run from repository root with existing compatible interpreter. Temporary files only; no model downloads. Both observed by independent architect; ARCH-001 additionally rerun by coordinator below.

## ARCH-001

```python
import json, tempfile
from pathlib import Path
import numpy as np
from corridorkey.video_io import cv2
from corridorkey.batch import jobs_from_data, run_batch
from corridorkey.session import InferenceSession
from test_spatial_draft import cpu_spatial_context, make_spatial_cpu_model
with tempfile.TemporaryDirectory(prefix='architecture-audit-') as folder:
    root = Path(folder)
    for name in ('plate', 'second'):
        (root / name).mkdir()
        assert cv2.imwrite(str(root / name / '0000.png'), np.full((64,64,3),128,np.uint8))
    hint = root / 'render/plate_alpha_exr/alpha_000000.exr'
    hint.parent.mkdir(parents=True)
    assert cv2.imwrite(str(hint), np.full((64,64),.9,np.float32))
    original = hint.read_bytes()
    settings = {'output_format':'exr','outputs':['alpha'],'exr_dtype':'float','hann_chunk':8,'hann_stride':4,'shape_quantum':64,'temp_dir':str(root/'scratch')}
    manifest = {'schema_version':1,'kind':'inference_batch','jobs':[
        {'input':str(root/'plate'),'output_dir':str(root/'render'),'settings':settings},
        {'input':str(root/'second'),'output_dir':str(root/'render2'),'settings':settings,'hint_first_frame':str(hint)}]}
    jobs,_ = jobs_from_data(manifest,root/'batch.json')
    def factory(**kwargs):
        return InferenceSession(**kwargs,loader=make_spatial_cpu_model,disposer=lambda model:None)
    with cpu_spatial_context():
        report=run_batch(jobs,session_factory=factory)
    print('ARCH-001_RESULT',json.dumps({'batch_status':report['status'],'jobs':len(report['jobs']),'second_job_hint_changed':hint.read_bytes()!=original,'before_alpha':.9,'after_alpha':float(cv2.imread(str(hint),-1)[0,0])}))
    assert report['status']=='complete' and hint.read_bytes()!=original
```

Command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /private/tmp/corridorkey-compatible-venv/bin/python REPRO.txt`.

## ARCH-002

```python
import json,tempfile
from pathlib import Path
from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest
from corridorkey.output_paths import MAX_SCAN_ENTRIES,output_layout,validate_report_destination
with tempfile.TemporaryDirectory(prefix='architecture-growth-') as folder:
    root=Path(folder); source=root/'plate'; source.mkdir(); render=root/'render'
    for index in range(17000): (source/f'{index:06d}.png').touch()
    config=InferenceConfig(output_format='exr',outputs={'alpha','fg'})
    job=JobRequest.create(source,render,config)
    validate_report_destination(root/'report.json',(job,))
    for output in output_layout(source,config)['outputs'].values():
        target=render/output['path']; target.mkdir(parents=True)
        for index in range(17000): (target/f"{output['frame_name']}_{index:06d}.exr").touch()
    try: validate_report_destination(root/'report.json',(job,))
    except ValueError as error:
        print('ARCH-002_GROWTH',json.dumps({'initial_admission':'PASS','after_output_growth':'FAIL','source_frames':17000,'output_files':34000,'limit':MAX_SCAN_ENTRIES,'error':str(error)}))
        assert str(error)=='report protection inventory entry limit exceeded'
    else: raise AssertionError('ARCH-002 not reproduced')
```
