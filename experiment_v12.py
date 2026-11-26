"""Version 12: offline, reproducible evaluation of exported behavioural-trace releases.
Does not handle APK binaries or claim independent sandbox authenticity.
"""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
import re
from pathlib import PurePosixPath
import pandas as pd
from verified_training import inspect_partitions
from sequence_benchmarks import compare_models

LIMIT = 30_000_000
REQUIRED = {'train.csv', 'test.csv', 'release_manifest.json'}

def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

def open_release(blob: bytes):
    if len(blob) > LIMIT:
        raise ValueError('Release archive exceeds 30 MB')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        files = z.infolist()
        if len(files) > 50:
            raise ValueError('Too many archive entries')
        for f in files:
            p = PurePosixPath(f.filename.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts or ((f.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unsafe ZIP entry')
            if f.file_size > LIMIT:
                raise ValueError('Oversized archive member')
        names_list = [x.filename for x in files if not x.is_dir()]
        if len(names_list) != len(set(names_list)):
            raise ValueError('Duplicate filenames in release ZIP')
        if any(x.flag_bits & 1 for x in files):
            raise ValueError('Encrypted ZIP entries are not supported')
        names = set(names_list)
        if not REQUIRED.issubset(names):
            raise ValueError('Release needs train.csv, test.csv and release_manifest.json')
        if sum(f.file_size for f in files)>LIMIT:
            raise ValueError('Uncompressed archive exceeds 30 MB')
        read = lambda n: z.read(n)
        train_bytes, test_bytes = read('train.csv'), read('test.csv')
        meta = json.loads(read('release_manifest.json').decode('utf-8-sig'))
    if not isinstance(meta, dict):
        raise ValueError('Release manifest must be an object')
    # Validate raw inputs *before* normalization; core.validate_data otherwise drops empty traces.
    raw_train = pd.read_csv(io.BytesIO(train_bytes))
    raw_test = pd.read_csv(io.BytesIO(test_bytes))
    for name, frame in [('train.csv', raw_train), ('test.csv', raw_test)]:
        required = {'sample_id', 'label', 'events', 'malware_family', 'apk_sha256'}
        if not required.issubset(frame.columns):
            raise ValueError(f'{name}: missing required columns: {sorted(required-set(frame.columns))}')
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f'{name}: empty dataset or missing required values')
        if any(frame[c].astype(str).str.strip().eq('').any() for c in required):
            raise ValueError(f'{name}: blank required values')
        if not frame.apk_sha256.astype(str).str.lower().str.fullmatch(r'[0-9a-f]{64}').all():
            raise ValueError(f'{name}: invalid APK SHA-256')
        if (frame.label.astype(str).str.lower().isin(['benign','0']) & frame.malware_family.astype(str).str.lower().ne('benign')).any():
            raise ValueError(f'{name}: benign rows must use malware_family=benign')
        # Empty token sequences such as ' | ' are not valid observations.
        from core import parse_events
        if frame.events.map(lambda v: len(parse_events(v)) == 0).any():
            raise ValueError(f'{name}: empty behavioural sequence')
    for member, value in [('train_csv_sha256', train_bytes), ('test_csv_sha256', test_bytes)]:
        declared = meta.get(member)
        if declared is not None and declared != digest(value):
            raise ValueError(f'Manifest {member} does not match CSV content')
    a, b, inspection = inspect_partitions(raw_train, raw_test)
    x=set(a.apk_sha256.astype(str).str.lower())
    y=set(b.apk_sha256.astype(str).str.lower())
    if x&y:
        raise ValueError('APK hash leakage detected across train/test')
    for partition_name, partition in [('train',a),('test',b)]:
        if len(partition) < 4:
            raise ValueError(f'{partition_name} partition needs at least 4 labelled traces')
    return a,b,meta,{'source_archive_sha256':digest(blob),'train_csv_sha256':digest(train_bytes),
                     'test_csv_sha256':digest(test_bytes),**inspection}

def class_coverage(train, test):
    def counts(df):
        return {str(k):int(v) for k,v in df.label.value_counts().items()}
    return {'train':counts(train),'test':counts(test),'train_families':int(train.malware_family.nunique()),
            'test_families':int(test.malware_family.nunique()),
            'train_samples':int(len(train)),'test_samples':int(len(test))}

def run_experiment(blob:bytes, *,models=('baseline',),epochs=4,seed=42):
    a,b,manifest,audit=open_release(blob)
    selected=list(models)
    if not selected or len(set(selected))!=len(selected):
        raise ValueError('Select at least one unique model')
    if not 1 <= int(epochs) <= 100:
        raise ValueError('epochs must be in 1..100')
    records,preds,benchmark_audit=compare_models(a,b,epochs=epochs,seed=seed,models=selected)
    return {'metrics':records,'predictions':preds,'manifest':manifest,
            'audit':{**audit,'coverage':class_coverage(a,b),'benchmark_audit':benchmark_audit},
            'configuration':{'seed':int(seed),'epochs':int(epochs),'models':list(models),
              'evaluation':'single frozen held-out test partition',
              'model_params':{'neural_max_sequence_length':48,'neural_learning_rate':0.003,'classification_threshold':0.5},
              'versions':{'torch':__import__('torch').__version__,'numpy':__import__('numpy').__version__,
                          'pandas':pd.__version__,'sklearn':__import__('sklearn').__version__},
              'limitation':'No proof of genuine sandbox execution or validated ground truth; no independent external test.'}}

def export_experiment(result):
    metrics=result['metrics'];predictions=result['predictions']
    serial={'metrics':metrics,'configuration':result['configuration'],'audit':result['audit'],
            'source_release_manifest':result['manifest']}
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('metrics.json',json.dumps(serial,indent=2,default=str))
        z.writestr('metrics.csv',pd.DataFrame([{k:v for k,v in m.items() if k!='confusion_matrix'} for m in metrics]).to_csv(index=False))
        z.writestr('heldout_predictions.csv',predictions.to_csv(index=False))
        z.writestr('README.txt','Sequence-CloV v12 evaluation archive. Do not interpret synthetic example metrics as real-world malware performance. Check dataset provenance and labels before publication. Test data were used for reporting only, not for hyperparameter tuning.\n')
    return output.getvalue()
