import io
import json
import zipfile
import pandas as pd
import pytest
from experiment_v12 import open_release, run_experiment, export_experiment
from core import generate_demo_data
from dataset_release import release_bundle

def fixture():
    df=generate_demo_data(n=80,seed=11)
    df['apk_sha256']=[f'{i+1:064x}' for i in range(len(df))]
    return release_bundle(df,seed=42)[0]

def test_good_release():
    a,b,m,audit=open_release(fixture())
    assert len(a)+len(b)==80
    assert 'source_archive_sha256' in audit

def test_baseline_run():
    result=run_experiment(fixture(),models=('baseline',),seed=19)
    assert len(result['metrics'])==1
    out=export_experiment(result)
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert 'heldout_predictions.csv' in z.namelist()

def test_missing_files():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:z.writestr('train.csv','sample_id,label,events\na,benign,hello')
    with pytest.raises(ValueError,match='Release needs'):open_release(b.getvalue())

def test_bad_zip_path():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('../bad.csv','evil')
        for f in ('train.csv','test.csv'):z.writestr(f,'sample_id,label,events\na,benign,hello')
        z.writestr('release_manifest.json','{}')
    with pytest.raises(ValueError,match='Unsafe'):open_release(b.getvalue())

def test_hash_leak():
    df=generate_demo_data(n=80,seed=11)
    df['apk_sha256']=[f'{i+1:064x}' for i in range(80)]
    b,meta=release_bundle(df)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        d={x.filename:z.read(x.filename) for x in z.infolist()}
    train=pd.read_csv(io.BytesIO(d['train.csv']))
    test=pd.read_csv(io.BytesIO(d['test.csv']))
    test.loc[0,'apk_sha256']=train.loc[0,'apk_sha256']
    d['test.csv']=test.to_csv(index=False).encode()
    import hashlib
    m=json.loads(d['release_manifest.json'])
    m['test_csv_sha256']=hashlib.sha256(d['test.csv']).hexdigest()
    d['release_manifest.json']=json.dumps(m).encode()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for k,v in d.items():z.writestr(k,v)
    with pytest.raises(ValueError,match='APK hash leakage'):open_release(out.getvalue())

def test_reject_duplicate_zip_filenames():
    b=fixture()
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        members={x.filename:z.read(x.filename) for x in z.infolist()}
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,data in members.items(): z.writestr(name,data)
        z.writestr('train.csv',members['train.csv'])
    with pytest.raises(ValueError,match='Duplicate filenames'):open_release(out.getvalue())

def test_rejects_edited_csv_when_manifest_has_checksum():
    b=fixture()
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        members={x.filename:z.read(x.filename) for x in z.infolist()}
    members['train.csv']+=b'\n'
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,data in members.items(): z.writestr(name,data)
    with pytest.raises(ValueError,match='does not match CSV'):open_release(out.getvalue())

def test_rejects_missing_behavior_rows():
    b=fixture()
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        members={x.filename:z.read(x.filename) for x in z.infolist()}
    df=pd.read_csv(io.BytesIO(members['train.csv']))
    df.loc[0,'events']=' | '
    members['train.csv']=df.to_csv(index=False).encode()
    meta=json.loads(members['release_manifest.json'])
    import hashlib
    meta['train_csv_sha256']=hashlib.sha256(members['train.csv']).hexdigest()
    members['release_manifest.json']=json.dumps(meta).encode()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,data in members.items():z.writestr(name,data)
    with pytest.raises(ValueError,match='empty behavioural sequence'):open_release(out.getvalue())

def test_no_models_not_allowed():
    with pytest.raises(ValueError,match='at least one unique'):run_experiment(fixture(),models=[])

def test_seeded_transformer_is_reproducible():
    b=fixture()
    a=run_experiment(b,models=['transformer'],epochs=1,seed=21)
    c=run_experiment(b,models=['transformer'],epochs=1,seed=21)
    import numpy as np
    np.testing.assert_allclose(a['predictions']['transformer_p_malicious'],c['predictions']['transformer_p_malicious'],atol=1e-7)
