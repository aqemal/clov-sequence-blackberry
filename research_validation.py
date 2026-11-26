"""Offline quality checks for pre-exported behavioural traces and APK manifests.
No execution, decompilation or networking. Hash matching corroborates metadata only.
"""
import json
import re
from collections import Counter
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SHA256 = re.compile(r"^[0-9a-f]{64}$")

def _str(x):
    return "" if pd.isna(x) else str(x).strip()

def check_provenance(manifest, reports):
    """Manifest columns: sample_id, sha256, family. Reports: sample_id, apk_sha256, malware_family.
    Never claim report authenticity merely because declared hashes match.
    """
    if 'sha256' not in manifest or 'apk_sha256' not in reports:
        raise ValueError('Provide sha256 in manifest and apk_sha256 in reports.')
    m=manifest.copy(); r=reports.copy()
    m['norm_sha']=m.sha256.map(lambda x:_str(x).lower())
    r['norm_sha']=r.apk_sha256.map(lambda x:_str(x).lower())
    m=m[m.norm_sha.map(lambda x:bool(SHA256.fullmatch(x)))]
    counts=Counter(m.norm_sha)
    indexed=m.groupby('norm_sha',sort=False).first()
    statuses=[]
    for _,item in r.iterrows():
        h=item.norm_sha
        if not SHA256.fullmatch(h): status='missing_or_invalid_report_hash'
        elif h not in counts:status='hash_not_in_manifest'
        else:
            recorded=indexed.loc[h]
            family=_str(item.get('malware_family',''))
            if family and _str(recorded.get('family','')) and family!=_str(recorded['family']):
                status='family_conflict'
            elif counts[h]>1:status='matched_hash_duplicate_manifest_entries'
            else:status='declared_hash_matches_manifest'
        statuses.append(status)
    r['hash_verification_status']=statuses
    r['provenance_note']='Declared hash match is not proof that the sandbox executed or generated this report from the corresponding APK.'
    return r.drop(columns=['norm_sha'])

def ordered_events(report):
    """Extract timestamp-bearing event arrays only; refuse misleading ordering.
    Supports events[] and flattened behavior.processes[].calls[].
    Returns (event names, status) with deterministic timestamp order where comparable.
    """
    from sandbox_adapter import token
    if isinstance(report.get('events'),list):raw=report['events']
    else:
        procs=report.get('behavior',{}).get('processes',[])
        if not isinstance(procs,list):raise ValueError('Invalid processes')
        raw=[]
        for p in procs:
            if not isinstance(p,dict):continue
            raw.extend(p.get('calls',[]) if isinstance(p.get('calls'),list) else [])
    if not raw:raise ValueError('No behavioural events available')
    records=[]
    for ix,e in enumerate(raw):
        if not isinstance(e,dict):return [token(x) for x in raw if token(x)],'array_order_only'
        t=e.get('timestamp',e.get('time',e.get('ts')))
        try:
            if t is None:raise ValueError()
            ts=float(t)
            if not (-float('inf')<ts<float('inf')):raise ValueError()
        except (TypeError,ValueError):return [token(x) for x in raw if token(x)],'array_order_only'
        records.append((ts,ix,token(e)))
    records.sort(key=lambda x:(x[0],x[1]))
    return [name for _,_,name in records if name], 'timestamp_sorted'

def audit_and_split(data, test_fraction=.25, seed=42, group_key='sample_id'):
    """Group-aware split: same group never appears in both partitions.
    Identical sequence fingerprints also grouped to reduce exact-sequence leakage.
    Reports explicit warnings where duplicate sequences occur across sample IDs.
    """
    required={'sample_id','label','events'}
    if not required.issubset(data.columns):raise ValueError('Requires sample_id,label,events')
    d=data.copy().reset_index(drop=True)
    if len(d)<4:raise ValueError('At least four rows required')
    if d[['sample_id','label','events']].isna().any().any():raise ValueError('Null required fields')
    d['sample_id']=d.sample_id.astype(str).str.strip()
    d['label']=d.label.astype(str).str.lower().str.strip()
    if not set(d.label).issubset({'benign','malicious','0','1'}):raise ValueError('Use explicit benign/malicious or 0/1 labels')
    d['label']=d.label.replace({'0':'benign','1':'malicious'})
    d['events']=d.events.astype(str).str.strip()
    if (d.events=='').any():raise ValueError('Empty event sequence')
    import hashlib
    d['sequence_fingerprint']=d.events.map(lambda x:hashlib.sha256(' | '.join(part.strip() for part in x.split('|')).encode()).hexdigest())
    conflicting=d.groupby('sequence_fingerprint').label.nunique()
    if (conflicting>1).any():raise ValueError('Identical event sequences have conflicting class labels. Resolve provenance before splitting.')
    if group_key not in d.columns:raise ValueError('Missing grouping column '+group_key)
    # Union identical sequence fingerprints and shared group IDs into connected components.
    parent=list(range(len(d)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    for col in [group_key,'sequence_fingerprint']:
        first={}
        for i,value in enumerate(d[col].astype(str)):
            if value in first:union(i,first[value])
            else:first[value]=i
    d['_group']=[find(i) for i in range(len(d))]
    if d._group.nunique()<2:raise ValueError('Not enough independent groups for split')
    splitter=GroupShuffleSplit(n_splits=1,test_size=test_fraction,random_state=seed)
    tr,te=next(splitter.split(d,groups=d._group))
    train=d.iloc[tr].drop(columns=['_group']);test=d.iloc[te].drop(columns=['_group'])
    diagnostics={
        'rows':len(d),'train_rows':len(train),'test_rows':len(test),
        'distinct_samples':int(d.sample_id.nunique()),
        'repeated_sample_ids':int(d.sample_id.duplicated().sum()),
        'duplicate_sequence_rows':int(d.sequence_fingerprint.duplicated().sum()),
        'independent_groups':int(d._group.nunique()),
        'train_class_counts':train.label.value_counts().to_dict(),
        'test_class_counts':test.label.value_counts().to_dict(),
        'group_key':group_key,
        'disjoint':not bool(set(train.sequence_fingerprint)&set(test.sequence_fingerprint)) and not bool(set(train[group_key])&set(test[group_key]))
    }
    if len(set(train.label))<2 or len(set(test.label))<2:
        diagnostics['warning']='One partition lacks both classes; acquire more independent samples or use a different validated split.'
    return train,test,diagnostics
