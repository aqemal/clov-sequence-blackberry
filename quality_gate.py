"""Read-only dataset quality gate and reproducible snapshot creation.

Input: pre-extracted CSV data. Never downloads, opens, or executes APK packages.
A PASS means structural/data-integrity checks, not ground-truth verification.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import io, json, re, zipfile
import pandas as pd
from core import parse_events

ALLOWED = {'benign','malicious','0','1'}
HASH = re.compile(r'^[a-f0-9]{64}$')
MAX_ROWS = 100_000
MAX_EVENTS = 10000

def _label(v):
    s=str(v).strip().lower()
    return {'0':'benign','1':'malicious'}.get(s,s)

def assess_dataset(frame: pd.DataFrame, *, min_events=2, require_hash=False):
    if not {'sample_id','label','events'}.issubset(frame.columns):
        raise ValueError('Required columns: sample_id, label, events')
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError('Dataset must have 1 to 100,000 rows')
    if not isinstance(min_events,int) or not 1 <= min_events <= 100:
        raise ValueError('min_events must be an integer from 1 to 100')
    if require_hash and 'apk_sha256' not in frame.columns:
        raise ValueError('apk_sha256 column is mandatory when hash validation is enabled')
    details=[]
    for idx,row in frame.reset_index(drop=True).iterrows():
        sid='' if pd.isna(row['sample_id']) else str(row['sample_id']).strip()
        lab=_label(row['label'])
        raw=row['events']
        events=[] if pd.isna(raw) else parse_events(raw)
        rawhash=row.get('apk_sha256','')
        hv='' if pd.isna(rawhash) else str(rawhash).strip().lower()
        warnings=[];errors=[]
        if not sid or sid.lower() in ('nan','none'):errors.append('missing_sample_id')
        if lab not in ('benign','malicious'):errors.append('invalid_label')
        if len(events)<min_events:errors.append('insufficient_events')
        if len(events)>MAX_EVENTS:errors.append('excessive_events')
        if require_hash and not HASH.fullmatch(hv):errors.append('missing_or_invalid_sha256')
        elif hv and not HASH.fullmatch(hv):warnings.append('invalid_optional_sha256')
        if len(events)>0 and len(set(events))==1:warnings.append('one_distinct_event')
        if len(events)>0 and len(set(events))<=2 and len(events)>20:warnings.append('low_event_diversity')
        details.append({'row_index':idx,'sample_id':sid,'label':lab,
            'malware_family':str(row.get('malware_family','')),
            'apk_sha256':hv,'event_count':len(events),'unique_events':len(set(events)),
            'sequence_sha256':sha256(json.dumps(events,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
            'errors':errors,'warnings':warnings,'events_clean':' | '.join(events)})
    by_id={};by_hash={};by_seq={}
    for d in details:
        for key,storage in [('sample_id',by_id),('apk_sha256',by_hash),('sequence_sha256',by_seq)]:
            v=d[key]
            if v:storage.setdefault(v,[]).append(d)
    for name,storage,reason in [('sample_id',by_id,'duplicate_sample_id'),('apk_sha256',by_hash,'duplicate_apk_hash'),('sequence_sha256',by_seq,'duplicate_event_sequence')]:
        for group in storage.values():
            if len(group)>1:
                for d in group:d['errors'].append(reason)
                if len({d['label'] for d in group})>1:
                    for d in group:d['errors'].append('conflicting_labels_for_'+name)
    audit=[]; accepted=[]
    for d in details:
        audit.append({k:v for k,v in d.items() if k!='events_clean'} | {
            'status':'REJECT' if d['errors'] else 'ACCEPT',
            'errors':'; '.join(sorted(set(d['errors']))),
            'warnings':'; '.join(sorted(set(d['warnings']))),})
        if not d['errors']:
            accepted.append({'sample_id':d['sample_id'],'label':d['label'],
                'malware_family':d['malware_family'] or ('benign' if d['label']=='benign' else 'unknown'),
                'events':d['events_clean'],'apk_sha256':d['apk_sha256'],
                'sequence_sha256':d['sequence_sha256']})
    aud=pd.DataFrame(audit)
    clean=pd.DataFrame(accepted,columns=['sample_id','label','malware_family','events','apk_sha256','sequence_sha256'])
    kinds=Counter(aud.label)
    summary={'total':len(frame),'accepted':len(clean),'rejected':len(frame)-len(clean),
             'class_counts_input':dict(kinds),'class_counts_accepted':dict(Counter(clean.label)),
             'min_events':min_events,'require_hash':require_hash,
             'error_frequency':dict(Counter(x for d in details for x in set(d['errors']))),
             'warning_frequency':dict(Counter(x for d in details for x in set(d['warnings']))),
             'provenance_warning':'This validates structure and consistency only, not whether an APK ran in a sandbox or whether labels are correct.'}
    return clean,aud,summary

def snapshot_zip(clean, audit, summary, *, source_label='pre_extracted_csv', seed=42):
    """Byte-stable content hashes; ZIP may have different timestamps between exports."""
    if clean.empty:raise ValueError('Cannot export a snapshot with zero accepted samples')
    ordered=clean.sort_values('sample_id',kind='stable').reset_index(drop=True)
    csv=ordered.to_csv(index=False,lineterminator='\n').encode('utf-8')
    aud=audit.to_csv(index=False,lineterminator='\n').encode('utf-8')
    manifest={'release_format':'Sequence-CloV v11 quality snapshot',
        'input_kind':source_label,'seed_for_downstream_experiments':int(seed),
        'accepted_csv_sha256':sha256(csv).hexdigest(),
        'audit_csv_sha256':sha256(aud).hexdigest(),
        'rows_accepted':len(ordered),'quality_summary':summary,
        'split_policy':'Create grouped holdout in the existing Leakage-aware training or model benchmark tab. This package does not claim to be a trained model.',
        'provenance':'User-supplied data. SHA-256 fingerprints are computed on exported CSV bytes; they do not authenticate labels or sandbox execution.'}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('accepted_traces.csv',csv)
        z.writestr('quality_audit.csv',aud)
        z.writestr('snapshot_manifest.json',json.dumps(manifest,indent=2,sort_keys=True))
    return buf.getvalue(),manifest
