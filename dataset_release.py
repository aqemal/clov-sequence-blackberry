"""Offline, non-executing research dataset release builder.
Consumes *exported* behavioural JSON reports only. Refuses executables and ZIP symlinks.
A report-declared hash or user-provided label is not independently authenticated.
"""
from __future__ import annotations
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import PurePosixPath
import pandas as pd
from sandbox_adapter import adapt_many
from verified_training import make_split

MAX_REPORT_BYTES = 5_000_000
MAX_REPORTS = 2000
MAX_TOTAL_BYTES = 50_000_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")

def _read_zip(blob:bytes):
    if len(blob)>MAX_TOTAL_BYTES:raise ValueError("ZIP exceeds 50 MB upload limit")
    reports=[];errors=[];total=0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        candidates=[i for i in z.infolist() if not i.is_dir()]
        if len(candidates)>MAX_REPORTS:raise ValueError("Too many ZIP entries")
        for i in candidates:
            p=PurePosixPath(i.filename.replace("\\","/"))
            if '..' in p.parts or p.is_absolute():
                errors.append({'source_report':i.filename,'error':'unsafe_archive_path'});continue
            # unix symlink file mode 0120000
            if ((i.external_attr>>16)&0o170000)==0o120000:
                errors.append({'source_report':i.filename,'error':'symlink_not_allowed'});continue
            if p.suffix.lower() not in ('.json','.jsonl'):
                errors.append({'source_report':i.filename,'error':'unsupported_extension; only JSON/JSONL exported logs allowed'});continue
            if i.file_size>MAX_REPORT_BYTES:
                errors.append({'source_report':i.filename,'error':'report_exceeds_5_MB'});continue
            total+=i.file_size
            if total>MAX_TOTAL_BYTES:raise ValueError('Uncompressed reports exceed 50 MB cumulative limit')
            try:
                with z.open(i) as f:raw=f.read(MAX_REPORT_BYTES+1)
                if len(raw)>MAX_REPORT_BYTES:raise ValueError('Report too large')
                if p.suffix.lower()=='.json':
                    reports.append((i.filename,json.loads(raw.decode('utf-8-sig')),{}))
                else:
                    for line_no,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
                        if line.strip():
                            reports.append((f'{i.filename}:{line_no}',json.loads(line),{}))
                if len(reports)>MAX_REPORTS:raise ValueError('More than 2000 reports')
            except (ValueError,UnicodeError,json.JSONDecodeError) as exc:
                errors.append({'source_report':i.filename,'error':str(exc)})
    return reports,errors

def ingest_report_archive(blob:bytes, labels:pd.DataFrame, *, require_hash=True):
    """labels CSV must explicitly carry sample_id,label,malware_family,apk_sha256.
    Label provenance is user-attested, not independently verified.
    """
    fields={'sample_id','label','malware_family','apk_sha256'}
    if not fields.issubset(labels.columns):raise ValueError('Labels CSV requires: '+', '.join(sorted(fields)))
    l=labels.copy().fillna('')
    for col in fields:l[col]=l[col].astype(str).str.strip()
    l['apk_sha256']=l.apk_sha256.str.lower()
    if l.sample_id.eq('').any() or l.sample_id.duplicated().any():raise ValueError('Each sample_id must be nonblank and unique')
    if not l.label.str.lower().isin(['malicious','benign']).all():raise ValueError('Explicit benign/malicious labels required')
    if require_hash and not l.apk_sha256.map(lambda v:bool(HEX64.fullmatch(v))).all():raise ValueError('Valid 64-digit SHA-256 required for all label rows')
    reports,errors=_read_zip(blob)
    by_id=l.set_index('sample_id')
    accepted=[];seen=set()
    for filename,payload,_ in reports:
        try:
            if not isinstance(payload,dict):raise ValueError('Report must be an object')
            sid=str(payload.get('sample_id','')).strip()
            if sid not in by_id.index:raise ValueError('sample_id absent from verified-label manifest')
            if sid in seen:raise ValueError('Duplicate sample_id in reports')
            manifest=by_id.loc[sid]
            declared_hash=str(payload.get('apk_sha256') or payload.get('sha256') or '').strip().lower()
            trusted_hash=manifest.apk_sha256
            if require_hash and (not HEX64.fullmatch(declared_hash) or declared_hash!=trusted_hash):
                raise ValueError('Report-declared APK SHA-256 absent/invalid or disagrees with label manifest')
            payload=dict(payload)
            # Always override self-reported class/family with separate label manifest.
            payload['label']=manifest.label.lower()
            payload['malware_family']=manifest.malware_family
            from sandbox_adapter import adapt_report
            row=adapt_report(payload,source_name=filename,sample_id=sid,label=manifest.label.lower(),family=manifest.malware_family)
            row['apk_sha256']=trusted_hash
            row['hash_match']='declared_matches_manifest' if declared_hash==trusted_hash and declared_hash else 'unverified'
            row['label_evidence']=str(manifest.get('label_evidence','unspecified'))
            row['sandbox_environment']=str(payload.get('sandbox_environment','unspecified'))
            row['report_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
            row['event_order_status']='report_array_order_only'
            seen.add(sid);accepted.append(row)
        except (ValueError,TypeError,KeyError) as exc:errors.append({'source_report':filename,'error':str(exc)})
    return pd.DataFrame(accepted),pd.DataFrame(errors,columns=['source_report','error'])

def release_bundle(rows:pd.DataFrame, *, seed=42, test_fraction=.25, group_key='sample_id'):
    """Create a reproducible ZIP with frozen partitions, metadata and rejection audit.
    No real-world performance claims; no APK handling.
    """
    needed={'sample_id','label','events','apk_sha256','malware_family'}
    if not needed.issubset(rows.columns):raise ValueError('Missing required report columns')
    if rows.empty:raise ValueError('No usable reports')
    if rows.apk_sha256.duplicated().any():
        # Multiple runs of identical APK could be valid, but should be reviewed before splitting.
        raise ValueError('Repeated APK hashes found; review duplicated APKs before dataset release')
    frame=rows[['sample_id','label','events','malware_family','apk_sha256']].copy()
    tr,te,audit=make_split(frame,seed=int(seed),test_fraction=float(test_fraction),group_key=group_key)
    splitmeta={'schema_version':'v8.0','source_kind':'externally_exported_sandbox_reports',
       'classification':'malware vs benign','seed_requested':int(seed),'seed_used':audit.get('split_seed'),
       'test_fraction_requested':float(test_fraction),'group_key':group_key,
       'provenance_limitation':'Labels and reported hashes are user-provided; matching declarations do not attest sandbox execution.',
       'timing_limitation':'Event order reflects exported report array order; cross-process wall-clock time is not verified.',
       'audit':audit}
    train_bytes=tr.to_csv(index=False).encode('utf-8')
    test_bytes=te.to_csv(index=False).encode('utf-8')
    splitmeta['train_csv_sha256']=hashlib.sha256(train_bytes).hexdigest()
    splitmeta['test_csv_sha256']=hashlib.sha256(test_bytes).hexdigest()
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('train.csv',train_bytes);z.writestr('test.csv',test_bytes)
        z.writestr('accepted_provenance.csv',rows.to_csv(index=False))
        z.writestr('release_manifest.json',json.dumps(splitmeta,indent=2,default=str))
        z.writestr('DATA_DICTIONARY.txt','sample_id: report-linked sample identifier\nlabel: explicit benign/malicious user assertion\nevents: pipe-delimited event tokens in report order\nmalware_family: user-supplied family metadata\napk_sha256: user-provided application hash\nsequence_fingerprint: normalized sequence SHA-256, used for audit\n')
    return bio.getvalue(),splitmeta
