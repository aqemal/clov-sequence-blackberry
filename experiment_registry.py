"""Offline record registry for authorized, externally executed sandbox experiments.
No APK processing or execution. Hash equality verifies metadata consistency only.
"""
from __future__ import annotations
import csv, hashlib, io, json, re, zipfile
from datetime import datetime
from pathlib import PurePosixPath

HEX64=re.compile(r"^[0-9a-f]{64}$")
REQUIRED={"run_id","sample_id","apk_sha256","environment_id","environment_version","android_api_level","started_at_utc","finished_at_utc","status","report_file"}
ACCEPTED_STATUS={"completed","failed","timeout","aborted"}
MAX_BYTES=50_000_000
MAX_FILE_BYTES=5_000_000

def _utc(s):
    try:
        t=datetime.fromisoformat(str(s).replace('Z','+00:00'))
        if t.tzinfo is None or t.utcoffset().total_seconds()!=0:raise ValueError('UTC timezone required')
        return t
    except (TypeError,ValueError,AttributeError) as exc:raise ValueError('Invalid UTC timestamp') from exc

def _check_zip(z):
    infos=[i for i in z.infolist() if not i.is_dir()]
    if len(infos)>2000:raise ValueError('Archive contains too many entries')
    if sum(i.file_size for i in infos)>MAX_BYTES:raise ValueError('Archive uncompressed data exceeds 50 MB')
    seen=set()
    for i in infos:
        p=PurePosixPath(i.filename.replace('\\','/'))
        if p.is_absolute() or '..' in p.parts or ((i.external_attr >> 16)&0o170000)==0o120000:
            raise ValueError('Unsafe archive path or symlink')
        if i.file_size>MAX_FILE_BYTES:raise ValueError('Individual archive file exceeds 5 MB')
        if i.filename in seen:raise ValueError('Duplicate archive filename')
        seen.add(i.filename)
        if p.suffix.lower() not in ('.json','.csv'):raise ValueError('Only JSON report and CSV manifest files are accepted')
    return infos

def audit_experiment_bundle(blob:bytes):
    """Input ZIP: run_manifest.csv plus exported JSON reports. Returns (accepted, rejected, summary)."""
    if len(blob)>MAX_BYTES:raise ValueError('Archive exceeds 50 MB')
    import pandas as pd
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        infos=_check_zip(z)
        names={i.filename for i in infos}
        if 'run_manifest.csv' not in names:raise ValueError('run_manifest.csv missing from archive root')
        manifest=pd.read_csv(io.BytesIO(z.read('run_manifest.csv')),dtype=str,keep_default_na=False)
        if not REQUIRED.issubset(manifest.columns):raise ValueError('Manifest missing: '+', '.join(sorted(REQUIRED-set(manifest.columns))))
        if manifest.run_id.duplicated().any():raise ValueError('run_id values must be unique')
        accepted=[];rejected=[]
        for _,record in manifest.iterrows():
            rec=record.to_dict();why=[]
            run=str(rec['run_id']).strip();sid=str(rec['sample_id']).strip();h=str(rec['apk_sha256']).lower().strip()
            if not run or not sid:why.append('missing_run_or_sample_id')
            if not HEX64.fullmatch(h):why.append('invalid_apk_sha256')
            if not rec['environment_id'].strip() or not rec['environment_version'].strip():why.append('missing_environment_identity')
            if not re.fullmatch(r'[0-9]{1,3}',str(rec['android_api_level']).strip()):why.append('invalid_android_api_level')
            status=str(rec['status']).strip().lower()
            if status not in ACCEPTED_STATUS:why.append('invalid_status')
            try:
                start=_utc(rec['started_at_utc']);finish=_utc(rec['finished_at_utc'])
                if finish<start:why.append('finish_precedes_start')
            except ValueError:why.append('invalid_utc_timing')
            report_name=str(rec['report_file']).strip()
            report=None;report_hash=''
            if report_name not in names or not report_name.lower().endswith('.json') or report_name=='run_manifest.csv':
                why.append('report_missing_or_not_json')
            else:
                try:
                    raw=z.read(report_name);report_hash=hashlib.sha256(raw).hexdigest()
                    report=json.loads(raw)
                    if not isinstance(report,dict):raise ValueError('JSON object expected')
                    if str(report.get('sample_id','')).strip()!=sid:why.append('report_sample_id_mismatch')
                    declared=str(report.get('apk_sha256') or report.get('sha256') or '').lower().strip()
                    if declared!=h:why.append('report_apk_hash_mismatch')
                    if 'run_id' in report and str(report['run_id']).strip()!=run:why.append('report_run_id_mismatch')
                except (ValueError,UnicodeError,KeyError):why.append('invalid_report_json')
            if status!='completed':why.append('run_not_completed')
            if not rec.get('label_evidence','').strip():why.append('missing_label_evidence')
            output={**rec,'report_sha256':report_hash,'integrity_status':'metadata_consistency_only'}
            if why:rejected.append({**output,'reasons':';'.join(sorted(set(why)))})
            else:accepted.append(output)
        summary={'schema_version':'9.0','source':'externally_exported_sandbox_run_reports',
           'records':len(manifest),'accepted':len(accepted),'rejected':len(rejected),
           'limitations':['This tool never runs APKs','No cryptographic attestation of sandbox execution',
            'Declared APK hash matching does not prove execution authenticity',
            'A matching report and manifest are not independent evidence of benign/malicious ground truth']}
        return pd.DataFrame(accepted),pd.DataFrame(rejected),summary

def export_audit(accepted,rejected,summary):
    """Serialize audit and accepted exported JSON reports metadata, not malware binaries."""
    import pandas as pd
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('accepted_runs.csv',accepted.to_csv(index=False))
        z.writestr('rejected_runs.csv',rejected.to_csv(index=False))
        z.writestr('experiment_audit.json',json.dumps(summary,indent=2))
        z.writestr('README.txt','Audit only. No APK executed. Validate exported reports separately using the real-trace release tab.\n')
    return bio.getvalue()
