"""Offline adapters for pre-exported, sanitized sandbox JSON reports.

No APKs are executed or inspected. The supported generic process/call shape is
similar to common sandbox exports but NOT a guarantee of vendor compatibility.
Never load arbitrary pickle or untrusted Python modules.
"""
import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

ALLOWED_FIELDS = ('api','api_call','event','event_type','action','name','method')
TOKEN_RE = re.compile(r'[^A-Za-z0-9_.:/-]+')

def token(event):
    if isinstance(event, str):
        source=event
        category=''
    elif isinstance(event,dict):
        source=next((str(event[k]) for k in ALLOWED_FIELDS if event.get(k) not in (None,'')), '')
        category=str(event.get('category','')).strip()
    else:
        return ''
    clean=TOKEN_RE.sub('_',source.strip())[:128].strip('_')
    cat=TOKEN_RE.sub('_',category)[:32].strip('_')
    return (cat+':'+clean if cat and clean else clean)

def read_report(data, *, max_bytes=20_000_000):
    if isinstance(data, bytes):
        if len(data)>max_bytes: raise ValueError('JSON report exceeds 20 MB limit')
        data=data.decode('utf-8-sig')
    if isinstance(data,str):
        if len(data.encode('utf-8'))>max_bytes: raise ValueError('JSON report exceeds 20 MB limit')
        data=json.loads(data)
    if not isinstance(data,dict): raise ValueError('Each sandbox report must be a JSON object')
    return data

def extract_events(report):
    """Return ordered event tokens and explicit uncertainty notes.

    Accepted shapes: {events:[...]}, {behavior:{processes:[{calls:[...]}]}},
    and {processes:[{calls:[...]}]}. Never fabricate events from an APK.
    """
    notes=[]
    if isinstance(report.get('events'), list):
        raw=report['events']
        origin='events'
    else:
        behavior=report.get('behavior',{})
        if not isinstance(behavior,dict): behavior={}
        processes=behavior.get('processes',report.get('processes',[]))
        if not isinstance(processes,list): processes=[]
        raw=[]
        for p in processes:
            if isinstance(p,dict) and isinstance(p.get('calls'),list):
                for call in p['calls']:
                    if isinstance(call,dict):
                        c=dict(call)
                        c.setdefault('category','api')
                        raw.append(c)
                    elif isinstance(call,str):raw.append({'category':'api','api':call})
        origin='behavior.processes.calls'
        if len(processes)>1:
            notes.append('Calls from multiple processes appear in process-list order; cross-process chronology is not guaranteed.')
    if not raw: raise ValueError('No supported ordered events found (expected events[] or behavior.processes[].calls[]).')
    # Preserve source array order. Deliberately do not infer timestamps/sort.
    events=[token(e) for e in raw]
    events=[e for e in events if e]
    if not events: raise ValueError('Report contains no usable event names')
    if not all(isinstance(e,dict) and any(k in e for k in ('timestamp','time','ts')) for e in raw):
        notes.append('Timestamps missing from one or more events; order follows report array, not verified execution time.')
    return events,origin,notes

def adapt_report(payload, *, sample_id=None, family=None, label=None, source_name='report.json', manifest=None):
    r=read_report(payload)
    inferred_id=str(r.get('sample_id') or r.get('sha256') or r.get('apk_hash') or '').strip()
    effective_id=str(sample_id or inferred_id).strip()
    if not effective_id: raise ValueError('sample_id is required (in report or provided explicitly).')
    effective_family=str(family or r.get('malware_family') or r.get('family') or '').strip()
    effective_label=str(label or r.get('label') or '').strip().lower()
    # Malware family from user-controlled metadata is treated as an annotation, not proof of maliciousness.
    if effective_label not in ('benign','malicious'):
        raise ValueError('Explicit verified benign/malicious label required; a family name alone is insufficient.')
    ev,origin,notes=extract_events(r)
    row={'sample_id':effective_id,'label':effective_label,'events':' | '.join(ev),
         'malware_family':effective_family or ('benign' if effective_label=='benign' else 'unknown'),
         'event_count':len(ev),'source_report':Path(source_name).name,'source_schema':origin,
         'integrity_status':'not_checked','warnings':' '.join(notes)}
    if manifest is not None:
        if not isinstance(manifest,pd.DataFrame) or 'sample_id' not in manifest.columns:
            raise ValueError('manifest must be a DataFrame with sample_id')
        matches=manifest.loc[manifest.sample_id.astype(str)==effective_id]
        if len(matches)==1:
            m=matches.iloc[0]
            if pd.notna(m.get('family')) and effective_family and str(m['family'])!=effective_family:
                row['integrity_status']='family_mismatch'
            else: row['integrity_status']='id_matched_only'
            # Matching filename stems does NOT verify the report belongs to APK.
            row['warnings']+=' Manifest sample_id match is not cryptographic provenance verification.'
        elif len(matches)>1: row['integrity_status']='ambiguous_id'
        else: row['integrity_status']='unmatched_id'
    return row

def adapt_many(reports, manifest=None):
    """reports: iterable of (source filename, JSON dict/str, overrides dict)."""
    rows=[];errors=[]
    for filename,payload,overrides in reports:
        try: rows.append(adapt_report(payload,source_name=filename,manifest=manifest,**(overrides or {})))
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc: errors.append({'source_report':filename,'error':str(exc)})
    df=pd.DataFrame(rows)
    if not df.empty and df.sample_id.duplicated().any():
        raise ValueError('Duplicate sample_id values detected. Resolve duplicates before exporting.')
    return df,pd.DataFrame(errors,columns=['source_report','error'])

def training_export(df):
    required={'sample_id','label','events'}
    if not required.issubset(df.columns): raise ValueError('Missing model-ready columns')
    return df[['sample_id','label','events','malware_family']].copy()
