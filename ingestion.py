"""Safe metadata inventory and sanitized behavioural trace conversion.

Never executes or imports APK content. Designed for local, authorized datasets.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import pandas as pd

HEX_NAME = re.compile(r'^[a-fA-F0-9]{32,64}$')
ALLOWED_REPORT_EXTS = {'.csv', '.json', '.jsonl'}
EVENT_FIELDS = ('event', 'event_type', 'action', 'api_call', 'api', 'type', 'name')


def apk_inventory(root, compute_sha256=False, max_files=100000):
    """Read APK file metadata under family/variety hierarchy; never parse/execute APKs.

    Does not follow symbolic links. Hashing is optional and reads bytes only.
    """
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('The supplied dataset location must be a directory.')
    rows = []
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != '.apk':
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        family = parts[0] if len(parts) >= 2 else 'unassigned'
        variety = parts[1] if len(parts) >= 3 else 'unspecified'
        digest = None
        if compute_sha256:
            h = hashlib.sha256()
            with path.open('rb') as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                    h.update(chunk)
            digest = h.hexdigest()
        rows.append({'sample_id': path.stem, 'family': family, 'variety': variety,
                     'filename': path.name, 'relative_path': relative.as_posix(),
                     'size_bytes': path.stat().st_size, 'sha256': digest})
        if len(rows) >= max_files:
            raise ValueError(f'Inventory limit of {max_files} APKs reached. Filter dataset or raise the explicit limit.')
    return pd.DataFrame(rows, columns=['sample_id','family','variety','filename','relative_path','size_bytes','sha256'])


def inventory_summary(df):
    if df.empty:
        return {'samples':0, 'families':0,'varieties':0,'duplicate_names':0,'total_mb':0}
    return {'samples':len(df), 'families':int(df.family.nunique()),
            'varieties':len(df[['family','variety']].drop_duplicates()),
            'duplicate_names':int(df.sample_id.duplicated(keep=False).sum()),
            'total_mb':round(float(df.size_bytes.sum())/1_000_000,2)}


def _normalize_event(value):
    if isinstance(value, str):
        s=value.strip()
    elif isinstance(value, dict):
        s=next((str(value[k]).strip() for k in EVENT_FIELDS if value.get(k) is not None), '')
        # When an event includes a category, preserve that category in the token.
        category=str(value.get('category','')).strip()
        if category and ':' not in s:
            s=f'{category}:{s}'
    else:
        s=''
    return re.sub(r'\s+', '_',s)[:160]


def _one_trace(item, default_id):
    if not isinstance(item,dict):
        raise ValueError('Every trace must be a JSON object.')
    sample_id=str(item.get('sample_id') or item.get('apk_hash') or item.get('sha256') or default_id).strip()
    label=str(item.get('label') or item.get('class') or '').strip().lower()
    family=str(item.get('malware_family') or item.get('family') or '').strip()
    raw_events=item.get('events',item.get('trace',item.get('behaviors')))
    if not isinstance(raw_events,list):
        raise ValueError(f'{sample_id}: events must be an ordered list.')
    # Event order is kept from the original list: do not sort timestamps automatically.
    events=[_normalize_event(x) for x in raw_events]
    events=[x for x in events if x]
    if not events:
        raise ValueError(f'{sample_id}: zero usable events.')
    if label not in ('benign','malicious','0','1'):
        # Family-only logs are known-malware examples, not evidence of benign class.
        label='malicious' if family and family.lower() != 'benign' else ''
    if not label:
        raise ValueError(f'{sample_id}: missing benign/malicious label (or malware family).')
    return {'sample_id':sample_id,'label':label,'events':' | '.join(events),'malware_family':family or 'unknown'}


def traces_to_csv_dataframe(fileobj, filename):
    """Convert sanitised CSV/JSON/JSONL reports to existing Sequence-CloV columns."""
    suffix=Path(filename).suffix.lower()
    if suffix not in ALLOWED_REPORT_EXTS:
        raise ValueError('Supported input formats: CSV, JSON, JSONL (not APK).')
    if suffix=='.csv':
        df=pd.read_csv(fileobj)
        for key in ('sample_id','label','events'):
            if key not in df:
                raise ValueError('CSV must contain sample_id, label, and events columns.')
        return df
    raw=fileobj.read()
    if len(raw)>25_000_000:
        raise ValueError('Report too large for interactive conversion (25MB limit).')
    text=raw.decode('utf-8-sig') if isinstance(raw,bytes) else raw
    if suffix=='.jsonl':
        records=[json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        source=json.loads(text)
        records=source if isinstance(source,list) else source.get('samples', [source])
    if not isinstance(records,list):
        raise ValueError('Expected an array of traces or an object with samples array.')
    return pd.DataFrame([_one_trace(item,f'report_{i+1}') for i,item in enumerate(records)],
                        columns=['sample_id','label','events','malware_family'])


def write_manifest(df, target):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(target,index=False)
    return target
