"""Data validation, demo trace generation, and conventional benchmark for Sequence-CloV.
All inputs are pre-extracted event sequences. Never executes APKs.
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

REQUIRED = {"sample_id", "label", "events"}
LABEL_MAP = {"benign": 0, "malicious": 1, "0": 0, "1": 1}


def parse_events(value):
    """Accept pipe-separated events or a JSON-style array of event strings."""
    import json
    s = str(value).strip()
    if s.startswith('['):
        try:
            seq = json.loads(s)
            if isinstance(seq, list):
                return [str(x).strip() for x in seq if str(x).strip()]
        except (ValueError, TypeError):
            pass
    return [x.strip() for x in s.split('|') if x.strip()]


def validate_data(df):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}. See example_dataset.csv")
    d = df.copy()
    d['sample_id'] = d['sample_id'].astype(str)
    if d['sample_id'].duplicated().any():
        raise ValueError('sample_id must be unique per row')
    label_str = d['label'].astype(str).str.strip().str.lower()
    bad = sorted(set(label_str) - set(LABEL_MAP))
    if bad:
        raise ValueError(f"Unsupported labels: {bad}. Use benign/malicious or 0/1.")
    d['target'] = label_str.map(LABEL_MAP).astype(int)
    d['event_list'] = d['events'].map(parse_events)
    d = d[d['event_list'].map(len) > 0].copy()
    if d.empty:
        raise ValueError('No usable nonempty event sequences')
    d['events'] = d['event_list'].map(lambda x: ' | '.join(x))
    if 'malware_family' not in d.columns:
        d['malware_family'] = np.where(d['target'] == 1, 'unknown', 'benign')
    d['malware_family'] = d['malware_family'].fillna('unknown').astype(str)
    return d.reset_index(drop=True)


def generate_demo_data(n=240, seed=42):
    """Entirely synthetic illustrative traces; not real malware observations."""
    rng = np.random.default_rng(seed)
    routine = ['app_launch', 'activity_resume', 'read_preferences', 'render_ui',
               'check_network', 'dns_lookup', 'http_get', 'write_cache',
               'read_contacts_permission_check', 'location_permission_check',
               'sqlite_query', 'schedule_job', 'tls_connect', 'file_read',
               'push_notification', 'app_background', 'file_write', 'api_call']
    risky = {
        'data_exfiltration': ['read_contacts', 'read_sms', 'device_id_access', 'encode_payload', 'unknown_domain_connect', 'http_post_bulk', 'outbound_transfer'],
        'privilege_abuse': ['overlay_permission', 'accessibility_service_enable', 'screen_capture', 'sensitive_api_call', 'persist_service', 'remote_command'],
        'command_control': ['unknown_domain_connect', 'beacon_timer', 'http_post_encrypted', 'remote_command', 'dynamic_code_load', 'persist_service'],
    }
    result = []
    for i in range(n):
        malicious = int(i % 2 == 1)
        family = list(risky)[int(rng.integers(0, len(risky)))] if malicious else 'benign'
        core = list(rng.choice(routine, size=int(rng.integers(7, 14)), replace=True))
        if malicious:
            signature = risky[family]
            start = int(rng.integers(1, len(core)))
            core[start:start] = list(rng.choice(signature, size=int(rng.integers(4, 7)), replace=True))
        else:
            if rng.random() < .3:
                core.insert(int(rng.integers(0, len(core))), 'http_post_bulk')  # purposeful overlap/noise
        result.append(dict(sample_id=f'DEMO-{i+1:04d}', label='malicious' if malicious else 'benign',
                           malware_family=family, events=' | '.join(core)))
    return pd.DataFrame(result)


def split_data(df, seed=42):
    counts = df['target'].value_counts()
    if len(counts) < 2 or counts.min() < 3 or len(df) < 12:
        raise ValueError('Training needs at least 12 samples, including 3 in each class; ideally many more.')
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=.25, random_state=seed,
                                            stratify=df['target'].to_numpy())
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


def event_analyzer(text):
    return parse_events(text)


def train_baseline(train, test):
    vectorizer = TfidfVectorizer(analyzer=event_analyzer, lowercase=False, binary=False)
    xtr = vectorizer.fit_transform(train['events'])
    xte = vectorizer.transform(test['events'])
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(xtr, train['target'])
    proba = clf.predict_proba(xte)[:, list(clf.classes_).index(1)]
    return {'model': clf, 'vectorizer': vectorizer, 'probability': proba,
            'metrics': score(test['target'].to_numpy(), proba)}


def score(y_true, prob):
    pred = (np.asarray(prob) >= .5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, pred)), 'precision': float(pr),
            'recall': float(rc), 'f1': float(f1),
            'confusion_matrix': confusion_matrix(y_true, pred, labels=[0,1]).tolist()}


def explain_baseline(result, events):
    """Per-event contributions to linear log-odds; valid for this linear baseline only."""
    vectorizer, clf = result['vectorizer'], result['model']
    x = vectorizer.transform([' | '.join(events)])
    weights = x.multiply(clf.coef_[0]).toarray()[0]
    terms = vectorizer.get_feature_names_out()
    return sorted([(t, float(w)) for t,w in zip(terms, weights) if abs(w)>1e-12], key=lambda z: abs(z[1]), reverse=True)
