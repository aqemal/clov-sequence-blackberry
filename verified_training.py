"""Research training over explicitly separate leakage-audited partitions.
Offline only. Never opens or executes APKs. Splits must be backed by real extracted traces.
"""
import json
import numpy as np
import pandas as pd
from core import validate_data, train_baseline
from transformer_model import train_transformer
from research_validation import audit_and_split


def inspect_partitions(train, test, group_key='sample_id'):
    a, b = validate_data(train), validate_data(test)
    for name, part in [('training',a),('testing',b)]:
        if part.target.nunique()<2:
            raise ValueError(f'{name} partition lacks both benign and malicious classes')
    # No identical event sequence anywhere across partitions, even under different sample IDs.
    overlapping_sequences=set(a.events).intersection(set(b.events))
    overlapping_ids=set(a.sample_id).intersection(set(b.sample_id))
    if overlapping_ids or overlapping_sequences:
        raise ValueError(f'Leakage detected: {len(overlapping_ids)} overlapping sample IDs and {len(overlapping_sequences)} identical event sequences')
    if group_key!='sample_id':
        if group_key not in a or group_key not in b:raise ValueError(f'Missing {group_key} in train/test')
        overlap=set(a[group_key].astype(str)).intersection(set(b[group_key].astype(str)))
        if overlap:raise ValueError(f'Leakage detected: {len(overlap)} overlapping {group_key} values')
    return a, b, {'train_rows':len(a),'test_rows':len(b), 'train_classes':a.label.value_counts().to_dict(),
        'test_classes':b.label.value_counts().to_dict(),'unique_train_samples':a.sample_id.nunique(),
        'unique_test_samples':b.sample_id.nunique(),'no_exact_sequence_or_id_overlap':True,'group_key':group_key}


def make_split(frame, seed=42, test_fraction=.25, group_key='sample_id', attempts=40):
    last=None
    for offset in range(attempts):
        train,test,audit=audit_and_split(frame, test_fraction=test_fraction,seed=seed+offset,group_key=group_key)
        try:
            a,b,qa=inspect_partitions(train,test,group_key=group_key)
            return a,b,{**audit,**qa,'split_seed':seed+offset}
        except ValueError as exc:
            last=exc
    raise ValueError(f'No two-class leakage-free split found across {attempts} deterministic seeds: {last}. Collect additional independent labelled samples.')


def evaluate(train, test, algorithm='baseline', epochs=8, seed=42, group_key='sample_id'):
    a,b,audit=inspect_partitions(train,test,group_key=group_key)
    if algorithm=='transformer':
        result=train_transformer(a,b,epochs=int(epochs),seed=int(seed))
    elif algorithm=='baseline':
        result=train_baseline(a,b)
    else:raise ValueError('algorithm must be baseline or transformer')
    probs=np.asarray(result['probability'])
    predictions=b[['sample_id','label','malware_family','events']].copy()
    predictions['predicted_label']=np.where(probs>=.5,'malicious','benign')
    predictions['malicious_probability']=probs
    return result,predictions,audit
