"""Model-agnostic, offline event-deletion attribution and faithfulness diagnostics.

Works on *already extracted* logs. Does not ingest or execute APK files.
Deletion impact is an intervention on the input sequence, not an estimate of
real-world malware causality. Results may be affected by out-of-distribution traces.
"""
from collections import Counter
import numpy as np
import pandas as pd


def probability(result, events, model_type):
    """Return P(malicious) from a previously fitted baseline or Transformer."""
    seq=list(events)
    if not seq:
        raise ValueError('At least one event is required for inference')
    if model_type=='baseline':
        return float(result['model'].predict_proba(result['vectorizer'].transform([' | '.join(seq)]))[0,1])
    if model_type=='transformer':
        import torch
        mdl=result['model']; voc=result['vocab']; max_len=result['max_len']
        mdl.eval()
        x=torch.tensor([voc.encode(seq,max_len)],dtype=torch.long)
        with torch.no_grad():
            logit,_=mdl(x)
            return float(torch.sigmoid(logit)[0].item())
    raise ValueError('Model type must be baseline or transformer')


def attribution(result, events, model_type, max_events=48):
    """Delete one event at a time; positive delta supports malicious prediction.

    Position-specific, not a unique-token importance score. A one-event trace
    cannot be deleted, so returns an empty impact table.
    """
    if not 2 <= int(max_events) <= 512:
        raise ValueError('max_events must be between 2 and 512')
    events=list(events)[:max_events]
    if not events: raise ValueError('Trace contains no events')
    original=probability(result,events,model_type)
    rows=[]
    for i,event in enumerate(events):
        if len(events)==1: break
        altered=events[:i]+events[i+1:]
        new_prob=probability(result,altered,model_type)
        rows.append({'position':i+1,'event':event,'probability_without':new_prob,
                     'delta_malicious_probability':original-new_prob,
                     'absolute_impact':abs(original-new_prob)})
    frame=pd.DataFrame(rows,columns=['position','event','probability_without',
        'delta_malicious_probability','absolute_impact'])
    return original,frame


def deletion_curve(result,events,model_type,importance,seed=42,steps=6):
    """Confidence change when deleting ranked vs random events, preserving order.

    Uses class-confidence score: P(malicious) for positive prediction, else
    1-P(malicious). No retraining; estimates are model-specific, not causal.
    """
    seq=list(events)
    if len(seq)<2: raise ValueError('At least two events required')
    if len(importance)!=len(seq): raise ValueError('Attribution length mismatch')
    p0=probability(result,seq,model_type)
    cls=1 if p0>=.5 else 0
    confidence=lambda p: p if cls==1 else 1-p
    impacts=np.asarray(importance,dtype=float)
    impacts=np.where(np.isfinite(impacts),impacts,0.0)
    # rank by reduction of *predicted class* confidence, not signed maliciousness
    ordered=np.argsort(-(impacts if cls==1 else -impacts),kind='stable')
    random_order=np.random.default_rng(seed).permutation(len(seq))
    amounts=sorted(set([0]+[max(1,int(round(x))) for x in np.linspace(1,len(seq)-1,min(steps,len(seq)-1))]))
    rows=[]
    for k in amounts:
        for mode,ordering in [('top_attributed',ordered),('random',random_order)]:
            gone=set(ordering[:k])
            remainder=[v for i,v in enumerate(seq) if i not in gone]
            p=probability(result,remainder,model_type)
            rows.append({'removed':k,'strategy':mode,'class_confidence':confidence(p),
                         'confidence_drop':confidence(p0)-confidence(p), 'remaining_events':len(remainder)})
    return pd.DataFrame(rows),cls


def event_summary(frame):
    if frame.empty:return pd.DataFrame(columns=['event','mean_abs_impact','count'])
    return (frame.groupby('event',as_index=False)
            .agg(mean_abs_impact=('absolute_impact','mean'),count=('event','size'))
            .sort_values('mean_abs_impact',ascending=False))
