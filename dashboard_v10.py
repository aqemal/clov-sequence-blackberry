"""Pure presentation helpers for the V10 main visual analytics dashboard.

Never opens or executes APKs; only consumes pre-extracted event strings.
"""
from collections import Counter
import html
import math
import pandas as pd


def frequency_ranking(events, limit=24):
    if not events:
        return []
    return [(name, float(count)) for name, count in Counter(events).most_common(limit)]


def impact_ranking(impact, limit=24):
    """Collapse event-level deletion impacts to unique behavioural terms."""
    if impact is None or impact.empty:
        return []
    needed={'event','absolute_impact'}
    if not needed.issubset(impact.columns):
        raise ValueError('Attribution table must contain event and absolute_impact')
    df=impact.copy()
    df['absolute_impact']=pd.to_numeric(df['absolute_impact'],errors='coerce')
    df=df[df['absolute_impact'].map(math.isfinite)]
    if df.empty:return []
    agg=df.groupby('event',as_index=False)['absolute_impact'].mean()
    agg=agg.sort_values(['absolute_impact','event'],ascending=[False,True]).head(limit)
    return [(str(r.event),float(r.absolute_impact)) for r in agg.itertuples(index=False)]


def wordcloud_html(ranking):
    """Escape event names to avoid injecting user-uploaded labels into Streamlit HTML."""
    if not ranking:
        return '<p>No behavioural events are available.</p>'
    values=[abs(float(score)) for _,score in ranking if math.isfinite(float(score))]
    high=max(values,default=0.0) or 1.0
    tones=['#0F6B8D','#7657B7','#087D75','#B05E86','#2759AA']
    words=[]
    for idx,(event,value) in enumerate(ranking):
        value=float(value)
        if not math.isfinite(value):continue
        size=15+35*abs(value)/high
        safe=html.escape(str(event),quote=True)
        words.append(f'<span title="{safe}" style="font-size:{size:.1f}px;color:{tones[idx%len(tones)]};padding:7px;display:inline-block;font-weight:600;overflow-wrap:anywhere">{safe}</span>')
    return '<div style="min-height:200px;line-height:1.55;overflow-wrap:anywhere">'+''.join(words)+'</div>'


def ordered_timeline(events):
    return pd.DataFrame({'step':list(range(1,len(events)+1)),'event':list(events)})


def training_matches(fit, stamp):
    return bool(fit and fit.get('stamp')==stamp and 'result' in fit and 'model_id' in fit)
