"""Sequence-CloV interactive research prototype.
Run: streamlit run app.py
"""
import html
import json
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import generate_demo_data, validate_data, split_data, train_baseline, explain_baseline, parse_events
from transformer_model import train_transformer, predict_one
from ingestion import apk_inventory, inventory_summary, traces_to_csv_dataframe
from sandbox_adapter import adapt_many, training_export
from research_validation import check_provenance, ordered_events, audit_and_split
from verified_training import make_split, evaluate
from sequence_benchmarks import compare_models
from explainability import attribution, deletion_curve, event_summary
from dataset_release import ingest_report_archive, release_bundle
from experiment_registry import audit_experiment_bundle, export_audit
from dashboard_v10 import frequency_ranking, impact_ranking, wordcloud_html, ordered_timeline, training_matches
from quality_gate import assess_dataset, snapshot_zip
from experiment_v12 import open_release, run_experiment, export_experiment, class_coverage

st.set_page_config(page_title='Sequence-CloV | Malware Behaviour XAI', page_icon='🛡️', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.7rem;} 
.hero {background:linear-gradient(110deg,#092448,#135c78);border-radius:18px;padding:27px 32px;color:#fff;margin-bottom:1.2rem}
.hero h1 {color:#fff;font-size:2.25rem;margin:0;} .hero p {color:#e3f6ff;margin:.4rem 0 0;}
.smallnote {font-size:.84rem;color:#66788c;}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🛡️ Sequence-CloV</h1><p>Transformer-Based Android Malware Detection · Behavioural Visual Analytics · Research Demonstrator</p></div>', unsafe_allow_html=True)
st.caption('UMT · TAPERG · Ts. Dr. Che Akmal Che Yahaya | Proof-of-concept: pre-extracted logs only. No APK execution.')

with st.sidebar:
    st.header('⚙️ Simulation setup')
    source = st.radio('Input data', ['Synthetic demonstration', 'Upload my behavioural traces'], index=0)
    upload = st.file_uploader('CSV file', type=['csv'], disabled=source != 'Upload my behavioural traces')
    seed=st.number_input('Train/test random seed',min_value=1,max_value=99999,value=42)
    algo=st.radio('Classification model', ['TF-IDF + Logistic Regression (baseline)', 'Sequence-CloV Transformer'],index=1)
    epochs=st.slider('Transformer training epochs',1,30,8,disabled=not algo.startswith('Sequence'))
    st.divider()
    st.markdown('**Expected CSV fields**')
    st.code('sample_id,label,events,malware_family\nA01,malicious,"app_launch | read_sms | http_post_bulk",trojan\nA02,benign,"app_launch | render_ui",benign',language='csv')
    st.caption('Events are ordered and separated by a pipe ( | ). Labels: benign/malicious or 0/1. malware_family optional.')
    with open('example_dataset.csv','rb') as f:
        st.download_button('Download example CSV',f.read(),file_name='sequence_clov_example.csv',mime='text/csv')

try:
    if source=='Synthetic demonstration':
        raw=generate_demo_data(seed=int(seed))
        st.info('Demo mode: all event sequences and labels are **synthetically generated**, not observed Android malware. Scores must not be reported as real-world performance.')
    elif upload is not None:
        raw=pd.read_csv(upload)
    else:
        st.warning('Upload a CSV of pre-extracted Android behavioural traces to continue.')
        st.stop()
    data=validate_data(raw)
except Exception as exc:
    st.error(f'Input data error: {exc}')
    st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric('Applications / traces',f'{len(data):,}')
m2.metric('Benign samples',f'{(data.target==0).sum():,}')
m3.metric('Malicious samples',f'{(data.target==1).sum():,}')
m4.metric('Event types',f'{len(set(t for row in data.event_list for t in row)):,}')

# Version 10: main visual analytics panel is always visible, even before model training.
st.markdown('## ☁️ Sequence-CloV · Behavioural XAI Dashboard')
st.caption('A presentation-first view of behavioural sequences. Word size reflects event frequency until a trained model supplies event-deletion attribution; these are different measurements.')
v10_left,v10_right=st.columns([2,1])
with v10_left:
    v10_id=st.selectbox('Select Android behavioural trace',data.sample_id.astype(str).tolist(),key='v10_sample')
with v10_right:
    v10_mode=st.radio('Word-cloud measurement',['Event frequency','Model deletion impact'],horizontal=False,key='v10_mode')
v10_record=data.loc[data.sample_id.astype(str)==v10_id].iloc[0]
v10_events=list(v10_record.event_list)
v10_model_id='transformer' if algo.startswith('Sequence') else 'baseline'
v10_stamp=(source,getattr(upload,'name',None),int(seed),v10_model_id,int(epochs),len(data))
v10_fit=st.session_state.get('fit')
v10_trained=training_matches(v10_fit,v10_stamp)
v10_probability=None
v10_attention=None
v10_attribution=None
v10_att_error=None
if v10_trained:
    try:
        if v10_model_id=='transformer':
            v10_probability, v10_seq, v10_attention, v10_weights=predict_one(v10_fit['result'],v10_events)
        else:
            v10_probability=float(v10_fit['result']['model'].predict_proba(v10_fit['result']['vectorizer'].transform([' | '.join(v10_events)]))[0,1])
    except Exception as exc:
        st.warning(f'Model prediction unavailable: {exc}')
if v10_mode=='Model deletion impact' and v10_trained and len(v10_events)>1:
    try:
        v10_probability,v10_attribution=attribution(v10_fit['result'],v10_events,v10_model_id,max_events=min(48,len(v10_events)))
    except Exception as exc:
        v10_att_error=str(exc)

v10_k1,v10_k2,v10_k3,v10_k4=st.columns(4)
v10_k1.metric('Selected sample',v10_id)
v10_k2.metric('Dataset label',str(v10_record.label))
v10_k3.metric('Behavioural events',str(len(v10_events)))
v10_k4.metric('Malicious probability',f'{v10_probability:.1%}' if v10_probability is not None else 'Not trained')
if source=='Synthetic demonstration':
    st.info('SYNTHETIC DATA: This visualisation is illustrative only. Neither events nor predictions are measurements from the Google Drive APK collection.')
if not v10_trained:
    st.caption('For a trained model prediction or attribution, use **Train & evaluate** with the currently selected dataset and model, then return here.')
if v10_mode=='Model deletion impact':
    if v10_att_error:st.warning(f'Attribution could not be computed: {v10_att_error}')
    elif not v10_trained:st.warning('Model deletion impact requires a trained model. Showing event frequency instead; no model-based attribution is implied.')
    elif len(v10_events)<2:st.warning('At least two events are required. Showing event frequency instead.')

v10_use_imp=(v10_mode=='Model deletion impact' and v10_attribution is not None and not v10_attribution.empty)
v10_rank=impact_ranking(v10_attribution) if v10_use_imp else frequency_ranking(v10_events)
v10_measure='Mean absolute change in P(malicious) after removing one occurrence' if v10_use_imp else 'Number of occurrences in this trace'
v10_a,v10_b=st.columns([1.2,1])
with v10_a:
    st.markdown('### ☁️ Behavioural word cloud')
    st.caption(v10_measure + '. Not proof of causal behaviour.')
    st.markdown(wordcloud_html(v10_rank),unsafe_allow_html=True)
    st.download_button('Download word-cloud values (CSV)',pd.DataFrame(v10_rank,columns=['event','score']).to_csv(index=False).encode('utf-8'),file_name='sequence_clov_v10_wordcloud_values.csv',mime='text/csv',key='v10_wc_export')
with v10_b:
    st.markdown('### 📊 Behaviour importance')
    if v10_rank:
        v10_chart=pd.DataFrame(v10_rank[:14],columns=['event','score']).sort_values('score')
        st.plotly_chart(px.bar(v10_chart,x='score',y='event',orientation='h',labels={'score':v10_measure,'event':'Behaviour'}),use_container_width=True)
    else:st.info('No events to chart.')
st.markdown('### 🕒 Behavioural event order')
v10_timeline=ordered_timeline(v10_events)
if not v10_timeline.empty:
    v10_timeline_plot=px.scatter(v10_timeline,x='step',y='event',hover_data=['step','event'],labels={'step':'Sequence position','event':'Behaviour'})
    v10_timeline_plot.update_traces(marker_size=11)
    v10_timeline_plot.update_layout(height=max(290,min(650,26*v10_timeline.event.nunique()+150)))
    st.plotly_chart(v10_timeline_plot,use_container_width=True)
    st.download_button('Download ordered events (CSV)',v10_timeline.to_csv(index=False).encode('utf-8'),file_name='sequence_clov_v10_ordered_events.csv',mime='text/csv',key='v10_timeline_export')
if v10_attention is not None:
    with st.expander('Transformer self-attention heatmap (descriptive, not causal)',expanded=False):
        v10_labels=[f'{i+1}:{e}' for i,e in enumerate(v10_events[:len(v10_attention)])]
        st.plotly_chart(px.imshow(v10_attention,x=v10_labels,y=v10_labels,color_continuous_scale='Purples',labels={'color':'Attention','x':'Key event','y':'Query event'}),use_container_width=True)
if v10_attribution is not None and not v10_attribution.empty:
    with st.expander('Event deletion diagnostic details',expanded=False):
        st.dataframe(v10_attribution,use_container_width=True,hide_index=True)
        st.download_button('Download event-level attribution (CSV)',v10_attribution.to_csv(index=False).encode(),file_name='sequence_clov_v10_attribution.csv',mime='text/csv',key='v10_att_export')
st.divider()

tabs=st.tabs(['📊 Dataset explorer','🧠 Train & evaluate','🔎 Explain a sample','🧭 Research workflow','📁 APK inventory & trace ingestion','🧪 Sandbox report adapter','🔬 Research validation','✅ Leakage-aware training','📈 CNN / BiLSTM benchmark','🔬 XAI faithfulness lab','📦 Real-trace release builder','🧾 Sandbox experiment registry','🧰 Dataset quality gate (v11)','🧬 Reproducible experiments (v12)'])
with tabs[0]:
    a,b=st.columns([1,2])
    with a:
        fig=px.pie(names=['Benign','Malicious'],values=[(data.target==0).sum(),(data.target==1).sum()],hole=.55,
                   title='Label distribution',color_discrete_sequence=['#1c9db5','#7359c9'])
        st.plotly_chart(fig,use_container_width=True)
    with b:
        counts=pd.Series([t for row in data.event_list for t in row]).value_counts().head(18)
        fig=px.bar(x=counts.values,y=counts.index,orientation='h',title='Most frequent behavioural events',labels={'x':'Occurrences','y':'Event'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(data[['sample_id','label','malware_family','events']],use_container_width=True,hide_index=True)

with tabs[1]:
    st.markdown('### Controlled 75% / 25% train–test evaluation')
    st.caption('Stratified random split; token vocabulary and TF-IDF are fitted only on the training set. For published research, additionally use deduplication, family/time-disjoint splits, and independent test sets.')
    try:
        train,test=split_data(data,int(seed))
    except ValueError as exc:
        st.error(str(exc)); st.stop()
    model_id=('transformer' if algo.startswith('Sequence') else 'baseline')
    stamp=(source, getattr(upload,'name',None),int(seed),model_id,int(epochs),len(data))
    if st.button('▶ Train / retrain model',type='primary'):
        with st.spinner('Training on behavioural event traces...'):
            try:
                if model_id=='transformer':
                    result=train_transformer(train,test,epochs=int(epochs))
                else:
                    result=train_baseline(train,test)
                st.session_state['fit']={'stamp':stamp,'result':result,'test':test,'model_id':model_id}
            except Exception as exc:
                st.error(f'Training failed: {exc}')
    fitting=st.session_state.get('fit')
    if fitting and fitting['stamp']==stamp:
        metrics=fitting['result']['metrics']
        cols=st.columns(4)
        for col,name in zip(cols,['accuracy','precision','recall','f1']):
            col.metric(name.upper() if name=='f1' else name.title(),f'{metrics[name]:.3f}')
        left,right=st.columns([1,1])
        with left:
            cm=np.asarray(metrics['confusion_matrix'])
            fig=px.imshow(cm,x=['Pred. benign','Pred. malicious'],y=['Actual benign','Actual malicious'],
                          text_auto=True,color_continuous_scale='Blues',title='Held-out confusion matrix')
            st.plotly_chart(fig,use_container_width=True)
        with right:
            if model_id=='transformer':
                fig=px.line(x=list(range(1,len(fitting['result']['losses'])+1)),y=fitting['result']['losses'],markers=True,
                            labels={'x':'Epoch','y':'Training loss'},title='Transformer training loss')
                st.plotly_chart(fig,use_container_width=True)
            else:
                st.info('Baseline model: TF-IDF event frequencies + logistic regression. Use this as a reference benchmark when comparing with the Transformer.')
        predicted=fitting['result']['probability']
        pred_table=fitting['test'][['sample_id','label','malware_family','events']].copy()
        pred_table['predicted']=np.where(predicted>=.5,'malicious','benign')
        pred_table['malicious_probability']=np.round(predicted,4)
        st.download_button('Download held-out predictions CSV',pred_table.to_csv(index=False).encode(),
                           'sequence_clov_predictions.csv','text/csv')
        st.dataframe(pred_table[['sample_id','label','predicted','malicious_probability']],hide_index=True,use_container_width=True)
    else:
        st.info('Choose a model and click **Train / retrain model** to generate evaluation results.')

with tabs[2]:
    st.markdown('### Individual behavioural investigation')
    fitting=st.session_state.get('fit')
    if not fitting or fitting['stamp']!=(source,getattr(upload,'name',None),int(seed),('transformer' if algo.startswith('Sequence') else 'baseline'),int(epochs),len(data)):
        st.warning('Train the currently selected model in the **Train & evaluate** tab first.')
    else:
        sel=st.selectbox('Choose an application trace',data.sample_id.tolist())
        rec=data.loc[data.sample_id==sel].iloc[0]
        events=rec.event_list
        result=fitting['result']
        if fitting['model_id']=='transformer':
            probability,seq,matrix,importance=predict_one(result,events)
            ranked=sorted(zip(seq,importance.tolist()),key=lambda x:x[1],reverse=True)
            method='Average self-attention received (descriptive, NOT a causal explanation)'
        else:
            probability=float(result['model'].predict_proba(result['vectorizer'].transform([rec.events]))[0,1])
            ranked=explain_baseline(result,events)
            seq=events
            matrix=None
            method='Signed TF-IDF contributions to baseline model log-odds'
        c1,c2,c3=st.columns(3)
        c1.metric('Sample',str(sel))
        c2.metric('Ground-truth label',str(rec.label))
        c3.metric('Predicted malicious probability',f'{probability:.1%}')
        st.progress(float(np.clip(probability,0,1)))
        st.write(f'**Classification:** {"Malicious" if probability >= .5 else "Benign"} (threshold 0.50)')
        st.markdown('#### Behavioural event timeline')
        timeline=pd.DataFrame({'Step':range(1,len(seq)+1),'Event':seq})
        fig=px.scatter(timeline,x='Step',y='Event',hover_data=['Step','Event'],title='Event order (not wall-clock timestamps)')
        fig.update_traces(marker_size=12)
        fig.update_layout(height=max(320,min(750,28*len(set(seq))+160)))
        st.plotly_chart(fig,use_container_width=True)
        l,r=st.columns([1,1])
        with l:
            st.markdown('#### Event importance')
            st.caption(method)
            ranked=ranked[:15]
            if ranked:
                vals=pd.DataFrame(ranked,columns=['event','score'])
                fig=px.bar(vals.sort_values('score'),x='score',y='event',orientation='h',labels={'score':'Importance / contribution','event':'Event'})
                st.plotly_chart(fig,use_container_width=True)
        with r:
            st.markdown('#### Behaviour word cloud')
            st.caption('Illustrative word sizing from the selected model, not a frequency-only word cloud.')
            if ranked:
                max_abs=max(abs(value) for _,value in ranked) or 1
                spans=[]
                for idx,(token,val) in enumerate(ranked):
                    font=int(15+22*abs(val)/max_abs)
                    color=['#116a89','#6849a9','#0d8a92','#3563ae'][idx%4]
                    spans.append(f'<span style="font-size:{font}px;color:{color};padding:7px;display:inline-block">{html.escape(token)}</span>')
                st.markdown('<div style="min-height:240px;line-height:1.8">'+''.join(spans)+'</div>',unsafe_allow_html=True)
        if matrix is not None:
            st.markdown('#### Self-attention heatmap')
            st.caption('Averaged across attention heads; indicative of token-to-token attention, not proof that a feature caused the prediction.')
            fig=px.imshow(matrix,x=[f'{i+1}:{x}' for i,x in enumerate(seq)],
                          y=[f'{i+1}:{x}' for i,x in enumerate(seq)],color_continuous_scale='Purples',
                          labels={'x':'Key event','y':'Query event','color':'Attention'})
            fig.update_layout(height=600)
            st.plotly_chart(fig,use_container_width=True)
        st.markdown('#### Dynamic trace events')
        st.dataframe(pd.DataFrame({'step':range(1,len(events)+1),'event':events}),hide_index=True,use_container_width=True)

with tabs[3]:
    st.markdown('### Architecture and implementation status')
    st.markdown('''**1 · Dataset ingestion:** uploaded, pre-extracted trace sequences (CSV), plus purely synthetic demo mode.\n\n**2 · Sandboxing / trace extraction:** external integration point only; this app intentionally does *not* execute APKs or malware.\n\n**3 · Sequence encoding:** ordered event tokens, learned vocabulary and position embeddings.\n\n**4 · Modelling:** simple Transformer with multi-head self-attention; TF-IDF/logistic regression benchmark.\n\n**5 · Visual analytics:** per-sample probability, word cloud, attention heatmap, event timeline, evaluation metrics.\n\n**6 · Validation:** prototype evaluation; expert usability assessment and rigorous external benchmarking are future research stages.''')
    st.warning('Attention is not necessarily explanation. Use independent attribution methods (e.g. perturbation or Integrated Gradients) for validated explanations in a research publication.')
    st.markdown('**Safe integration recommendation:** execute any future malware collection in a properly isolated, authorized lab, export sanitized behavioural logs, then import those logs into this dashboard.')
    st.caption('Research prototype only. Do not use detection scores as a production security verdict.')

with tabs[4]:
    st.markdown('### Phase 1 · APK inventory (read-only)')
    st.info('This tool inventories local APK filenames and metadata. **It never executes APKs, unpacks them, or starts a sandbox.** Do not download or run untrusted APKs on your everyday Windows system. Use a properly isolated and authorized research environment.')
    st.caption('Download the approved dataset into an isolated analysis workstation first. Folder layout: dataset / FAMILY / varietyN / filename.apk. The Google Drive folder link itself cannot be supplied as a local path.')
    folder = st.text_input('Local dataset folder path (optional)',placeholder=r'D:\Research\AndroidDataset')
    hash_files=st.checkbox('Compute SHA-256 hash (reads APK bytes; may take time)',value=False)
    if st.button('Scan local APK inventory'):
        try:
            inventory=apk_inventory(folder,compute_sha256=hash_files)
            st.session_state['apk_inventory']=inventory
        except Exception as exc:
            st.error(f'Inventory could not be generated: {exc}')
    inventory=st.session_state.get('apk_inventory')
    if inventory is not None:
        summary=inventory_summary(inventory)
        a,b,c,d=st.columns(4)
        a.metric('APK files',summary['samples'])
        b.metric('Malware families',summary['families'])
        c.metric('Family/variety groups',summary['varieties'])
        d.metric('Total file size (MB)',summary['total_mb'])
        st.dataframe(inventory.head(1000),use_container_width=True,hide_index=True)
        st.download_button('Download inventory manifest CSV',inventory.to_csv(index=False),file_name='apk_manifest.csv',mime='text/csv')
        if len(inventory):
            counts=inventory.groupby('family').size().sort_values(ascending=False)
            st.bar_chart(counts)
        st.warning('An APK manifest contains malware family labels but **not** dynamic behavioural sequences. It must not be passed into the binary trace classifier as though it contained logs.')
    st.divider()
    st.markdown('### Phase 2 · Import extracted behavioural traces')
    st.caption('Import only sanitised output from an external isolated analysis pipeline. Supported: CSV with sample_id,label,events; JSON/JSONL with a list of event names or objects. Event list ordering is preserved.')
    report=st.file_uploader('Upload sanitised behavioural report',type=['csv','json','jsonl'],key='report_ingest')
    if report is not None:
        try:
            normalised=traces_to_csv_dataframe(report,report.name)
            st.success(f'Converted {len(normalised)} traces to the Sequence-CloV CSV schema.')
            st.dataframe(normalised.head(15),hide_index=True,use_container_width=True)
            st.download_button('Download model-ready trace CSV',normalised.to_csv(index=False).encode('utf-8'),file_name='sequence_clov_extracted_traces.csv',mime='text/csv')
            label_counts=normalised['label'].astype(str).str.lower().value_counts()
            if len(label_counts)<2:
                st.warning('This file contains only one class. Add verified benign traces before training a binary benign/malicious classifier; do not invent benign labels.')
            st.caption('To train, download the converted CSV and choose “Upload my behavioural traces” in the sidebar.')
        except Exception as exc:
            st.error(f'Invalid trace report: {exc}')
    st.markdown('**Research integrity:** Folder names are provisional family annotations. Verify samples, avoid duplicates and data leakage, and preserve original provenance. An attention heatmap is descriptive, not causal validation.')


with tabs[5]:
    st.markdown('### Phase 3 · Offline sandbox-report adapter')
    st.info('Upload **exported, sanitized JSON reports only**. This module does not execute APKs, start an emulator, or connect to sandbox services. Check authorization and isolation separately.')
    st.caption('Supported structures: events[] or behavior.processes[].calls[]. A report must contain a sample_id and an explicit verified label; you may supply both in an accompanying metadata CSV.')
    meta_file=st.file_uploader('Optional report metadata CSV (source_report,sample_id,label,malware_family)',type=['csv'],key='sandbox_meta')
    manifest_file=st.file_uploader('Optional existing APK manifest CSV (matching is identifier-only)',type=['csv'],key='sandbox_manifest')
    report_files=st.file_uploader('Sanitized sandbox JSON reports',type=['json'],accept_multiple_files=True,key='sandbox_jsons')
    try:
        meta=pd.read_csv(meta_file).fillna('') if meta_file else None
        manifest=pd.read_csv(manifest_file) if manifest_file else None
        if report_files:
            reports=[]
            for obj in report_files:
                override={}
                if meta is not None:
                    if 'source_report' not in meta.columns: raise ValueError('Metadata CSV requires source_report column')
                    selection=meta.loc[meta.source_report.astype(str)==obj.name]
                    if len(selection)>1: raise ValueError(f'Multiple metadata rows for {obj.name}')
                    if len(selection)==1:
                        m=selection.iloc[0]
                        override={'sample_id':str(m.get('sample_id','')) or None,
                                  'family':str(m.get('malware_family','')) or None,
                                  'label':str(m.get('label','')) or None}
                reports.append((obj.name,obj.getvalue(),override))
            converted,rejected=adapt_many(reports,manifest=manifest)
            c1,c2=st.columns(2)
            c1.metric('Converted',len(converted));c2.metric('Rejected',len(rejected))
            if len(converted):
                st.dataframe(converted,use_container_width=True,hide_index=True)
                st.download_button('Download model-ready CSV',training_export(converted).to_csv(index=False).encode('utf-8'),file_name='sandbox_model_ready.csv',mime='text/csv')
                st.download_button('Download provenance and warnings CSV',converted.to_csv(index=False).encode('utf-8'),file_name='sandbox_provenance.csv',mime='text/csv')
                if converted.label.nunique()<2:
                    st.warning('Single class detected. Binary classification training needs both verified benign and malicious traces.')
            if len(rejected): st.dataframe(rejected,use_container_width=True,hide_index=True)
            st.warning('Source array order is not necessarily real execution order. For publication, establish timestamp integrity, trace provenance, sandbox version, and family-label quality.')
    except Exception as exc:
        st.error(f'Adapter error: {exc}')


with tabs[6]:
    st.markdown('### Phase 4 · Research integrity and leakage audit')
    st.info('All checks are performed on imported metadata and sanitized traces. APKs are never executed or unpacked.')
    st.markdown('#### A. Report-to-APK hash corroboration')
    st.caption('Manifest CSV requires sha256, family; report CSV requires apk_sha256, malware_family. Matching self-reported hashes corroborates declarations but does **not** authenticate sandbox execution.')
    hash_manifest=st.file_uploader('APK inventory CSV (with SHA-256)',type='csv',key='v4_hash_manifest')
    hash_reports=st.file_uploader('Report metadata CSV (with apk_sha256)',type='csv',key='v4_hash_reports')
    if hash_manifest and hash_reports:
        try:
            checked=check_provenance(pd.read_csv(hash_manifest),pd.read_csv(hash_reports))
            st.dataframe(checked,hide_index=True,use_container_width=True)
            st.download_button('Download hash corroboration audit',checked.to_csv(index=False).encode(),file_name='hash_audit.csv',mime='text/csv')
        except Exception as exc:st.error(f'Hash audit: {exc}')
    st.divider()
    st.markdown('#### B. Timestamp-aware report preview')
    json_report=st.file_uploader('Exported sandbox JSON report',type='json',key='v4_time_json')
    if json_report:
        try:
            import json
            doc=json.loads(json_report.getvalue())
            seq,confidence=ordered_events(doc)
            st.write('Ordering status:',confidence)
            st.dataframe(pd.DataFrame({'index':range(1,len(seq)+1),'event':seq}),hide_index=True)
        except Exception as exc:st.error(f'Timeline: {exc}')
    st.divider()
    st.markdown('#### C. Group-aware holdout preparation')
    st.caption('Groups rows by sample ID and identical event sequences to avoid those overlaps between train and test. This is NOT family-disjoint validation unless you select malware_family and the dataset supports it.')
    split_file=st.file_uploader('Model-ready labelled behavioural CSV',type='csv',key='v4_split_csv')
    if split_file:
        try:
            split_df=pd.read_csv(split_file)
            choices=['sample_id']+(['malware_family'] if 'malware_family' in split_df.columns else [])
            split_group=st.selectbox('Grouping strategy',choices,key='v4_group')
            train_v4,test_v4,audit=audit_and_split(split_df,seed=int(seed),group_key=split_group)
            st.json(audit)
            if 'warning' in audit:st.warning(audit['warning'])
            st.download_button('Download train split',train_v4.to_csv(index=False).encode(),file_name='train_group_aware.csv',mime='text/csv')
            st.download_button('Download test split',test_v4.to_csv(index=False).encode(),file_name='test_group_aware.csv',mime='text/csv')
            st.warning('The original model-training tab still uses its own random split. For research reporting, train/evaluate from this split through a dedicated external training script; do not present random-split scores as leakage-controlled results.')
        except Exception as exc:st.error(f'Split audit: {exc}')


with tabs[7]:
    st.markdown('### Phase 5 · Leakage-aware training on exported behavioural traces')
    st.info('Only pre-extracted, labelled behavioural traces are accepted. No APKs are executed. This module uses a group-aware split and fits the model **only on training data**.')
    st.warning('Requires verified **benign and malicious** traces. The Google Drive folders inspected so far contain malware APKs, not observed behavioural reports or a known benign class. Do not use synthetic scores as research findings.')
    mode=st.radio('Training input',['Single dataset (create audited split)','Separate prepared training and testing CSVs'],horizontal=True,key='v5_mode')
    if mode.startswith('Single'):
        source_v5=st.file_uploader('Labelled behavioural dataset CSV',type='csv',key='v5_single')
        train_in=test_in=None
        if source_v5:
            try:
                all_data_v5=pd.read_csv(source_v5)
                grouping=st.selectbox('Holdout grouping', ['sample_id']+(['malware_family'] if 'malware_family' in all_data_v5.columns else []),key='v5_group')
                frac=st.slider('Test fraction',0.15,0.40,0.25,step=0.05,key='v5_frac')
                train_in,test_in,split_details=make_split(all_data_v5,seed=int(seed),test_fraction=float(frac),group_key=grouping)
                st.success('Audited split generated. No exact sequence or sample-ID overlap between partitions.')
                st.json(split_details)
            except Exception as exc: st.error(f'Split preparation failed: {exc}')
    else:
        train_file=st.file_uploader('Training split CSV',type='csv',key='v5_train')
        test_file=st.file_uploader('Independent testing split CSV',type='csv',key='v5_test')
        grouping='sample_id'
        train_in=test_in=None
        if train_file and test_file:
            try:
                from verified_training import inspect_partitions
                train_in,test_in,split_details=inspect_partitions(pd.read_csv(train_file),pd.read_csv(test_file))
                st.json(split_details)
            except Exception as exc: st.error(f'Partition validation failed: {exc}')
    if train_in is not None and test_in is not None:
        st.caption('Keep exported train/test splits frozen for comparison across models. Report provenance, collection environment, sample hashes, and split method in your publication.')
        c1,c2=st.columns(2)
        c1.download_button('Download audited training CSV',train_in.to_csv(index=False).encode(),'v5_train.csv','text/csv',key='v5_download_train')
        c2.download_button('Download audited testing CSV',test_in.to_csv(index=False).encode(),'v5_test.csv','text/csv',key='v5_download_test')
        algorithm_v5=st.selectbox('Model',['TF-IDF + Logistic Regression','Transformer'],key='v5_algo')
        epochs_v5=st.slider('Epochs (Transformer)',1,25,6,key='v5_epochs',disabled=algorithm_v5!='Transformer')
        if st.button('Train and evaluate on audited partitions',type='primary',key='v5_train_btn'):
            with st.spinner('Training research model on the prepared holdout...'):
                try:
                    result_v5,pred_v5,diagnostics_v5=evaluate(train_in,test_in,algorithm='transformer' if algorithm_v5=='Transformer' else 'baseline',epochs=epochs_v5,seed=int(seed),group_key=grouping)
                    st.session_state['v5_training']={'metrics':result_v5['metrics'],'pred':pred_v5,'diagnostics':diagnostics_v5,'algo':algorithm_v5}
                except Exception as exc:st.error(f'Model training failed: {exc}')
        if 'v5_training' in st.session_state:
            fitted_v5=st.session_state['v5_training']
            st.caption('Most recent successful audit-aware training run (may refer to a previously uploaded dataset; retrain after changing input).')
            c=st.columns(4)
            for i,name in enumerate(['accuracy','precision','recall','f1']):c[i].metric(name.title(),f"{fitted_v5['metrics'][name]:.3f}")
            st.write('Confusion matrix (rows: actual benign/malicious; columns: predicted benign/malicious):')
            st.dataframe(pd.DataFrame(fitted_v5['metrics']['confusion_matrix'],index=['benign','malicious'],columns=['benign','malicious']))
            st.download_button('Download held-out predictions',fitted_v5['pred'].to_csv(index=False).encode(),'v5_predictions.csv','text/csv',key='v5_predictions')
            st.download_button('Download evaluation JSON',json.dumps({'model':fitted_v5['algo'],'metrics':fitted_v5['metrics'],'diagnostics':fitted_v5['diagnostics']},indent=2).encode(),'v5_evaluation.json','application/json',key='v5_json')
            st.warning('Hash matches and sample labels must still be independently validated. This split reduces exact-duplicate leakage but does not rule out shared code, malware variants, or collection artefacts.')


with tabs[8]:
    st.markdown('### Phase 6 · Reproducible model comparison')
    st.info('Compare TF-IDF + Logistic Regression, 1D CNN, BiLSTM and Transformer on the **same** audited train/test partition. Import pre-extracted traces only; APKs are never executed.')
    st.warning('No actual sandbox traces or verified benign dataset have been supplied. Synthetic data results are illustrative, not publishable detection claims.')
    v6mode=st.radio('Dataset input',['Single CSV (audited group split)','Two pre-separated CSVs'],key='v6_mode',horizontal=True)
    v6train=v6test=None
    v6_group='sample_id'
    if v6mode.startswith('Single'):
        u=st.file_uploader('Labelled trace CSV',type=['csv'],key='v6_single')
        if u:
            try:
                df=pd.read_csv(u)
                opts=['sample_id']+(['malware_family'] if 'malware_family' in df.columns else [])
                v6_group=st.selectbox('Split group key',opts,key='v6_group')
                v6train,v6test,v6audit=make_split(df,seed=int(seed),group_key=v6_group)
                st.json(v6audit)
            except Exception as exc:st.error(f'Cannot prepare audited split: {exc}')
    else:
        f1=st.file_uploader('Training CSV',type=['csv'],key='v6_train')
        f2=st.file_uploader('Testing CSV',type=['csv'],key='v6_test')
        if f1 and f2:
            try:
                from verified_training import inspect_partitions
                v6train,v6test,v6audit=inspect_partitions(pd.read_csv(f1),pd.read_csv(f2))
                st.json(v6audit)
            except Exception as exc:st.error(f'Invalid partitions: {exc}')
    selected=st.multiselect('Models to evaluate',['baseline','cnn','bilstm','transformer'],default=['baseline','cnn','bilstm','transformer'],key='v6_selected')
    v6epochs=st.slider('Neural training epochs',1,30,6,key='v6_epochs')
    if v6train is not None and v6test is not None:
        st.caption(f'Frozen partition: {len(v6train)} training / {len(v6test)} testing. Vocabularies and model parameters are learned on training data only.')
        if st.button('Run fair benchmark',type='primary',key='v6_run'):
            if not selected:st.error('Select at least one model.')
            else:
                with st.spinner('Training selected models on the same audited partition...'):
                    try:
                        rows, preds, audit = compare_models(v6train,v6test,epochs=v6epochs,seed=int(seed),group_key=v6_group,models=selected)
                        st.session_state['v6_results']={'rows':rows,'preds':preds,'audit':audit,'seed':int(seed),'synthetic_warning':True}
                    except Exception as exc:st.error(f'Benchmark failed: {exc}')
    if 'v6_results' in st.session_state:
        st.warning('Results shown are from the most recent benchmark run; rerun after changing files, split, seed or model selection.')
        v6r=st.session_state['v6_results']
        metr=pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix'} for r in v6r['rows']])
        st.dataframe(metr,hide_index=True,use_container_width=True)
        if len(metr):
            st.bar_chart(metr.set_index('model')[['precision','recall','f1']])
        st.download_button('Download model comparison CSV',metr.to_csv(index=False).encode(),'sequence_clov_v6_model_comparison.csv','text/csv',key='v6_m')
        st.download_button('Download per-sample predictions',v6r['preds'].to_csv(index=False).encode(),'sequence_clov_v6_predictions.csv','text/csv',key='v6_p')
        st.download_button('Download reproducibility report',json.dumps({'results':v6r['rows'],'audit':v6r['audit'],'seed':v6r['seed'],'limitation':'Metrics do not establish sandbox provenance or cross-family generalization.'},indent=2).encode(),'sequence_clov_v6_report.json','application/json',key='v6_j')
        st.caption('Attention weights alone are not validated causal explanations; future XAI experiments should include perturbation and faithfulness tests.')


with tabs[9]:
    st.markdown('### Phase 7 · Explainability & faithfulness lab')
    st.info('Use an already-extracted, labelled behavioural dataset. This tab never executes APKs. Explanations are based on removing one input event at a time and measuring the change in the trained model prediction.')
    st.warning('Ablation effects are model-specific sensitivity measures, **not proof of causal malware behaviour**. The word cloud is not a validation of reasoning; deletion can create unrealistic sequences.')
    xmode=st.radio('Input split', ['One labelled CSV (audited split)','Separate train and test CSVs'],horizontal=True,key='v7_mode')
    xtrain=xtest=None
    if xmode.startswith('One'):
        xfile=st.file_uploader('Behavioural sequence CSV',type=['csv'],key='v7_full')
        if xfile:
            try:
                whole=pd.read_csv(xfile)
                xtrain,xtest,xaudit=make_split(whole,seed=int(seed))
                st.caption(f"Audited split: {len(xtrain)} train / {len(xtest)} test; seed {xaudit.get('split_seed')}")
            except Exception as exc:st.error(f'Input: {exc}')
    else:
        xtr=st.file_uploader('Train CSV',type=['csv'],key='v7_train')
        xte=st.file_uploader('Test CSV',type=['csv'],key='v7_test')
        if xtr and xte:
            try:
                from verified_training import inspect_partitions
                xtrain,xtest,xaudit=inspect_partitions(pd.read_csv(xtr),pd.read_csv(xte))
                st.caption(f"Validated separate splits: {len(xtrain)} train / {len(xtest)} test")
            except Exception as exc:st.error(f'Input: {exc}')
    choice=st.selectbox('Explanation model',['baseline','transformer'],key='v7_choice')
    xepochs=st.slider('Transformer epochs',1,20,4,key='v7_epochs',disabled=choice!='transformer')
    if xtrain is not None and xtest is not None:
        st.caption('Explanations are computed from held-out testing traces only, with training limited to the training partition.')
        if st.button('Train XAI model on audited split',key='v7_fit',type='primary'):
            try:
                with st.spinner('Training selected model...'):
                    from core import train_baseline
                    from transformer_model import train_transformer
                    fit=train_baseline(xtrain,xtest) if choice=='baseline' else train_transformer(xtrain,xtest,epochs=int(xepochs),seed=int(seed))
                st.session_state['v7_fit']={'result':fit,'test':xtest,'choice':choice,
                    'description':f'{choice}; {len(xtrain)} train / {len(xtest)} test; seed {int(seed)}'}
                st.success('Model fitted. Select a test sample below.')
            except Exception as exc:st.error(f'Training error: {exc}')
    if 'v7_fit' in st.session_state:
        xf=st.session_state['v7_fit']
        st.caption('Showing last fitted model: '+xf['description']+'. Refit after changing input files or parameters.')
        test_samples=xf['test']
        sample_choice=st.selectbox('Held-out trace',list(range(len(test_samples))),
            format_func=lambda i: str(test_samples.iloc[i]['sample_id']),key='v7_sample')
        record=test_samples.iloc[sample_choice]
        seq=record['event_list']
        max_events=st.slider('Maximum events for explanation',2,48,24,key='v7_limit')
        if st.button('Analyze selected trace',key='v7_analyze'):
            try:
                # Do not attribute beyond model window; same prefix for all comparisons
                effective=seq[:min(max_events,xf['result'].get('max_len',48))]
                p,imp=attribution(xf['result'],effective,xf['choice'],max_events=len(effective) if len(effective)>=2 else 2)
                st.session_state['v7_analysis']={'sample':str(record['sample_id']),
                    'events':effective,'p':p,'impact':imp,'model_choice':xf['choice'],
                    'result':xf['result'],'limit':max_events}
            except Exception as exc:st.error(f'Explanation error: {exc}')
        analyzed=st.session_state.get('v7_analysis')
        if analyzed:
            st.caption('Last analyzed trace: '+analyzed['sample']+' (recompute after changing the selected sample or model)')
            st.metric('P(malicious)',f"{analyzed['p']:.1%}")
            st.markdown('#### Position-specific deletion impact')
            st.caption('Positive value: deleting the event lowered malicious probability. Negative value: deleting it raised malicious probability.')
            st.dataframe(analyzed['impact'],hide_index=True,use_container_width=True)
            if len(analyzed['impact']):
                chart=px.bar(analyzed['impact'],x='position',y='delta_malicious_probability',hover_data=['event'],
                    title='Leave-one-event-out perturbation',labels={'position':'Event position','delta_malicious_probability':'Change in P(malicious)'})
                st.plotly_chart(chart,use_container_width=True)
                st.markdown('#### Behavioural word cloud (aggregated absolute deletion impact)')
                words=event_summary(analyzed['impact']).head(24)
                max_word=float(words['mean_abs_impact'].max()) or 1.0
                cloud=[]
                for _,item in words.iterrows():
                    fontsize=14+26*float(item['mean_abs_impact'])/max_word
                    cloud.append(f'<span style="font-size:{fontsize:.0f}px;padding:8px;display:inline-block;color:#1683a0">{html.escape(str(item["event"]))}</span>')
                st.markdown(''.join(cloud),unsafe_allow_html=True)
                try:
                    curve,predclass=deletion_curve(analyzed['result'],analyzed['events'],analyzed['model_choice'],
                        analyzed['impact']['delta_malicious_probability'].to_numpy(),seed=int(seed))
                    st.markdown('#### Deletion faithfulness diagnostic')
                    st.caption(f"Original predicted class: {'malicious' if predclass else 'benign'}. Compare predicted-class confidence after deleting top-ranked versus randomly selected events. A strong explanation should usually lower confidence faster than random deletion, but this is only a diagnostic.")
                    st.plotly_chart(px.line(curve,x='removed',y='class_confidence',color='strategy',markers=True),use_container_width=True)
                    st.download_button('Download attribution CSV',analyzed['impact'].to_csv(index=False).encode(),
                        file_name='sequence_clov_v7_attribution.csv',mime='text/csv',key='v7_dl_att')
                    st.download_button('Download deletion curve CSV',curve.to_csv(index=False).encode(),
                        file_name='sequence_clov_v7_faithfulness.csv',mime='text/csv',key='v7_dl_curve')
                except Exception as exc:st.error(f'Deletion diagnostics: {exc}')
            else:
                st.info('At least two events are needed to calculate deletion attribution.')


with tabs[10]:
    st.markdown('### Phase 8 · Reproducible real-trace dataset release')
    st.info('Offline only: upload *exported sandbox JSON/JSONL reports*, never raw APKs. This tool does not run, unpack or analyze Android applications.')
    st.warning('Labels and hashes come from your separate manifest. A matching report-declared hash is NOT independent evidence of real sandbox execution. This tab produces no measured malware-detection accuracy.')
    st.markdown('**Input A — CSV label manifest**: `sample_id,label,malware_family,apk_sha256` (optional `label_evidence` column).')
    manifest_file=st.file_uploader('Upload label manifest',type=['csv'],key='v8_manifest')
    archive_file=st.file_uploader('Upload ZIP of exported JSON / JSONL sandbox reports',type=['zip'],key='v8_reports')
    require_hash=st.checkbox('Require matching declared SHA-256 in every report',value=True,key='v8_hash')
    if st.button('Validate and convert reports',key='v8_convert'):
        if not manifest_file or not archive_file:st.error('Provide both a label manifest and a ZIP of reports.')
        else:
            try:
                accepted,rejected=ingest_report_archive(archive_file.getvalue(),pd.read_csv(manifest_file),require_hash=require_hash)
                st.session_state['v8_processed']={'accepted':accepted,'rejected':rejected}
            except Exception as exc:st.error(f'Ingestion rejected: {exc}')
    result_v8=st.session_state.get('v8_processed')
    if result_v8 is not None:
        good,bad=result_v8['accepted'],result_v8['rejected']
        a,b,c=st.columns(3)
        a.metric('Accepted traces',len(good));b.metric('Rejected reports',len(bad))
        c.metric('Classes',good.label.nunique() if not good.empty else 0)
        if not good.empty:
            st.dataframe(good,use_container_width=True,hide_index=True)
            st.download_button('Download imported trace CSV',good.to_csv(index=False).encode('utf-8'),file_name='v8_accepted_traces.csv',mime='text/csv',key='v8_good')
        if not bad.empty:
            st.dataframe(bad,use_container_width=True,hide_index=True)
            st.download_button('Download rejected report log',bad.to_csv(index=False).encode('utf-8'),file_name='v8_rejected_reports.csv',mime='text/csv',key='v8_bad')
        if not good.empty:
            st.markdown('#### Freeze reproducible train/test partitions')
            st.caption('Requires enough independent benign and malicious samples in BOTH partitions; refuses shared identical sequences, IDs, or APK hashes.')
            fraction=st.slider('Test fraction',.15,.4,.25,.05,key='v8_fraction')
            grouping=st.selectbox('Grouping',options=['sample_id','malware_family'],key='v8_group')
            if st.button('Build release ZIP',key='v8_release'):
                try:
                    blob,meta=release_bundle(good,seed=int(seed),test_fraction=fraction,group_key=grouping)
                    st.session_state['v8_release']={'blob':blob,'meta':meta}
                except Exception as exc:st.error(f'Release blocked: {exc}')
            saved=st.session_state.get('v8_release')
            if saved:
                st.json(saved['meta'])
                st.download_button('Download frozen research dataset release',saved['blob'],file_name='Sequence-CloV_v8_dataset_release.zip',mime='application/zip',key='v8_release_download')
            st.caption('Train models on the frozen train.csv and test.csv in the existing CNN/BiLSTM benchmark tab. Exported reports must come from an approved, isolated sandbox.')


with tabs[11]:
    st.markdown('### Phase 9 · Sandbox experiment registry (offline)')
    st.info('Import externally exported sandbox reports and run records. This app never executes or opens APK binaries. Use a dedicated authorized isolated environment for any malware execution.')
    st.markdown('**Required ZIP contents:** `run_manifest.csv` in the archive root, plus JSON files referenced by its `report_file` column.')
    st.code('run_id,sample_id,apk_sha256,environment_id,environment_version,android_api_level,started_at_utc,finished_at_utc,status,report_file,label_evidence',language='text')
    f9=st.file_uploader('Upload exported run archive (ZIP)',type=['zip'],key='v9_runs')
    if st.button('Audit sandbox run records',key='v9_audit'):
        if f9 is None:st.error('Select a ZIP containing run_manifest.csv and exported JSON reports.')
        else:
            try:
                ok9,bad9,summary9=audit_experiment_bundle(f9.getvalue())
                st.session_state['v9_audit_result']={'ok':ok9,'bad':bad9,'summary':summary9,'file_id':(f9.name,len(f9.getvalue()))}
            except Exception as exc:st.error(f'Archive rejected: {exc}')
    audit9=st.session_state.get('v9_audit_result')
    if audit9:
        if f9 is None or (f9.name,len(f9.getvalue()))!=audit9['file_id']:
            st.warning('Upload has changed; re-run audit before exporting.')
        else:
            p,q=st.columns(2);p.metric('Accepted completed runs',len(audit9['ok']));q.metric('Rejected runs',len(audit9['bad']))
            st.json(audit9['summary'])
            st.caption('Accepted = internally consistent run metadata, NOT verified execution. Only completed runs with matching IDs/hashes and documented label evidence pass.')
            if not audit9['ok'].empty:st.dataframe(audit9['ok'],use_container_width=True,hide_index=True)
            if not audit9['bad'].empty:st.dataframe(audit9['bad'],use_container_width=True,hide_index=True)
            st.download_button('Download experiment audit ZIP',export_audit(audit9['ok'],audit9['bad'],audit9['summary']),file_name='sequence_clov_v9_experiment_audit.zip',mime='application/zip',key='v9_dl')
    st.caption('After checking the experiment registry, import original reports + independent label manifest in the Real-trace release builder tab. The audit alone does not produce training data.')


with tabs[12]:
    st.markdown('### Dataset quality gate · Version 11')
    st.info('Preflight research dataset validation before training. This tool only reads uploaded extracted-trace CSVs. It does not execute APKs or authenticate sandbox provenance.')
    input_v11=st.file_uploader('Upload extracted behavioural trace CSV',type=['csv'],key='v11_upload')
    min_v11=st.number_input('Minimum event count per sample',min_value=1,max_value=100,value=2,key='v11_min')
    hash_v11=st.checkbox('Require 64-character APK SHA-256 values',value=False,key='v11_hash')
    if input_v11 is not None:
        try:
            v11data=pd.read_csv(input_v11)
            accepted_v11,audit_v11,summary_v11=assess_dataset(v11data,min_events=int(min_v11),require_hash=hash_v11)
            aa,bb,cc=st.columns(3)
            aa.metric('Input records',summary_v11['total'])
            bb.metric('Accepted',summary_v11['accepted'])
            cc.metric('Rejected',summary_v11['rejected'])
            st.subheader('Quality audit')
            st.dataframe(audit_v11,use_container_width=True,hide_index=True)
            st.json({k:summary_v11[k] for k in ('class_counts_accepted','error_frequency','warning_frequency')})
            st.warning(summary_v11['provenance_warning'])
            st.download_button('Download quality audit CSV',audit_v11.to_csv(index=False).encode(),file_name='quality_audit_v11.csv',mime='text/csv',key='v11_audit')
            if not accepted_v11.empty:
                blob_v11,manifest_v11=snapshot_zip(accepted_v11,audit_v11,summary_v11,source_label=input_v11.name,seed=int(seed))
                st.download_button('Download reproducible dataset snapshot ZIP',blob_v11,file_name='sequence_clov_v11_dataset_snapshot.zip',mime='application/zip',key='v11_snapshot')
                st.caption('Export a frozen validated snapshot, then use Leakage-aware training or CNN / BiLSTM benchmark for group-aware training and evaluation.')
            else:st.warning('No accepted traces. Review errors and correct the input before training.')
        except Exception as exc:st.error(f'Dataset quality gate failed: {exc}')


with tabs[13]:
    st.markdown('### Reproducible held-out experiment · Version 12')
    st.info('This module evaluates pre-extracted behavioural traces from a frozen V8 dataset release. It does not execute APKs or independently authenticate sandbox reports. Synthetic examples are NOT research findings.')
    release12=st.file_uploader('Upload V8 release ZIP (train.csv + test.csv + release_manifest.json)', type=['zip'],key='v12_release')
    if release12 is not None:
        try:
            payload12=release12.getvalue()
            train12,test12,meta12,audit12=open_release(payload12)
            coverage12=class_coverage(train12,test12)
            a12,b12=st.columns(2)
            a12.metric('Train traces',coverage12['train_samples'])
            b12.metric('Test traces',coverage12['test_samples'])
            st.write('Class representation:',coverage12)
            st.caption('Source ZIP SHA-256: '+audit12['source_archive_sha256'])
            st.json({'source_manifest':meta12,'partition_audit':audit12})
            selection12=st.multiselect('Benchmarks',options=['baseline','cnn','bilstm','transformer'],default=['baseline','cnn','bilstm','transformer'],key='v12_models')
            epochs12=st.slider('Neural training epochs',min_value=1,max_value=30,value=4,key='v12_epochs')
            seed12=st.number_input('Experiment seed',min_value=1,max_value=999999,value=42,key='v12_seed')
            st.warning('Train/test leakage checks are necessary but not sufficient: family overlap, collection artifacts, and uncertain ground truth can still bias results. Avoid tuning against the held-out test set.')
            if st.button('Run frozen-partition benchmark',key='v12_run'):
                if not selection12:st.error('Select at least one benchmark model.')
                else:
                    with st.spinner('Training selected models on pre-extracted traces'):
                        result12=run_experiment(payload12,models=selection12,epochs=int(epochs12),seed=int(seed12))
                    st.session_state['v12_result']={'result':result12,'source_hash':audit12['source_archive_sha256'],'selection':list(selection12),'seed':int(seed12),'epochs':int(epochs12)}
            saved12=st.session_state.get('v12_result')
            if saved12 and (saved12['source_hash'],saved12['selection'],saved12['seed'],saved12['epochs'])==(audit12['source_archive_sha256'],list(selection12),int(seed12),int(epochs12)):
                st.subheader('Held-out evaluation')
                st.dataframe(pd.DataFrame([{k:v for k,v in m.items() if k!='confusion_matrix'} for m in saved12['result']['metrics']]),use_container_width=True,hide_index=True)
                st.download_button('Download experiment results ZIP',export_experiment(saved12['result']),file_name='sequence_clov_v12_experiment.zip',mime='application/zip',key='v12_export')
            elif saved12:st.warning('Inputs changed since the last run; rerun before exporting results.')
        except Exception as exc:
            st.error(f'Release validation failed: {exc}')
    else:
        st.caption('First create a release ZIP using Real-trace release builder (v8), including both benign and malicious traces, then upload it here.')
