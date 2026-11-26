# Sequence-CloV v7: Explainability & faithfulness lab

Run `python -m pip install -r requirements.txt` then `python -m streamlit run app.py`.

The new **XAI faithfulness lab** tab accepts pre-extracted behavioural sequences formatted as `sample_id,label,events,malware_family`. Use a group-aware split or two independently prepared train/test CSVs. Train a baseline or Transformer, choose a held-out trace, and compute event-deletion attribution, an aggregated impact word cloud, and a top-attributed vs random deletion curve.

Run unit tests: `python -m unittest discover -p 'test_*.py'`.

**Interpretation restrictions:** Attribution is a model sensitivity diagnostic, not an explanation of real-world malicious intent. Removing events may produce unrealistic or out-of-distribution sequences. Attention weights remain descriptive, not causal. This app does not download, unpack, or execute APK files. No real sandbox-derived or benign benchmark dataset was provided; do not report synthetic results as published performance. The tab stores the most recent trained model and last-analyzed sample in session state: retrain/recalculate when inputs change. Event display is limited to the model's 48-event context window.
