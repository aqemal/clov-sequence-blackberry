# Sequence-CloV 5.0 — Audited training integration

Version 5 adds **Leakage-aware training** tab to v4 while preserving previous functionality.

## Run on Windows
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Workflow
1. Export sanitized behavioural events from an authorized, isolated Android analysis environment.
2. Import reports in the **Sandbox report adapter** tab; maintain explicit, independently established benign/malicious labels and provenance.
3. Use the **Research validation** tab for hash consistency and chronological inspection. Remember: a claimed hash is not independent proof of sandbox execution.
4. In **Leakage-aware training**, upload a combined CSV (`sample_id,label,events,malware_family`), generate audited partitions, and train a TF-IDF baseline or small Transformer. Alternatively, import separately held-out train/test datasets.
5. Save frozen train/test CSVs, held-out predictions, and JSON evaluation metrics.

## IMPORTANT limitations
- **No APK execution**, remote sandbox automation, or malware downloads are included.
- Original **Train & evaluate** tab still has its original simple stratified split. Use the new tab for group-aware experiments.
- The original code is only a two-class classifier; malware-family names are preserved for stratification/provenance, but not predicted by this prototype.
- Datasets with only malware samples **cannot** train a meaningful benign/malicious binary classifier; add verified benign traces.
- The hash audit verifies declared metadata consistency, not authenticity of sandbox capture.
- Auditing reduces exact-sequence/sample overlap. It does not eliminate variant or distribution leakage; independent temporally separated or family-disjoint benchmarks are needed.
- Attention weights are **not validated causal explanations**.
- Reported results on synthetic sample data are not publishable Android malware performance claims.

## Run tests
```powershell
python -m unittest discover -p "test_*.py" -v
```
