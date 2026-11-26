# Sequence-CloV v6 — CNN, BiLSTM and Transformer benchmarking

## Launch (Windows PowerShell)
```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the new **CNN / BiLSTM benchmark** tab to load CSV behavioural traces and run four models on one leakage-aware partition.

CSV columns: `sample_id,label,events` (optional `malware_family`); events separated by ` | `.

This application **does not run or decompile APK files**. Google Drive malware folders contain raw APK files; separately exported sandbox traces and verified benign traces are required before real-world binary detection. All code runs locally/offline.

**Methodological limitations:** Random/group holdouts do not guarantee family-, developer- or temporal-disjoint evaluation. Verify provenance, split by appropriate real-world collection identifiers, use fixed holdouts, repeated seeds, confidence intervals, and an untouched external test set before publication. Attention weights are descriptive and not inherently faithful explanations. Training metrics on synthetic data are demonstration-only.

## Automated tests
```
python -m unittest discover -p "test_*.py"
```
