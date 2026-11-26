# Sequence-CloV v11 — Dataset Quality Gate

This update adds **Dataset quality gate (v11)** as a new Streamlit tab, preserving the earlier features. The additional module is `quality_gate.py`.

## Run on Windows PowerShell

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a CSV with `sample_id,label,events` (optional `malware_family,apk_sha256`). The new tab computes event counts and fingerprints, detects duplicate sample IDs, repeated APK hashes, identical ordered sequences, conflicting labels and invalid hashes. It exports a quality audit and a snapshot ZIP with SHA-256 checksums of accepted traces and audit records. Sample dataset: `example_dataset.csv` (entirely synthetic).

## Limitations

- No APK execution, dynamic trace collection, or malware analysis environment is included.
- A passing quality gate is **not** proof of correct class labels or authentic sandbox provenance.
- Rows sharing the same hash or exact sequence are conservatively rejected; review before deduplication and before any split.
- Data snapshot export does not train a model. Train on a *separate audited split* in the existing benchmarking tabs.
- Existing tabs may still use synthetic data. Never present their scores as validation on real malware.
- The new tab does not contact Google Drive automatically.

Run tests: `python -m unittest discover -p 'test_*.py'`.
