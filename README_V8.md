# Sequence-CloV v8 — Offline real-trace dataset release

Run `python -m pip install -r requirements.txt` and `python -m streamlit run app.py`.

## New tab
`Real-trace release builder` takes a CSV label manifest and a ZIP of *already exported* JSON / JSONL sandbox reports. It never executes APKs and refuses non-report file types.

Required manifest columns: `sample_id,label,malware_family,apk_sha256`. `label_evidence` is optional. Every report should contain `sample_id`, `apk_sha256` (or `sha256`), and `events` or `behavior.processes[].calls`. For research, use labels independently reviewed and documented by the researchers.

**Security:** use isolated systems for APK execution, not your working PC. ZIP inputs are size limited; the app does not extract reports onto disk. Keep untrusted exports sanitized and avoid uploading identifiable payload details.

**Limitations:** hashes in reports can be falsified, and labels are user assertions. This module cannot verify sandbox authenticity, true collection time, or malware ground truth. Array order is not wall-clock chronology. A benign dataset and genuine traces are still necessary before real-data evaluation.

**Output:** accepted and rejected logs; if both classes have enough independent samples, a reproducible ZIP containing `train.csv`, `test.csv`, `release_manifest.json`, `accepted_provenance.csv`, and data dictionary. Train/test sets can be loaded into the existing benchmark tab.

**Test:** `python -m unittest discover -p 'test_*.py'`
