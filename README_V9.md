# Sequence-CloV v9: Offline sandbox experiment registry

The new **Sandbox experiment registry** tab accepts a ZIP containing `run_manifest.csv` at the root and exported JSON reports. It checks UTC run timestamps, sandbox environment information, matching sample identifiers and declared APK hashes, completed status, and label evidence. It does **not** execute APKs, download malware, attest a sandbox's authenticity, or independently verify labels.

## Getting started

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload `example_v9_experiment_bundle_SYNTHETIC.zip` in the **Sandbox experiment registry** tab. A successful audit exports a CSV/JSON ZIP for documentation. Use the existing real-trace release builder for the separate labelled dataset release.

## Run manifest fields

Required: `run_id,sample_id,apk_sha256,environment_id,environment_version,android_api_level,started_at_utc,finished_at_utc,status,report_file`.

Also provide `label_evidence` to make a completed run audit-eligible. Status must be `completed`. UTC timestamps must have `+00:00` or `Z` offset.

**Limitations:** No live sandbox connection or real malware trace validation. Hash and run-ID matches are metadata consistency checks only.
