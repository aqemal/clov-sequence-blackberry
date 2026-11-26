# Sequence-CloV: Android behavioural trace research demonstrator

**Research basis:** TAPERG project *Sequence-CloV: Explainable Transformer-Based Android Malware Detection with Behavioural Visualization Techniques* (UMT).

This package is a **working proof-of-concept**, not a validated production malware detection system. It consumes **pre-extracted ordered behavioural logs**, not APK files. It does **not** execute malware or create a sandbox; the sandbox stage is a future integration point.

## Quick start

1. Install Python 3.10+ (Python 3.11 recommended).
2. Extract this ZIP, open a terminal in the `Sequence-CloV_Prototype` folder.
3. Create an isolated Python environment (recommended):

   Windows PowerShell:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the URL that Streamlit displays, generally `http://localhost:8501`.
5. In demo mode, select a model, then select **Train / retrain model** under the training tab. Explore the interpretation tab afterward.

## Dataset CSV format

Columns: `sample_id,label,events` (required), `malware_family` (optional).

```csv
sample_id,label,events,malware_family
001,benign,"app_launch | render_ui | file_read | app_background",benign
002,malicious,"app_launch | read_sms | encode_payload | outbound_transfer",example_family
```

* Each row is one Android application execution trace.
* Preserve event order using ` | ` between events; a JSON array of strings is also accepted.
* `label` can be `benign` / `malicious` or `0` / `1`.
* CSV traces should have event names (e.g. `api:sendTextMessage`, `network:http_post`, `permission:READ_SMS`) rather than raw unstructured log lines. Convert upstream where necessary.
* At least **12 traces** and **3 per class** are required for the proof-of-concept training split. Research-quality evaluation requires substantially larger, properly sampled datasets.
* `example_dataset.csv` contains **synthetic sequences**, not collected malware samples.

## Feature mapping to the proposed research framework

| Grant proposal stage | Current implementation |
| --- | --- |
| APK samples / datasets | CSV intake + example synthetic traces |
| Sandbox execution | Not implemented; safe external integration point |
| Dynamic behavioural logs | Ordered event sequences with basic validation |
| Sequence encoding | Vocabulary, integer tokens, learned positional embeddings |
| Transformer-based model | Small PyTorch self-attention classifier |
| Conventional benchmarking | TF-IDF + logistic regression (not CNN / BiLSTM yet) |
| Attention analysis | Mean multi-head self-attention visualization |
| Word cloud / heatmap / timeline | Interactive visual summaries and event-order graph |
| Cloud deployment | Local Streamlit application; cloud deployment not configured |
| Evaluation | 75/25 stratified holdout, accuracy, precision, recall, F1, confusion matrix |

## Research cautions and limitations

- **Attention weights do not establish a causal explanation.** The transformer attention heatmap is a diagnostic visualization. The baseline model has interpretable linear contributions, which are a different technique. Add perturbation, Integrated Gradients, and analyst studies for an XAI validity claim.
- The demo dataset was programmatically generated with artificial signatures. High synthetic accuracy **must not** be used as scientific evidence of malware detection performance.
- A random trace-level train/test split is not a sufficient measure of robustness, especially where near-duplicate applications, related variants, families, or shared time periods occur. For research, deduplicate, group by app/family/source where appropriate, and use independent holdout periods.
- This demonstrator does not include trusted APK sandboxing, malware family classification, dynamic instrumentation, timestamps, cloud orchestration, authentication, or deployment hardening.
- Keep malicious APKs outside the web app, on appropriately isolated and approved research infrastructure. Import only sanitized extracted logs.

## Next steps for real dataset integration

1. Identify dataset format (CSV, JSON, sandbox report) and available features (API calls, permissions, network, file operations, event timestamps).
2. Implement a dataset adapter that preserves order, metadata, and sample provenance.
3. Replace arbitrary event token assumptions with appropriate normalized tokenization.
4. Add CNN and BiLSTM comparison models, robust evaluation and calibration.
5. Validate XAI against independent feature attribution methods and domain expert feedback.
6. Containerize and deploy the dashboard to an approved cloud or institutional environment.

## Contents

- `app.py` — Streamlit interactive UI.
- `core.py` — CSV validation, synthetic data generation and TF-IDF benchmark.
- `transformer_model.py` — ordered-event Transformer model and attention extraction.
- `example_dataset.csv` — synthetic sample dataset.
- `requirements.txt` — dependencies.

## Version 2.0 add-on — APK inventory and extracted-trace importer

The **APK inventory & trace ingestion** tab adds a read-only metadata scanner and JSON/JSONL/CSV trace converter. It retains the original working app and model. This is **not** a sandbox, and no APK is run or unpacked.

**Your Google Drive folder:** The shared Android malware collection is organized by malware family (`BankBot`, `Triada`, etc.), then `variety` subfolders, then `.apk` samples. For example, `BankBot/variety1/*.apk`. The exact file counts across all 71 families have not been independently counted. The app cannot read the Google Drive URL as a Windows directory path: use a local authorized dataset mirror or a manifest prepared outside the normal host OS.

**Safe procedure:**

1. Use an institution-approved **isolated research workstation** to obtain or stage APKs. Do not download, preview, or execute untrusted malware on your personal computer. A controlled malware sandbox requires separate deployment and review.
2. In the new tab, enter the absolute local path to the staged dataset folder, then click **Scan local APK inventory**. The scanner reads filenames, sizes, and optionally SHA-256 hashes. Export the manifest CSV. The scanner does **not** execute or parse APK files.
3. Obtain **sanitized dynamic behavioural logs** from an approved *external* sandbox workflow. Do not substitute static APK file names or package hashes for behaviour traces.
4. Under **Import extracted behavioural traces**, upload JSON, JSONL, or CSV. For JSON each record may include `sample_id`, `label`, `malware_family` or `family`, and an **ordered** `events` array (strings, or objects with `event`, `event_type`, `action`, `api_call`, etc.).
5. Download the normalized CSV and use the existing sidebar **Upload my behavioural traces** option to train the existing classifier.

Example JSON is provided in `example_extracted_traces.json`. It is entirely **illustrative** and should not be used as evidence of measured performance.

**Critical:** This collection appears to contain malware APKs only. Folder names can serve as provisional *malware family* labels. They do not provide a verified benign class; a binary classifier cannot be validly trained or assessed from those alone. Do not invent benign traces. Family-based multiclass training requires a separate experimental design and model output change. Beware cross-sample duplicate hashes, reused families, temporal biases, and train/test leakage.

This add-on does not automatically connect to Drive, run APKs, collect traces, claim dynamic data exists, or validate family ground truth.

**Tests:** `python -m unittest test_ingestion.py`.


## V3: offline sandbox-report adapter

New dashboard tab: **Sandbox report adapter**. Upload exported sanitized `.json` reports in either `events[]` or `behavior.processes[].calls[]` format. Optional metadata CSV columns: `source_report,sample_id,label,malware_family`. The adapter requires an explicit verified benign/malicious label, never assumes family name proves ground truth, preserves report-list order, and annotates provenance limitations. Download the model-ready CSV and upload it using the existing sidebar.

**Synthetic example:** `sample_sandbox_report.json` and `sample_report_metadata.csv` are invented demonstration values, not observations from the Google Drive APK collection.

**Safety:** No sandbox is deployed and no APK execution occurs in this app. Execute no malware on a host workstation; use a purpose-built authorized isolated system with strict network egress controls and a resettable snapshot. The process/call JSON adapter is a generic schema, not a verified drop-in format for every Cuckoo/DroidBox/MobSF release. Export/sanitize reports and adapt their fields if needed.

**Testing:** `python -m unittest test_ingestion test_sandbox_adapter -v`.


## Version 4.0: research integrity checks

New **Research validation** tab: declared SHA-256 match audit, timestamp ordering preview, and group-aware train/test exports. These are metadata validation tools, not malware execution or proof of dynamic trace authenticity.

- `research_validation.py`: `check_provenance`, `ordered_events`, `audit_and_split`.
- `test_research_validation.py`: new regression tests.
- Do not train on raw APK files. Do not execute unknown APKs on host machines.
- Original Train & evaluate tab retains the prior random split and is unsuitable for claiming group-disjoint evaluation. Export a holdout with the new tab for subsequent research-grade training.
- A reported hash matching an APK's computed digest does not independently prove that report came from execution of that APK; maintain signed sandbox run records when available.
- Family-disjoint splitting may produce single-class partitions if all benign samples belong to the `benign` family; inspect the diagnostics and choose a task-appropriate evaluation design.
