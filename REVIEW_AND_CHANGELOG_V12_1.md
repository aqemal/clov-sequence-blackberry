# Sequence-CloV Version 12.1 — Code Audit and Corrections

## Scope
This maintenance release is based on the complete Version 12 source code. It keeps the existing dashboard, benchmark models, XAI tabs, and offline-only treatment of APKs. It does not run malware, authenticate sandbox execution, or establish real-world accuracy.

## Corrected issues
1. **Silent record loss:** The Version 12 reader relied on normalization that silently dropped rows with empty extracted events. Version 12.1 rejects releases containing empty, null, or missing required values *before* normalization.
2. **Archive ambiguity:** ZIP archives with duplicate filenames could lead to ambiguous source data. They are now rejected; encrypted entries are also refused.
3. **Dataset changes not detected:** Fresh release ZIPs now contain SHA-256 digests of their exact train.csv and test.csv bytes, which Version 12.1 verifies. Older releases without these fields remain readable but cannot benefit from this checksum verification. This is not a digital signature and does not authenticate the source.
4. **Incomplete metadata checks:** Missing/invalid APK hashes, mismatched benign-family labels, and very small partitions now produce explicit validation errors.
5. **Training reproducibility:** Transformer seeds are applied to Python, NumPy and PyTorch, including the data-loader shuffle generator. Reproducibility may still vary across hardware, libraries and runtimes.
6. **Transformer residual architecture:** Corrected the attention residual and feed-forward residual paths to preserve input information.
7. **Empty selection:** A programmatic empty model selection is now rejected instead of silently defaulting to every model.
8. **Audit detail:** Exported experiment metadata now includes library versions, classification threshold and neural network settings.

## Evidence
- Automated test suite: 74 passed.
- End-to-end synthetic smoke test: 96 input traces, frozen ZIP, validation, all four models, results export, successful.
- Streamlit browser interaction not tested in this environment.

## Remaining limitations and required research controls
- No APK execution, malware sandboxing, or authentic dynamic traces. The 71 Drive families were observed through folder structure, not empirically analyzed.
- Dataset labels and declared hashes are not independent proof of provenance.
- Exact ID, hash and sequence matching does not rule out repackaged apps, shared family signatures, temporal and data-source leakage.
- 48-token truncation in neural models may remove relevant late events; validate its effect and try larger sequence lengths in future controlled experiments.
- The current neural models use a fixed threshold of 0.5 and no separate tuning/validation split. Avoid using held-out test results for hyperparameter changes.
- The sample ZIP is synthetic and must not be used for scientific performance claims.
- The existing XAI tools are exploratory; attention weights are not reliable causal explanations on their own.
- Model artifact serialization, full prediction provenance, confidence intervals, calibration and cross-dataset validation are future work.

## Upgrade
Extract into a **new** directory, install requirements, then run `python -m streamlit run app.py`. Your original Version 12 files are not modified.
