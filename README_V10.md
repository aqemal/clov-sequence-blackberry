# Sequence-CloV v10 — Presentation-first dashboard

Run from the `Sequence-CloV_Prototype` directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## What's new

The main page immediately displays a per-sample word cloud, behavioural importance chart and sequence-order timeline. No training is needed for the **event frequency** view. After you train a model in **Train & evaluate**, switch to **Model deletion impact** to display model sensitivity. A prediction score and optional Transformer self-attention heatmap also become available. Training and model settings must stay the same when switching tabs.

Model deletion impact computes the absolute change in malignant-class probability after one event occurrence is removed. It is a model sensitivity diagnostic, not proof that an event caused a cyberattack. Word clouds from event frequency and model attribution must not be conflated. This view uses at most the first 48 events for computationally intensive event deletion; the full original sequence remains in the frequency and timeline views.

No APK execution or extraction occurs in this app. Synthetic examples are illustrative and must not be reported as measured malware accuracy. The real APK Drive collection still requires independently verified and authorized isolated sandbox execution to generate behavioural reports.

## Test
`python -m unittest discover -p 'test_*.py'`
