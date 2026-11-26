# Sequence-CloV v12.1 — Cloud deployment source

This is an **unchanged research prototype code baseline** with a deployment guide and non-secret configuration files added. It has **not** been deployed to a live public URL and its web UI has not been browser-tested in cloud.

## Recommended demonstration use
Use only the synthetic example in `example_v12_release_SYNTHETIC.zip`. Do not upload malware APKs, SELENE raw datasets, genuine sandbox logs, identifiable records, or secrets to public hosting.

## Steps
1. Create a new GitHub repository. Initially choose **private**.
2. Upload all these files so that `app.py` and `requirements.txt` are at the repo root. Do not upload this ZIP as one file.
3. Go to https://share.streamlit.io, connect GitHub, create an app, select your repo and `app.py`.
4. Set a Python version supported by torch and other dependencies in the hosting UI; avoid assuming the user's local Python 3.14 will be used.
5. Deploy and verify the generated `*.streamlit.app` URL. Choose restricted access if your deployment plan supports it.
6. Rehearse the synthetic example and prepare a local backup.

## Notes
- Model training can exceed community service resources; use small example and model selections for demonstration.
- No APK should execute on this server.
- A specific custom domain is **not** provisioned. Names depend on availability and hosting configuration.
- See https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
