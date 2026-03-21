# Audit Speaker Clustering (Isolated)

This folder is an isolated workflow for local speaker clustering audits, with pinned dependency versions that are known to be compatible with SpeechBrain ECAPA loading.

## 1) Create isolated environment

From repo root:

```bash
python3 -m venv .venv-audit-speaker
source .venv-audit-speaker/bin/activate
pip install -r src/tests/audio/audit_speaker/requirements.txt
```

## 2) Run the audit

```bash
python src/tests/audio/audit_speaker/audit_speaker_clusters.py
```

## Inputs / Outputs

- Input samples: `src/tests/audio/speaker_samples`
- Outputs: `src/tests/audio/speaker_samples/audit_clusters`

Generated files:

- `cluster_report.csv`
- `clip_assignments.csv`
- `cluster_summary.txt`
- `params_used.json`
