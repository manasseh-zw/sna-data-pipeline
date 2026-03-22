You are helping build a data engineering pipeline for a Shona language (sna) speech dataset. This is a capstone project where data engineering is a graded objective, so audit reports and clean documentation matter as much as the code itself.

**What we are building**

A Modal-based data cleaning and preparation pipeline that takes the raw `google/WaxalNLP` Shona ASR dataset and produces a cleaned, annotated dataset for downstream use. The pipeline lives in its own repository `sna-data-pipeline` and is completely separate from model training code. The goal is a general-purpose, opinionation-free dataset — no training-time decisions baked in. Consumers filter using metadata columns.

The owner has a prior published version (`manassehzw/sna-tts-refined-v2`, 5,000 clips, ~21.66h) produced without speaker tracking or manual quality auditing. This pipeline is the rigorous replacement.

Status update: `manassehzw/sna-dataset` is already published from this pipeline. Current priority is now a second pass focused on speaker-label quality (`sna-dataset-labeled`, effectively a v2 labeling pass), not immediate TTS subset curation.

---

**Infrastructure**

We use Modal for all compute. Every pipeline script is a Modal app with a single function that runs remotely. Locally we only need `modal` and `python-dotenv` installed. All heavy dependencies are installed inside the Modal image definition at the top of each script. We have one Modal volume called `sna-data-vol` mounted at `/data` inside every container. The folder structure inside that volume is:

```
/data/raw/           — ingested + metadata-annotated dataset
/data/refined/       — output of text and audio cleaning phases
/data/final/         — split, normalised, upload-ready
/data/reports/       — all audit JSON files from every phase
/data/curate_test/   — temporary: 500-clip sample for local curation testing
/data/speaker_audit/ — earlier speaker ranking report + sample clips
/data/speaker_samples/ — pulled stratified clips per top speaker for local identity audit
/data/wav_cache/     — extracted WAV files at 24kHz, one per clip named {source_id}.wav
                       written by classify_speakers.py; also used by later audio normalization
/data/models/        — model artifacts uploaded from local (e.g. gender_classifier_ecapa.pkl)
/data/relabel/       — output of classify_speakers.py: relabel_mapping.csv, cluster_report.csv
```

Secrets are loaded via `modal.Secret.from_dotenv()`. The `.env` file contains `HF_TOKEN` and `HF_USERNAME`.

---

**Source dataset**

`google/WaxalNLP`, subset `sna_asr`. Three labeled splits: train (14.1k), validation (1.73k), test (1.75k) — concatenated into a flat dataset of 17,585 rows. The unlabeled split (85.4k, audio only) is out of scope for this pipeline.

Source columns: `id`, `speaker_id`, `transcription`, `gender`, `language`, `audio`. At ingest we rename `id` → `source_id` and `speaker_id` → `source_speaker_id`. These are never overwritten.

---

**Final dataset schema**

```
audio                 — trimmed, 24kHz, LUFS-normalised float32
transcription         — normalised Shona text
source_id             — original id from WaxalNLP
source_speaker_id     — original speaker hash from WaxalNLP
speaker_idx           — stable integer 0..N sorted by speaker frequency descending
speaker_clip_count    — total clips for this speaker across the full dataset
language              — normalised to lowercase (sna)
gender                — normalised to Male / Female
has_punctuation       — boolean derived from normalised transcription
snr_db                — signal-to-noise ratio in dB
speech_ratio          — fraction of VAD frames classified as speech
quality_score         — composite score: snr_db minus reliability penalties
duration              — trimmed clip duration in seconds
```

No opinionated flag columns. Consumers filter using `speaker_clip_count`, `snr_db`, `speech_ratio`, and `duration` directly.

---

**Key design decisions — always respect these**

- Hard-drop clips shorter than 5 seconds at ingest.
- Never overwrite `source_speaker_id` or `source_id`.
- `speaker_idx` mapping is stable from ingest — never recompute it in later phases.
- Hard-drop only: duration < 5s at ingest, VAD finds zero speech, trimmed audio length is zero, or speaker is blacklisted.
- All other rows are kept regardless of quality metrics.
- Each script reads from one path and writes to one path. Never read and write to the same path without a temp → rename pattern.
- Every script writes a numbered audit JSON to `/data/reports/`.
- Pipeline runs in strict order: 1 → 2 → 3 → 4 → 5 → 6.
- All scripts run from the **repo root** (not from `src/`).

---

**Repo structure**

```
sna-data-pipeline/
├── src/
│   ├── ingest.py
│   ├── annotate_metadata.py
│   ├── speaker_analysis.py       — analysis utility, not a pipeline phase
│   ├── normalize_text.py
│   ├── normalize_audio.py        — phase script retained for reproducibility
│   ├── split_and_upload.py       — used for published `sna-dataset` release
│   ├── audit.py                  — final reporting utilities (status may vary by branch)
│   └── tests/
│       ├── text/
│       │   ├── unnormalized.txt
│       │   ├── normalized.txt
│       │   └── test_normalize.py
│       ├── audio/
│       │   ├── pull_samples.py        — Modal: pull 500 clips from volume → zip
│       │   ├── test_curate.py         — local: run normalize_audio logic + write audit outputs
│       │   ├── speaker_audit.py       — Modal: rank all speakers by talk time, pull 3 clips each
│       │   └── samples/               — gitignored
│       └── artifact_check/
│           ├── detect.py              — kurtosis + HF energy artifact detector (see note)
│           ├── input/                 — drop WAVs here to test
│           └── rejected/              — copies of flagged clips
├── .docs/
│   ├── context.md                     — this file
│   ├── audit_pre_post_classification.md — what to audit before/after classification; data for report figures
│   ├── phase_classify_speakers.md     — Modal full-dataset classify/label plan
│   └── artifact_detection_attempt.md  — documents the dropped artifact detection approach
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

**Phase 1 — ingest.py** ✅ complete

Pulls train + validation + test splits from WaxalNLP, concatenates them, hard-drops clips shorter than 5 seconds, renames columns for provenance, builds `speaker_idx` mapping sorted by speaker frequency descending, writes `01_ingest_audit.json`, saves to `/data/raw/`.

**Phase 2 — annotate_metadata.py** ✅ complete

Loads from `/data/raw/`. Normalises `gender` to `Male`/`Female`, normalises `language` to lowercase, adds `speaker_clip_count` integer column derived from speaker frequency. Writes back to `/data/raw/` via temp → rename. Writes `annotate_metadata_audit.json`.

**Phase 3 — normalize_text.py** ✅ complete

Loads from `/data/raw/`. Normalises transcriptions: strips smart quotes to ASCII apostrophe, collapses em/en dashes to spaces, normalises spaced hyphens, inserts space after sentence-ending period followed by capital, strips characters outside `[A-Za-z0-9.,?!'" -]`, collapses whitespace. Casing is preserved. Adds `has_punctuation` boolean. Writes `02_normalize_text_audit.json`. Saves to `/data/refined/`.

**Phase 4 — normalize_audio.py** ✅ completed historically

Loads from `/data/refined/`. Resamples to 24kHz mono. Runs WebRTC VAD (aggressiveness=2, 30ms frames) with smoothing (drop bursts <3 frames, bridge gaps ≤2 frames). Trims leading/trailing silence with 0.4s buffer. Applies flat intra-utterance gap trimming: any internal gap >150ms is trimmed to 80ms. Recomputes VAD mask on trimmed audio. Computes `snr_db`, `speech_ratio`, `quality_score`, `duration`. Hard-drops only rows where VAD finds zero speech or audio is empty after trimming. Also hard-drops blacklisted speakers (see below). Writes `04_normalize_audio_audit.json`. Saves back to `/data/refined/`.

**Blacklisted speakers in normalize_audio.py:**
```python
BLACKLISTED_SPEAKER_IDS = {
    "DVRNxPvJnmebFbLnQhG9VSCLhdf2",   # 185 clips, all distorted/mumbled — manual review
}
```
To add more: append to this set with a comment documenting the reason.

**Run command:** `modal run src/normalize_audio.py`

**Phase 5 — cleanup_audio.py** ✅ written

Loads from `/data/refined/` after Phase 4. Drops clips with `duration < 5s`, then drops clips from speakers that have only one remaining clip (singleton speakers). Refreshes `speaker_clip_count` from the post-cleanup dataset. Writes `05_cleanup_audio_audit.json`. Saves back to `/data/refined/` via temp → rename.

**Run command:** `modal run src/cleanup_audio.py`

**Phase 5.5 — pre_classification_audit.py** ✅ written

Runs before `classify_speakers.py`. Loads the flat dataset from `/data/refined/`. Computes a "before" snapshot: total clips, total hours, unique `source_speaker_id` count, gender distribution by original label, per-speaker clip count and majority gender, and flags speakers with conflicting gender labels in the source data. Writes `pre_classification_audit.json` to `/data/reports/` and `pre_audit_metadata.csv` to `/data/relabel/` (one row per clip — the "before" CSV for post-classification diff). Also extracts all clips to `/data/wav_cache/{source_id}.wav` at 24 kHz (idempotent — skips files that already exist). This WAV cache is reused by `classify_speakers.py` and all later audio passes, so extraction happens exactly once.

**Run command:** `modal run src/pre_classification_audit.py`

**Phase 6 — split_and_upload.py** ✅ completed historically

Loads from `/data/refined/`. Performs stratified 80/10/10 train/valid/test split by `speaker_idx`. Reorders columns for clean HuggingFace dataset viewer presentation. Saves `DatasetDict` to `/data/final/`. Pushes to HuggingFace as `{HF_USERNAME}/sna-dataset` with a dataset card. Writes `06_split_audit.json`.

**Phase 7 — audit.py** ✅ completed/iterated as needed

Loads from `/data/final/`. Produces capstone-facing summary: total clips, total hours, speaker distribution, SNR stats, speech ratio distribution, gender balance, duration histogram. Writes `07_final_audit.json`.

---

**Speaker audit findings (completed)**

Full dataset: **168 speakers, 17,585 clips, ~99.4 hours of speech.**

The top 20 speakers were manually ear-tested (3 clips each) and rated. Results:

| Quality | Speakers | Clips | Hours | Notes |
|---|---|---|---|---|
| Pristine | 7 | 5,155 | 28.5h | 3,801 M / 1,354 F |
| High | 4 | 1,839 | 10.5h | all female |
| Medium-High | 4 | 1,418 | 9.4h | mixed |
| Medium | 4 | 1,383 | 8.4h | mixed |
| Medium-Low | 1 | 1,549 | 9.8h | rank-1 speaker, male |

**8 junk speakers** (ranks 86, 95, 116, 164–168) have mean clip duration ~1s — likely upload errors. Only 380 clips, 0.11h. Consider adding to blacklist.

**Planned published datasets:**
1. **Full general dataset** (`sna-dataset`): all ~17k clips post-curation, all speakers, all quality levels. General-purpose, community contribution.
2. **Premium TTS subset** (`sna-tts-v3`): filter `source_speaker_id IN (pristine_set + high_set)` = 11 speakers, ~6,994 clips, ~39 hours. For TTS fine-tuning (e.g. Sesame CSM 1B). No reprocessing needed — just a metadata filter on top of the full dataset.

**Pristine speaker IDs** (for premium filter):
```
T33w3KIsJJYMb9tmz2XxSYlgpcA2   male
7UbpWlepR6OHT8S5tcibbkcQWOC2   male
2Eud8lyLlsMcciYhmlkwVRtBwi82   male
6yMVEvNaWJXi2UbPLujF6C4uGVJ3   female
1PHOsx2JIpUU4lMLgTd3ssjITjf1   male
CUBWhUHYrpdoA4u6bodqHGer0my1   female
CZQ37aLaUZfNpliatFqfC42MBUC3   female
```

**High quality speaker IDs** (also included in premium):
```
4HdcZXLtmjhpO6zY9qLnLNCs7OJ2   female (note: gender mislabelled in source)
f4LbqfoJ6HXJeYxnuIrfZlP7qaM2   female
akPZ3sZLNmWepmQ73pcfOBhbAXC3   female
2Q9Qx8uHQ5VAkWNdgqd7mVEeE2u1   female
```

---

**Dropped exploration: artifact detection**

Attempted kurtosis + high-frequency energy ratio detection for click/pop artifacts (`src/tests/artifact_check/detect.py`). False positive rate was ~93% on raw clips — Shona plosive consonants and natural speech variation are indistinguishable from artifacts at the signal level without a labelled baseline. Dropped. Documented in `.docs/artifact_detection_attempt.md` for dissertation reference.

---

**Gender classifier artifact**

A Shona-calibrated gender classifier has been built and validated as part of the relabeling workstream. It lives under `src/tests/audio/audit_speaker/` and is auto-loaded by the audit pipeline.

```
src/tests/audio/audit_speaker/
├── train_gender_classifier.py        — trains logistic regression on ECAPA embeddings
├── probe_gender_classifier.py        — validates classifier at scale, outputs female/male/unknown folders
├── gender_training_data/
│   ├── female/                       — labeled female WAV clips (gitignored)
│   └── male/                         — labeled male WAV clips (gitignored)
└── gender_classifier_ecapa.pkl       — trained model artifact (gitignored)
```

Key properties:
- Model: `sklearn.linear_model.LogisticRegression` on L2-normalised 192-d ECAPA-TDNN embeddings
- Training data: 391 clips total (bootstrapped from clean audit clusters + active learning ear-test pass)
- 5-fold CV accuracy: 100%
- Probe validation over 777 clips (39 speakers): 9.4% Unknown (clips below 0.65 confidence), zero confident wrong-gender predictions on verified speakers
- Supersedes `prithivMLmods/Common-Voice-Gender-Detection` (Wav2Vec2) which produced confident mispredictions on Shona speech due to training distribution mismatch

---

**Current state**

Core pipeline phases are complete and `sna-dataset` is published on Hugging Face. The active workstream is second-pass speaker identity relabeling.

The local clustering audit (`audit_speaker_clusters.py`) has completed two runs:
- **v2**: ECAPA + HDBSCAN + Wav2Vec2 gender model → 31 clusters, 4.6% noise, 4 MIXED_GENDER clusters. Identified gender model as unreliable on Shona.
- **v3 (pending)**: Same HDBSCAN setup with the new logistic regression gender classifier loaded from `.pkl`. Gender-separated clustering (HDBSCAN within each gender partition) to be explored.

The gender classifier has been validated via a probe script and an active learning ear-test loop. It is ready for the v3 audit run.

---

**Immediate next steps (in order)**

1. **Run pre-classification audit:** `modal run src/pre_classification_audit.py` — captures the "before" speaker/gender snapshot and extracts all WAV files to `/data/wav_cache/`. Must complete before classify_speakers.py.

2. **Upload gender classifier to volume:** `modal volume put sna-data-vol src/tests/audio/audit_speaker/gender_classifier_ecapa.pkl /models/gender_classifier_ecapa.pkl`

3. **Run full-dataset speaker classification:** Write and run `src/classify_speakers.py` (spec in `phase_classify_speakers.md`). Reads WAV cache, runs ECAPA + gender-separated HDBSCAN + noise rescue, writes `relabel_mapping.csv` and `cluster_report.csv` to `/data/relabel/`.

4. **Diff pre/post audit:** Join `pre_audit_metadata.csv` against `relabel_mapping.csv` on `source_id` to produce contamination figures for the dissertation.

5. **Apply relabel mapping:** Join mapping onto `/data/refined/` and publish as `sna-dataset-labeled`.

6. **Validate gender classifier on full 168-speaker set:** If Unknown rate < 12%, publish as `{HF_USERNAME}/sna-gender-shona`.

7. **Document contamination handling:** Record assumptions, thresholds, and known edge cases for dissertation reproducibility.

---

**Things noted but deliberately deferred**

- Premium TTS subset curation/selection is deferred until speaker relabeling quality is acceptable.
- Noise reduction / reverb correction: not applied. Full noise reduction (e.g. `noisereduce`) is deferred to future work.
