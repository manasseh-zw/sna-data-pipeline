You are helping build a data engineering pipeline for a Shona language (sna) speech dataset. This is a capstone project where data engineering is a graded objective, so audit reports and clean documentation matter as much as the code itself.

**What we are building**

A Modal-based data cleaning and preparation pipeline that takes the raw `google/WaxalNLP` Shona ASR dataset and produces a cleaned, annotated dataset ready for two downstream tasks: TTS fine-tuning on Sesame CSM-1B, and ASR fine-tuning on Whisper-small. The pipeline lives in its own repository `sna-data-pipeline` and is completely separate from any model training code.

**Infrastructure**

We use Modal for all compute. Every script is a Modal app with a single function that runs remotely. Locally we only need `modal` and `python-dotenv` installed. All heavy dependencies are installed inside the Modal image definition at the top of each script. We have one Modal volume called `sna-data-vol` mounted at `/data` inside every container. The folder structure inside that volume is:

```
/data/raw/        — ingested dataset saved here
/data/refined/    — output of audio and text cleaning
/data/final/      — split, normalised, upload-ready
/data/reports/    — all audit JSON files from every phase
```

We have a second volume called `sna-weights-vol` which is reserved for model checkpoints in the training repo and is not used here. Secrets are loaded via `modal.Secret.from_dotenv()`. The `.env` file contains `HF_TOKEN` and `HF_USERNAME`.

**Source dataset**

`google/WaxalNLP`, subset `sna_asr`. It has four splits: train (14.1k rows), validation (1.73k rows), test (1.75k rows), and unlabeled (85.4k rows — audio only, no transcriptions). We only work with the three labeled splits, concatenated into a single flat dataset of ~17.5k rows. The unlabeled split is ignored for now but will be used later for pseudo-labeling once a working Whisper-Shona model exists.

Source columns from WaxalNLP: `id`, `speaker_id`, `transcription`, `gender`, `language`, `audio`. At ingest we immediately rename `id` → `source_id` and `speaker_id` → `source_speaker_id` to preserve provenance. These original values are never overwritten.

**Final dataset schema**

Every row in the cleaned dataset must have exactly these columns:

```
audio                 — trimmed, 16kHz, LUFS-normalised float32
transcription         — normalised Shona text
source_id             — original sna_XXXXX id from WaxalNLP
source_speaker_id     — original speaker hash from WaxalNLP
speaker_idx           — stable integer 0..N, sorted by speaker frequency descending
gender                — Male / Female / Other from source
snr_db                — signal to noise ratio in dB
speech_ratio          — fraction of frames containing speech
quality_score         — composite SNR score used for ranking
duration              - in seconds
```

The column `selection_tier` from the old pipeline is dropped. There is no top-K filtering — all 17.5k rows are kept unless they are hard failures (VAD finds zero speech, trimmed audio is empty). Duration filtering is NOT done at pipeline time — it is a training-time decision. The `tts_eligible` and `is_clipped` flags let training scripts filter appropriately without re-running the pipeline.

**Pipeline phases and scripts**

The repo structure is:

```
sna-data-pipeline/
├── src/
│   ├── ingest.py
│   ├── normalize_text.py
│   ├── curate.py
│   ├── diarize.py
│   ├── normalize_audio.py
│   ├── split_and_upload.py
│   └── audit.py
├── reports/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

**Phase 1 — ingest.py** (already written and working)

Pulls train + validation + test splits from WaxalNLP, concatenates them, renames `id` and `speaker_id` for provenance, builds `speaker_idx` integer mapping sorted by speaker frequency descending, adds `tts_eligible` flag, prints a full speaker and gender distribution audit to stdout, writes `01_ingest_audit.json` to `/data/reports/`, saves the dataset to `/data/raw/`.

**Phase 2 — normalize_text.py**

Loads from `/data/raw/`. Applies text normalisation: lowercase, smart apostrophe fusion to ASCII apostrophe, replace dashes/hyphens with spaces, strip characters outside `[a-z0-9.,?' ]`, collapse whitespace. Adds `has_punctuation` boolean column. Prints a character frequency audit to catch unexpected characters or encoding artefacts. Writes `02_normalize_text_audit.json` to `/data/reports/`. Saves to `/data/refined/` as an intermediate checkpoint.

**Phase 3 — curate.py**

Loads from `/data/refined/`. Resamples all audio to 16kHz mono. Runs WebRTC VAD with frame smoothing (existing logic from v1 was good — keep it). Trims leading and trailing silence with a 0.4s buffer. Then applies intra-utterance gap trimming — this is the most important new step: using the VAD frame mask, find gaps between speech segments within the utterance. If `has_punctuation=False`, trim any internal gap >150ms to 80ms. If `has_punctuation=True`, use 130ms target at commas and 220ms at periods. Computes SNR, speech_ratio, speech_seconds, trimmed_duration_s, quality_score. Adds `is_clipped` flag. Only hard-drops rows where VAD finds zero speech or trimmed audio length is zero. Writes `03_curate_audit.json`. Saves back to `/data/refined/` overwriting the intermediate.

**Phase 4 — normalize_audio.py**

Loads from `/data/refined/`. Applies per-clip LUFS normalisation to -23 LUFS using `pyloudnorm`. Run after gap trimming so silence is not included in loudness calculation. Writes `04_normalize_audio_audit.json`. Saves back to `/data/refined/`.

**Phase 5 — split_and_upload.py**

Loads from `/data/refined/`. Performs a stratified train/valid/test split by `speaker_idx` so every speaker appears proportionally in all three splits (80/10/10). Uses `sklearn.model_selection.StratifiedShuffleSplit`. Saves the final `DatasetDict` to `/data/final/`. Then pushes to HuggingFace as `{HF_USERNAME}/sna-dataset-v3` with the dataset card describing the schema. Writes `05_split_audit.json`.

**Phase 6 — audit.py**

Loads from `/data/final/`. Produces a comprehensive summary report across the full pipeline: total clips, total speech hours, speaker distribution, SNR statistics, speech ratio distribution, TTS-eligible vs ASR-only clip counts, gender balance, duration histogram buckets. This is the capstone-facing report. Writes `06_final_audit.json` and prints a formatted summary to stdout.

**Key design decisions to always respect**

Never filter by duration in the pipeline — record `trimmed_duration_s` and let training scripts decide. Never overwrite `source_speaker_id` or `source_id`. The `speaker_idx` mapping is written to `/data/reports/speaker_idx_mapping.json` at ingest time and must remain stable — do not recompute it in later phases. All filtering is done via boolean flag columns, not by dropping rows, except hard failures. Each script reads from one path and writes to one path — no script reads and writes to the same path in a way that could corrupt the dataset if it fails halfway (write to a temp path then rename if needed). Every script writes its own numbered audit JSON to `/data/reports/`. The pipeline is designed to be run in order: 1 → 2 → 3 → 4 → 5 → 6.

**Current state**

Phase 1 `ingest.py` is complete and working. The next script to write is `normalize_text.py`.

---

That gives the agent everything it needs. Hand it that and say "write `normalize_text.py` next" and it will have full context to continue without you re-explaining anything.
