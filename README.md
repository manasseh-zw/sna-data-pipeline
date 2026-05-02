# SNA Data Pipeline

A reproducible, Modal-based data engineering pipeline that transforms raw Shona speech data from [Google's WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) into cleaned, metadata-rich datasets ready for downstream ASR and TTS workflows.

> **Two published datasets were produced by this pipeline:**
>
> | Dataset | Clips | Hours | Speakers | Description |
> |---------|------:|------:|---------:|-------------|
> | [manassehzw/sna-dataset](https://huggingface.co/datasets/manassehzw/sna-dataset) | 16,980 | 86.2 h | 133 | Cleaned base release with quality metrics |
> | [manassehzw/sna-dataset-annotated](https://huggingface.co/datasets/manassehzw/sna-dataset-annotated) | 15,239 | 78.5 h | 46 | Speaker-relabelled, loudness-normalised, decontaminated |

---

## Table of Contents

- [Motivation](#motivation)
- [Pipeline Overview](#pipeline-overview)
- [Pipeline Phases (Detailed)](#pipeline-phases-detailed)
  - [Phase 1 — Ingest](#phase-1--ingest)
  - [Phase 2 — Metadata Annotation](#phase-2--metadata-annotation)
  - [Phase 3 — Text Normalization](#phase-3--text-normalization)
  - [Phase 4 — Audio Normalization](#phase-4--audio-normalization)
  - [Phase 5 — Cleanup](#phase-5--cleanup)
  - [Phase 6 — Split & Upload (sna-dataset)](#phase-6--split--upload-sna-dataset)
  - [Phase 7 — Speaker Classification & Relabelling](#phase-7--speaker-classification--relabelling)
  - [Phase 8 — Rebuild Annotated Dataset](#phase-8--rebuild-annotated-dataset)
  - [Phase 9 — Upload Annotated (sna-dataset-annotated)](#phase-9--upload-annotated-sna-dataset-annotated)
- [Data Contamination: The Problem We Solved](#data-contamination-the-problem-we-solved)
- [Custom Gender Classifier](#custom-gender-classifier)
- [Pipeline Metrics Summary](#pipeline-metrics-summary)
- [Infrastructure](#infrastructure)
- [Run Order](#run-order)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Motivation

Google's WaxalNLP is a large-scale multilingual speech dataset released at LREC-COLING 2024. The `sna_asr` subset contains **~17,600 labeled Shona speech clips** across train, validation, and test splits — a valuable resource for a low-resource language. However, we discovered significant data quality issues:

- **Speaker label contamination** — multiple distinct voices were assigned to the same speaker ID, and single voices were split across multiple IDs
- **Inconsistent metadata** — gender labels had mixed casing (`Male` vs `male`), speaker IDs were opaque Firebase hashes
- **No audio quality metrics** — no SNR, speech ratio, or duration metadata for filtering
- **No loudness standardisation** — clips varied widely in volume (input LUFS ranged from −46 to −10)
- **Noisy audio** — leading/trailing silence, excessive internal gaps, and blacklist-worthy speakers with consistently distorted audio

This pipeline addresses all of the above through a systematic, audited, multi-phase process.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          google/WaxalNLP                               │
│                     sna_asr subset (17,585 clips)                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Phase 1 · INGEST           │
                │   Concatenate splits,        │
                │   rename for provenance,     │
                │   assign speaker_idx         │
                │   → 17,585 clips             │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Phase 2 · METADATA         │
                │   Normalize gender/language,  │
                │   add speaker_clip_count     │
                │   → 17,585 clips             │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Phase 3 · TEXT NORM        │
                │   Clean transcriptions,      │
                │   strip bad chars,           │
                │   add has_punctuation        │
                │   → 17,585 clips             │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Phase 4 · AUDIO NORM       │
                │   Resample 24kHz mono,       │
                │   VAD trim + gap compress,   │
                │   compute SNR / quality,     │
                │   blacklist bad speakers     │
                │   → 17,400 clips (−185)      │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Phase 5 · CLEANUP          │
                │   Drop <5s clips,            │
                │   drop singleton speakers    │
                │   → 16,980 clips (−420)      │
                └──────────────┬───────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌───────────────────────┐         ┌──────────────────────────────┐
│  Phase 6 · SPLIT &    │         │  Phase 7 · SPEAKER           │
│  UPLOAD               │         │  CLASSIFICATION              │
│  80/10/10 split,      │         │  ECAPA embeddings,           │
│  push to HuggingFace  │         │  HDBSCAN clustering,         │
│                       │         │  noise rescue,               │
│  ┌─────────────────┐  │         │  gender reclassification     │
│  │ sna-dataset      │  │         │  168 → 46 true speakers      │
│  │ 16,980 clips     │  │         │  1,741 noise clips dropped   │
│  │ 86.2 hours       │  │         └──────────────┬───────────────┘
│  │ 133 speakers     │  │                        │
│  └─────────────────┘  │                        ▼
└───────────────────────┘         ┌──────────────────────────────┐
                                  │  Phase 8 · REBUILD           │
                                  │  ANNOTATED                   │
                                  │  Apply relabel mapping,      │
                                  │  LUFS loudness norm (−23),   │
                                  │  speaker-stratified split    │
                                  │  → 15,239 clips              │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  Phase 9 · UPLOAD            │
                                  │  ANNOTATED                   │
                                  │  Push to HuggingFace         │
                                  │                              │
                                  │  ┌─────────────────────────┐ │
                                  │  │ sna-dataset-annotated    │ │
                                  │  │ 15,239 clips             │ │
                                  │  │ 78.5 hours               │ │
                                  │  │ 46 speakers              │ │
                                  │  └─────────────────────────┘ │
                                  └──────────────────────────────┘
```

---

## Pipeline Phases (Detailed)

### Phase 1 — Ingest

**Script:** `src/ingest.py`

Pulls the three labeled splits (train, validation, test) from `google/WaxalNLP` subset `sna_asr` and concatenates them into a single flat dataset.

**Key operations:**
- Rename `id` → `source_id` and `speaker_id` → `source_speaker_id` to preserve provenance
- Build a stable `speaker_idx` mapping sorted by speaker frequency (descending)
- Write `01_ingest_audit.json`

**Audit snapshot:**
| Metric | Value |
|--------|-------|
| Total rows ingested | 17,585 |
| Unique speakers | 168 |
| Gender distribution | Female: 9,385 · Male: 5,688 · male: 1,735 · female: 777 |
| Mean transcription length | 192 chars |

> Note the inconsistent gender casing — `Male` vs `male`, `Female` vs `female` — inherited from the source dataset. This is fixed in Phase 2.

---

### Phase 2 — Metadata Annotation

**Script:** `src/annotate_metadata.py`

**Key operations:**
- Normalize `gender` to title case (`Male`/`Female`) — resolving 2,512 inconsistently cased labels
- Normalize `language` to lowercase (`sna`)
- Add `speaker_clip_count` column derived from speaker frequency

**Before → After gender normalization:**
| Label | Before | After |
|-------|-------:|------:|
| Female | 9,385 + 777 | **10,162** |
| Male | 5,688 + 1,735 | **7,423** |

---

### Phase 3 — Text Normalization

**Script:** `src/normalize_text.py`

Cleans transcriptions while preserving Shona-appropriate casing and punctuation.

**Normalization rules:**
- Strip smart quotes → ASCII apostrophe
- Collapse em/en dashes → spaces
- Normalize spaced hyphens
- Insert space after sentence-ending periods followed by capitals
- Strip characters outside `[A-Za-z0-9.,?!'" -]`
- Collapse whitespace

**Unexpected characters cleaned:**

| Character | Count before | Count after |
|-----------|------------:|------------:|
| `"` (double quote) | 358 | 358 (preserved) |
| `\n` (newline) | 291 | 0 |
| `(` / `)` | 117 each | 0 |
| Accented chars (í, à, ú, etc.) | 126 total | 0 |
| `&`, `` ` ``, `;`, `/` | 16 total | 0 |

**Added column:** `has_punctuation` — 98.91% of clips have sentence punctuation.

---

### Phase 4 — Audio Normalization

**Script:** `src/normalize_audio.py`

The core audio processing phase. Every clip goes through a multi-step cleanup pipeline:

1. **Resample** to 24kHz mono
2. **WebRTC VAD** (aggressiveness=2, 30ms frames) with smoothing:
   - Drop speech bursts < 3 frames
   - Bridge gaps ≤ 2 frames
3. **Leading/trailing silence trim** with 0.4s buffer
4. **Internal gap compression** — gaps > 150ms are trimmed to 90ms
5. **Quality metrics computed:**
   - `snr_db` — signal-to-noise ratio
   - `speech_ratio` — fraction of VAD frames classified as speech
   - `quality_score` — composite: `snr_db` minus reliability penalties
   - `duration` — post-processing clip duration
6. **Hard-drop** clips with zero speech, empty audio, or from blacklisted speakers

**Blacklisted speakers:**
- `DVRNxPvJnmebFbLnQhG9VSCLhdf2` — 185 clips, all distorted/mumbled (confirmed by manual review)

**Audit snapshot:**

| Metric | Value |
|--------|-------|
| Input rows | 17,585 |
| Kept | 17,400 |
| Dropped | 185 (all blacklisted speaker) |
| Total hours | 86.4 h |
| Mean SNR | 67.84 dB |
| Mean speech ratio | 0.953 |
| Duration: 10–20s | 12,034 clips (69.2%) |

---

### Phase 5 — Cleanup

**Script:** `src/cleanup_audio.py`

Post-normalization cleanup to remove low-value data:

- **Drop clips under 5 seconds** — too short for meaningful speech (393 clips)
- **Drop singleton speakers** — speakers with only 1 remaining clip (27 clips)
- **Refresh** `speaker_clip_count`

**Audit snapshot:**

| Metric | Value |
|--------|-------|
| Input rows | 17,400 |
| Kept | **16,980** |
| Dropped | 420 (2.41%) |
| Unique speakers remaining | 133 |
| Total hours | **86.222 h** |

---

### Phase 6 — Split & Upload (`sna-dataset`)

**Script:** `src/split_and_upload.py`

Performs speaker-stratified 80/10/10 split and publishes to HuggingFace.

| Split | Clips | Unique Speakers |
|-------|------:|----------------:|
| Train | 13,532 | 133 |
| Validation | 1,640 | 85 |
| Test | 1,808 | 133 |

> **Published as:** [`manassehzw/sna-dataset`](https://huggingface.co/datasets/manassehzw/sna-dataset) — 16,980 clips, 86.2 hours, 133 source speakers

---

### Phase 7 — Speaker Classification & Relabelling

**Script:** `src/classify_speakers.py`

This is the most complex phase — addressing the **speaker label contamination** problem discovered in the source dataset.

#### The Problem

The WaxalNLP source labels treated each contributor's Firebase UID as a unique speaker. In reality:
- Multiple distinct voices were recorded under the same UID (e.g., shared devices)
- The same voice appeared under different UIDs
- This made the 133 "speakers" unreliable for any speaker-dependent task

#### The Solution

A three-stage acoustic speaker classification pipeline:

**Stage 1 — Embedding Extraction:**
- Extract 192-dimensional speaker embeddings using [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (`speechbrain/spkrec-ecapa-voxceleb`)
- Process all 16,980 clips in GPU-accelerated batches with adaptive batch sizing and OOM recovery
- L2-normalize all embeddings

**Stage 2 — Gender Classification:**
- Classify each clip's gender using a [custom-trained Logistic Regression model](#custom-gender-classifier) on L2-normalised ECAPA embeddings
- Partition clips into Female / Male / Unknown sets (threshold: 0.65 confidence)
- Gender distribution after classification: Female 8,338 · Male 7,646 · Unknown 996

**Stage 3 — Acoustic Clustering:**
- Run [HDBSCAN](https://hdbscan.readthedocs.io/) independently on Female and Male partitions (prevents mixed-gender clusters by construction)
- Parameters: `min_cluster_size=50`, `min_samples=10`, metric `euclidean`, selection method `eom`
- **Noise rescue:** unassigned clips with cosine similarity ≥ 0.75 to a cluster centroid are rescued

**Results:**

| Metric | Value |
|--------|-------|
| Speaker clusters found | **46** (down from 133 source labels) |
| Mixed-gender clusters | **0** |
| Noise before rescue | 2,535 clips |
| Clips rescued | 794 |
| Noise after rescue | **1,741** clips (10.3%) |
| Runtime | ~90 minutes (A10G GPU) |

---

### Phase 8 — Rebuild Annotated Dataset

**Script:** `src/rebuild_annotated.py`

Applies the relabel mapping and produces the final annotated dataset.

**Key operations:**
1. **Drop noise** — remove 1,741 clips with `cluster_id == -1` (unrescued noise)
2. **Remap schema** — `cluster_id` → `speaker_id`, `cluster_gender` → `gender`
3. **Recompute** `speaker_clip_count` from new speaker assignments
4. **LUFS loudness normalization** — target −23 LUFS (EBU R128 standard)
   - Skip tolerance: ±1 LU
   - Post-gain protection: hard clip to [−1.0, 1.0]
   - 12,458 clips normalised, 2,781 skipped (already within tolerance)
5. **Speaker-stratified 80/10/10 split**

**Loudness normalization impact:**

| Metric | Input | Output |
|--------|------:|-------:|
| Mean LUFS | −22.659 | **−22.999** |
| Std LUFS | 5.301 | **0.243** |
| Min LUFS | −46.381 | −24.379 |
| Max LUFS | −9.712 | −22.0 |

> The standard deviation dropped from **5.3 LU to 0.24 LU** — a 22× improvement in loudness consistency.

---

### Phase 9 — Upload Annotated (`sna-dataset-annotated`)

**Script:** `src/upload_annotated.py`

Pushes the rebuilt dataset with dataset card to HuggingFace.

> **Published as:** [`manassehzw/sna-dataset-annotated`](https://huggingface.co/datasets/manassehzw/sna-dataset-annotated)

| Split | Clips |
|-------|------:|
| Train | 12,170 |
| Validation | 1,504 |
| Test | 1,565 |
| **Total** | **15,239** |

| Metric | Value |
|--------|-------|
| Total hours | 78.5 h |
| Unique speakers | 46 |
| Gender distribution | Female 7,420 · Male 7,163 · Unknown 656 |

---

## Data Contamination: The Problem We Solved

The most significant finding during this pipeline's development was **widespread speaker label contamination** in the WaxalNLP source data.

### Before Classification (Source Labels)

- **133 unique speaker IDs** — each a Firebase UID from the crowdsourcing platform
- Speaker assignments were based on the account that uploaded the clip, not the voice in the clip
- **Result:** Multiple distinct voices shared a single UID (likely shared devices), and the same voice could appear under different UIDs

### After Acoustic Classification

- **46 true acoustic speakers** identified through ECAPA embedding clustering
- **1,741 clips** (10.3%) could not be reliably assigned to any speaker cluster and were dropped as noise
- The 133 → 46 reduction demonstrates that nearly 2/3 of the original "speakers" were either duplicates of other speakers or unreliable fragments

### Why This Matters

For any speaker-dependent task (TTS, speaker verification, speaker-stratified evaluation), the source labels would have produced:
- **Contaminated training data** — different voices mixed under one "speaker"
- **Invalid evaluations** — test sets potentially containing the same voice as training sets under different IDs
- **Poor TTS quality** — models trying to learn a single voice from a mix of voices

The annotated dataset resolves this with acoustically-verified speaker assignments and per-clip confidence scores.

---

## Custom Gender Classifier

Existing pretrained gender classifiers (e.g., `prithivMLmods/Common-Voice-Gender-Detection` based on Wav2Vec2) produced **confident mispredictions on Shona speech** due to training distribution mismatch with African language prosody.

We trained a Shona-calibrated gender classifier from scratch:

| Property | Value |
|----------|-------|
| Architecture | Logistic Regression on L2-normalised 192-d ECAPA-TDNN embeddings |
| Training data | 312 manually ear-tested clips (160 female, 152 male) |
| Train accuracy | 100% |
| 5-fold CV accuracy | **100%** |
| Unknown rate (probe on 777 clips) | 9.4% (below 0.65 confidence threshold) |
| Confident wrong-gender predictions | **0** on verified speakers |

The classifier is lightweight (a single `.pkl` file) and runs inference on CPU in milliseconds — no GPU required for gender prediction.

---

## Pipeline Metrics Summary

**Full pipeline data flow:**

```
google/WaxalNLP sna_asr
        │
        ▼
   17,585 clips ingested
        │
        │  −185  blacklisted speaker (distorted/mumbled)
        ▼
   17,400 clips after audio normalization
        │
        │  −393  clips under 5 seconds
        │   −27  singleton speakers
        ▼
   16,980 clips → sna-dataset (86.2 hours, 133 source speakers)
        │
        │  −1,741 noise clips (unassignable to any speaker cluster)
        ▼
   15,239 clips → sna-dataset-annotated (78.5 hours, 46 true speakers)
```

**Quality metrics after audio normalization (16,980 clips):**

| Metric | Mean | Std | Min | Max |
|--------|-----:|----:|----:|----:|
| SNR (dB) | 67.84 | 12.23 | 3.07 | 90.11 |
| Speech ratio | 0.953 | 0.027 | 0.647 | 1.0 |
| Duration (s) | 18.28 | — | 3.24 | 35.28 |

---

## Infrastructure

All compute runs on [Modal](https://modal.com/). Each pipeline script is a Modal app with a single remote function. Locally, only `modal` and `python-dotenv` are needed.

**Modal volume:** `sna-data-vol` mounted at `/data` inside every container.

```
/data/raw/              — ingested + metadata-annotated dataset
/data/refined/          — output of text and audio cleaning phases
/data/final/            — split, normalised, upload-ready
/data/wav_cache/        — extracted WAV files at 24kHz (one per clip)
/data/wav_normalised/   — LUFS-normalised WAVs for annotated build
/data/models/           — model artifacts (e.g. gender_classifier_ecapa.pkl)
/data/relabel/          — speaker classification outputs (mapping CSV, cluster report)
/data/sna_annotated/    — final annotated DatasetDict
/data/reports/          — audit JSON files from every phase
```

Secrets are loaded via `modal.Secret.from_dotenv()` — the `.env` file contains `HF_TOKEN` and `HF_USERNAME`.

---

## Run Order

### Core Pipeline (produces `sna-dataset`)

```bash
uv run modal run src/ingest.py
uv run modal run src/annotate_metadata.py
uv run modal run src/normalize_text.py
uv run modal run src/normalize_audio.py
uv run modal run src/cleanup_audio.py
uv run modal run src/split_and_upload.py
```

### Speaker Relabelling Pipeline (produces `sna-dataset-annotated`)

```bash
uv run modal run src/pre_classification_audit.py
uv run modal run src/classify_speakers.py
uv run modal run src/rebuild_annotated.py
uv run modal run src/upload_annotated.py
```

---

## Repository Structure

```
sna-data-pipeline/
├── src/
│   ├── ingest.py                    — Phase 1: ingest from WaxalNLP
│   ├── annotate_metadata.py         — Phase 2: normalize metadata fields
│   ├── normalize_text.py            — Phase 3: clean transcriptions
│   ├── normalize_audio.py           — Phase 4: VAD trim, gap compress, quality metrics
│   ├── cleanup_audio.py             — Phase 5: drop short/singleton clips
│   ├── split_and_upload.py          — Phase 6: stratified split → HuggingFace
│   ├── pre_classification_audit.py  — Pre-classification baseline snapshot
│   ├── classify_speakers.py         — Phase 7: ECAPA + HDBSCAN speaker relabelling
│   ├── rebuild_annotated.py         — Phase 8: apply relabels + LUFS normalization
│   ├── upload_annotated.py          — Phase 9: push annotated dataset to HuggingFace
│   ├── speaker_analysis.py          — Analysis utility (not a pipeline phase)
│   ├── export_speaker.py            — Export clips for a specific speaker
│   ├── build_tts_sesame_export.py   — Build TTS training export
│   └── tests/                       — Test harnesses and validation scripts
├── reports/                         — Audit JSON files from every pipeline phase
├── models/                          — Model artifacts (gitignored)
├── .docs/                           — Internal planning and context documents
├── pyproject.toml
└── README.md
```

---

## Citation

If you use either dataset, please cite the original source:

```bibtex
@inproceedings{niang2024waxalnlp,
  title     = {WaxalNLP: A Large Scale High Quality Speech Dataset for African Languages},
  author    = {Niang, El Hadj Mamadou and Dieng, Moustapha and Ba, Thierno Ibrahima
               and Ndiaye, Mamadou Boumedine and others},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational
               Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year      = {2024}
}
```

---

*Curated by [Manasseh Changachirere](https://www.manasseh.dev/) — Harare Institute of Technology*
