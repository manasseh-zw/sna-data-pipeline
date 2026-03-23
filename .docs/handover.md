# SNA Dataset — Phase A & B Handover Brief

# Agent: read this fully before writing any code.

# All scripts go in src/ and follow the existing Modal pipeline conventions.

---

## CONTEXT

We have completed speaker clustering and gender classification on the full
16,980-clip Shona (sna) speech dataset. The outputs are sitting on the Modal
volume `sna-data-vol` and are ready for the next two processing phases.

### Volume directory layout

| Path                                | Contents                                                               | Notes                                                       |
| ----------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| `/data/refined/`                    | Source dataset — flat HuggingFace Dataset, all shards in one directory | This is the canonical source. Do NOT overwrite or delete.   |
| `/data/final/`                      | Old published train/val/test splits — already pushed to HuggingFace    | DELETE this directory at the start of Phase A to free space |
| `/data/wav_cache/`                  | 16,980 WAV files at 24kHz mono, named `<source_id>.wav`                | Phase A reads audio from here, not from /data/refined       |
| `/data/relabel/relabel_mapping.csv` | source_id → cluster_id, gender, confidence                             | Join key for Phase A                                        |
| `/data/relabel/cluster_report.csv`  | Per-cluster summary                                                    | Used for Phase B speaker selection                          |
| `/data/reports/`                    | Existing audit reports                                                 | Append new reports here, do not delete existing             |

### IMPORTANT — /data/refined is a flat Dataset, not a DatasetDict

Load it with:

```python
from datasets import load_from_disk
ds = load_from_disk("/data/refined")
# ds is a datasets.Dataset directly — no splits, no concatenation needed
# ds.column_names includes: source_id, source_speaker_id, audio, transcription,
#   language, gender, has_punctuation, snr_db, speech_ratio, quality_score,
#   duration, speaker_idx, speaker_clip_count
```

Do NOT call `concatenate_datasets()` — there is only one dataset object here.

### relabel_mapping.csv columns

- `source_id` — join key back to original dataset
- `cluster_id` — new integer speaker identity (-1 = noise, drop these)
- `cluster_gender` — resolved gender per cluster (Female / Male / Unknown)
- `gender_predicted` — per-clip gender prediction
- `gender_confidence` — per-clip classifier confidence
- `cluster_confidence` — HDBSCAN assignment confidence
- `noise_rescued` — bool, whether clip was rescued from noise via centroid similarity
- `flag` — pipe-separated flags (NOISE, RESCUED, MANY_SOURCES etc.)

### Key numbers going into Phase A

- Total clips in /data/refined: 16,980
- Noise clips to drop (cluster_id == -1): 1,741
- Usable clips after noise removal: 15,239
- Unique speakers (clusters): 46
- Estimated hours after noise removal: ~83

---

## PHASE A — Rebuild, Annotate, Loudness Normalise, Publish

### Goal

Produce a clean, fully annotated HuggingFace dataset at
`manassehzw/sna-dataset-annotated` where:

- Speaker identities are acoustically derived (not the original contaminated labels)
- All clips are loudness-normalised to -23 LUFS (EBU R128)
- Schema is clean with no legacy contamination columns
- Dataset is saved to `/data/sna_annotated/` on the volume as a DatasetDict
  with train / validation / test splits

### Script to write

`src/rebuild_annotated.py` — single Modal function `rebuild_annotated()`

### App and volume

```python
app = modal.App("sna-rebuild-annotated")
data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)
```

### Image dependencies

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "datasets[audio]",
        "soundfile",
        "numpy",
        "pandas",
        "pyloudnorm",
        "huggingface_hub",
        "tqdm",
    )
)
```

### Modal function spec

```python
@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=7200,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_dotenv()],
)
```

### Step-by-step logic

**Step 0 — Clean up old directories**

```python
import shutil, os
# Delete old split directory — already published, no longer needed
if os.path.exists("/data/final"):
    shutil.rmtree("/data/final")
    print("Deleted /data/final")
```

**Step 1 — Load relabel mapping**

- Load `/data/relabel/relabel_mapping.csv` as pandas DataFrame
- Filter out all rows where `cluster_id == -1` (noise — drop permanently)
- Result: mapping_df with ~15,239 rows
- Log: how many noise clips were dropped

**Step 2 — Load original dataset**

- `ds = load_from_disk("/data/refined")` — loads as flat Dataset directly
- Convert to pandas DataFrame
- Keep only these columns from the original:
  `source_id, transcription, language, has_punctuation, snr_db, speech_ratio, quality_score, duration`
- Drop everything else:
  `source_speaker_id, speaker_idx, speaker_clip_count, gender, audio`
  (audio comes from wav_cache, identity columns are replaced by new labels)

**Step 3 — Join mapping onto original**

- Left join `mapping_df` onto `original_df` on `source_id`
- After join, rename columns:
  - `cluster_id` → `speaker_id`
  - `cluster_gender` → `gender`
- Keep `cluster_confidence` as-is
- Drop from mapping: `gender_predicted`, `gender_confidence`,
  `noise_rescued`, `flag`
- Result: clean_df with columns:
  `source_id, transcription, language, has_punctuation, snr_db,
speech_ratio, quality_score, duration, speaker_id, gender,
cluster_confidence`

**Step 4 — Recompute speaker_clip_count**

- Count clips per `speaker_id` in clean_df
- Map counts back as new column `speaker_clip_count`
- This reflects the true per-speaker count in the cleaned dataset, not the
  original contaminated counts

**Step 5 — Loudness normalisation**
For every clip:

- Load WAV from `/data/wav_cache/<source_id>.wav`
- Apply EBU R128 loudness normalisation to -23 LUFS
- Write normalised WAV to `/data/wav_normalised/<source_id>.wav`
- Do NOT overwrite the original `/data/wav_cache/` — keep it intact

```python
import pyloudnorm as pyln
import soundfile as sf
import numpy as np

TARGET_LUFS = -23.0

def normalise_loudness(arr: np.ndarray, sr: int) -> np.ndarray:
    meter = pyln.Meter(sr)  # BS.1770 meter
    loudness = meter.integrated_loudness(arr)
    # Guard against silence or clips too short to measure
    if not np.isfinite(loudness):
        return arr
    normalised = pyln.normalize.loudness(arr, loudness, TARGET_LUFS)
    # Hard clip to prevent any values exceeding [-1, 1] after normalisation
    return np.clip(normalised, -1.0, 1.0)
```

Track and log:

- `n_normalised` — clips that were adjusted
- `n_skipped` — clips already within ±1 LU of target (pass through as-is)
- `n_failed` — clips where loudness measurement failed (pass through as-is)

`data_vol.commit()` after all WAVs are written.

**Step 6 — Build HuggingFace Dataset object**
Construct a `datasets.Dataset` from a list of dicts. For each row in clean_df:

- Load normalised WAV from `/data/wav_normalised/<source_id>.wav`
- Read as float32 numpy array
- Include as `{"array": arr, "sampling_rate": 24000}` under `audio` key

Use this feature definition:

```python
from datasets import Dataset, Audio as AudioFeature, Features, Value

features = Features({
    "audio":              AudioFeature(sampling_rate=24000),
    "transcription":      Value("string"),
    "source_id":          Value("string"),
    "speaker_id":         Value("int32"),
    "speaker_clip_count": Value("int32"),
    "language":           Value("string"),
    "gender":             Value("string"),
    "has_punctuation":    Value("bool"),
    "snr_db":             Value("float32"),
    "speech_ratio":       Value("float32"),
    "quality_score":      Value("float32"),
    "duration":           Value("float32"),
    "cluster_confidence": Value("float32"),
})
```

Final column order must match the features dict above.

**Step 7 — Speaker-stratified train/val/test split**

- Ratio: 80 / 10 / 10
- Seed: 42
- Stratify by `speaker_id` so every speaker appears in all three splits
- Use the same per-speaker shuffle + slice logic as the original
  `src/split_and_upload.py`:

  ```python
  import random, numpy as np
  from collections import defaultdict

  SPLIT_SEED = 42
  TRAIN_RATIO, VALID_RATIO = 0.8, 0.1

  speaker_to_indices = defaultdict(list)
  for idx, spk in enumerate(dataset["speaker_id"]):
      speaker_to_indices[int(spk)].append(idx)

  rng = random.Random(SPLIT_SEED)
  train_idx, valid_idx, test_idx = [], [], []
  for _, idxs in speaker_to_indices.items():
      idxs = idxs.copy()
      rng.shuffle(idxs)
      n = len(idxs)
      n_train = int(np.floor(n * TRAIN_RATIO))
      n_valid = int(np.floor(n * VALID_RATIO))
      if n_train == 0 and n > 0:
          n_train = 1
      train_idx.extend(idxs[:n_train])
      valid_idx.extend(idxs[n_train:n_train + n_valid])
      test_idx.extend(idxs[n_train + n_valid:])
  ```

**Step 8 — Save to volume**

```python
from datasets import DatasetDict
dataset_dict = DatasetDict({
    "train":      dataset.select(sorted(train_idx)),
    "validation": dataset.select(sorted(valid_idx)),
    "test":       dataset.select(sorted(test_idx)),
})

# Safe save pattern — write to tmp, promote to final
tmp_path   = "/data/sna_annotated_tmp"
final_path = "/data/sna_annotated"

if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)

dataset_dict.save_to_disk(tmp_path)

if os.path.exists(final_path):
    shutil.rmtree(final_path)
shutil.move(tmp_path, final_path)

data_vol.commit()
```

**Step 9 — Push to HuggingFace**

```python
from huggingface_hub import HfApi, login
import os

hf_token    = os.environ["HF_TOKEN"]
hf_username = os.environ["HF_USERNAME"]
login(token=hf_token)

REPO_ID = f"{hf_username}/sna-dataset-annotated"
dataset_dict.push_to_hub(REPO_ID, token=hf_token)
```

Then upload README.md with the dataset card below using `api.upload_file()`.

**Dataset card content:**

```markdown
---
language:
  - sna
task_categories:
  - automatic-speech-recognition
  - text-to-speech
license: other
pretty_name: Shona Speech Dataset — Annotated (SNA)
tags:
  - shona
  - speech
  - asr
  - tts
---

# manassehzw/sna-dataset-annotated

## Summary

Annotated and loudness-normalised Shona (sna) speech corpus derived from
[manassehzw/sna-dataset](https://huggingface.co/datasets/manassehzw/sna-dataset),
itself adapted from [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP).

The original dataset contained widespread speaker ID contamination — different
speakers and genders incorrectly assigned to the same identity throughout.
This version replaces all original speaker labels with programmatically
derived identities using acoustic speaker embeddings.

## What changed from sna-dataset

- **Speaker IDs completely rederived** using ECAPA-TDNN embeddings +
  HDBSCAN clustering. Original source_speaker_id labels are dropped entirely.
- **Gender labels rederived** using a custom logistic regression classifier
  trained on ECAPA embeddings. Original gender column replaced.
- **Noise clips excluded** — 1,741 clips (10.2%) that could not be
  confidently assigned to any speaker cluster are excluded. They remain
  available in the source dataset.
- **Loudness normalised** to -23 LUFS (EBU R128) for consistency.

## Speaker derivation methodology

1. ECAPA-TDNN 192-dim embeddings extracted per clip (speechbrain/spkrec-ecapa-voxceleb)
2. Clips partitioned by predicted gender (Female / Male / Unknown)
3. HDBSCAN clustering within each gender partition
   (min_cluster_size=50, min_samples=10)
4. Noise rescue: unassigned clips within cosine similarity 0.75 of a
   cluster centroid reassigned to that cluster
5. Cluster gender resolved by majority vote (>=75% threshold)
6. 1,741 unassignable clips excluded

## Schema

| Column             | Type         | Description                                       |
| ------------------ | ------------ | ------------------------------------------------- |
| audio              | Audio(24kHz) | Loudness-normalised 24kHz mono WAV                |
| transcription      | string       | Normalised Shona transcription                    |
| source_id          | string       | Original clip ID — traceable to sna-dataset       |
| speaker_id         | int32        | Acoustically derived speaker identity (0-indexed) |
| speaker_clip_count | int32        | Total clips for this speaker in the dataset       |
| language           | string       | ISO code — always "sna"                           |
| gender             | string       | Programmatically derived: Female / Male / Unknown |
| has_punctuation    | bool         | Whether transcription contains punctuation        |
| snr_db             | float32      | Signal-to-noise ratio in dB                       |
| speech_ratio       | float32      | Proportion of clip containing speech              |
| quality_score      | float32      | Composite quality metric (SNR with penalties)     |
| duration           | float32      | Clip duration in seconds                          |
| cluster_confidence | float32      | HDBSCAN speaker assignment confidence (0-1)       |

## Notes

- speaker_id is acoustically derived, not identity-verified.
- 46 stable integer speaker IDs (0 = largest cluster by clip count).
- See manassehzw/sna-dataset for unfiltered version with original labels.
```

**Step 10 — Write audit report**
Save `/data/reports/rebuild_annotated_audit.json`:

```json
{
  "phase": "rebuild_annotated",
  "timestamp": "...",
  "source_path": "/data/refined",
  "input_clips": 16980,
  "noise_dropped": 1741,
  "final_clips": 15239,
  "unique_speakers": 46,
  "total_hours": "...",
  "loudness_target_lufs": -23.0,
  "clips_normalised": "...",
  "clips_skipped_loudness": "...",
  "clips_failed_loudness": "...",
  "splits": {
    "train": "...",
    "validation": "...",
    "test": "..."
  },
  "gender_distribution": {
    "Female": "...",
    "Male": "...",
    "Unknown": "..."
  },
  "hf_repo": "manassehzw/sna-dataset-annotated"
}
```

---

## PHASE B — TTS Subset (run after Phase A completes)

### Goal

Produce a high-quality TTS-optimised subset at `manassehzw/sna-tts` using
the top 15 speakers by clip count, with selective audio enhancement,
strict quality filtering, and output as metadata.json files compatible
with sesame-finetune pretokenize.py.

### Script to write

`src/build_tts_subset.py` — single Modal function `build_tts_subset()`

### Additional image dependencies (on top of Phase A deps)

```python
.uv_pip_install(
    "noisereduce",
    "speechbrain",
    "torch==2.4.0",
    "torchaudio==2.4.0",
)
```

### Top 15 speakers (from /data/relabel/cluster_report.csv)

| speaker_id | clips | gender | mean_cluster_confidence | note                                  |
| ---------- | ----- | ------ | ----------------------- | ------------------------------------- |
| 28         | 1,682 | Male   | 0.949                   | 2 source IDs — ear test recommended   |
| 29         | 1,490 | Male   | 0.862                   | clean single source                   |
| 0          | 1,381 | Female | 0.952                   | 4 source IDs — MANY_SOURCES, ear test |
| 1          | 867   | Female | 0.965                   | clean                                 |
| 2          | 708   | Female | 0.865                   | 4 source IDs — MANY_SOURCES, ear test |
| 32         | 700   | Male   | 0.838                   | 4 source IDs — MANY_SOURCES, ear test |
| 30         | 633   | Male   | 0.912                   | clean single source                   |
| 31         | 618   | Male   | 0.882                   | 2 source IDs                          |
| 3          | 461   | Female | 0.899                   | clean single source                   |
| 4          | 461   | Female | 0.921                   | clean single source                   |
| 6          | 442   | Female | 0.887                   | clean single source                   |
| 33         | 424   | Male   | 0.985                   | clean single source                   |
| 5          | 419   | Female | 0.905                   | clean single source                   |
| 34         | 335   | Male   | 0.892                   | clean single source                   |
| 7          | 332   | Female | 0.917                   | 2 source IDs                          |

**Estimated total: ~10,953 clips, ~60 hours, 8 Female + 7 Male speakers**

BEFORE RUNNING Phase B: ear test clusters 0, 2, 28, 32 by sampling
a few WAVs from `/data/wav_cache/` using the source_speaker_ids listed
in cluster_report.csv. If any cluster fails the ear test, replace it
with speaker_id 8 (322 clips, Female) or speaker_id 35 (281 clips, Male).

Define this as a constant at the top of the script so it is easy to edit:

```python
TTS_SPEAKER_IDS = [28, 29, 0, 1, 2, 32, 30, 31, 3, 4, 6, 33, 5, 34, 7]
```

### Step-by-step logic

**Step 1 — Load from Phase A output**

```python
from datasets import load_from_disk, concatenate_datasets
dataset_dict = load_from_disk("/data/sna_annotated")
ds = concatenate_datasets([dataset_dict[s] for s in dataset_dict.keys()])
```

Filter to TTS_SPEAKER_IDS only:

```python
ds = ds.filter(lambda x: x["speaker_id"] in TTS_SPEAKER_IDS)
```

**Step 2 — Quality filter**
Drop clips where `quality_score < 12.0`
This removes the bottom ~15% within the already-good top-15 speakers.
Log clips dropped per speaker.

**Step 3 — Duration filter**
Keep only `3.0 <= duration <= 20.0` seconds.
CSM-1B sweet spot. Log clips dropped.

**Step 4 — Selective noise reduction**
Apply ONLY to clips where `snr_db < 20.0`.
Do NOT apply to clips with snr_db >= 20 — risk of degrading clean audio.

```python
import noisereduce as nr

APPLY_NOISE_REDUCTION = True  # set False to disable entirely

def apply_noise_reduction(arr: np.ndarray, sr: int) -> np.ndarray:
    reduced = nr.reduce_noise(
        y=arr,
        sr=sr,
        stationary=True,
        prop_decrease=0.75
    )
    return reduced.astype(np.float32)
```

**Step 5 — Selective dereverberation**
Apply ONLY to clips where `speech_ratio < 0.65 AND snr_db < 18.0`.
Use speechbrain MetricGAN+ model.

```python
APPLY_DEREVERBERATION = True  # set False to disable entirely

from speechbrain.inference.enhancement import SpectralMaskEnhancement
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="/tmp/metricgan_cache",
    run_opts={"device": "cuda"},
)
```

Both enhancement flags should be at the top of the script for easy toggling.

**Step 6 — Re-normalise loudness**
After any audio modification, re-apply -23 LUFS normalisation.
Reuse the same `normalise_loudness()` function from Phase A.
Always normalise last — processing can shift loudness levels.

**Step 7 — Write WAVs and build metadata**
Write processed WAVs to `/data/sna_tts/wavs/<speaker_id>/<source_id>.wav`

Build two metadata JSON files compatible with sesame-finetune pretokenize.py.
Speaker-stratified 90/10 train/val split (seed=42):

```json
[
  {
    "text": "Motokari nhema yakamira...",
    "path": "/data/sna_tts/wavs/0/sna_22490.wav",
    "speaker": 0
  }
]
```

Write:

- `/data/sna_tts/metadata_train.json`
- `/data/sna_tts/metadata_val.json`

**Step 8 — Push to HuggingFace**
Repo: `manassehzw/sna-tts`
Push WAVs + metadata files.
Include dataset card documenting:

- Top 15 speaker selection criteria and IDs
- Quality threshold (12.0) and duration range (3-20s)
- Audio enhancements applied with per-clip counts
- Compatibility note for sesame-finetune pretokenize.py
- Link back to sna-dataset-annotated as source

**Step 9 — Write audit report**
Save `/data/reports/build_tts_subset_audit.json`:

```json
{
  "phase": "build_tts_subset",
  "timestamp": "...",
  "source_path": "/data/sna_annotated",
  "tts_speaker_ids": [...],
  "clips_before_quality_filter": "...",
  "clips_after_quality_filter": "...",
  "clips_after_duration_filter": "...",
  "clips_noise_reduced": "...",
  "clips_dereverbed": "...",
  "final_clips": "...",
  "total_hours": "...",
  "unique_speakers": 15,
  "gender_distribution": {"Female": "...", "Male": "..."},
  "metadata_train_clips": "...",
  "metadata_val_clips": "...",
  "hf_repo": "manassehzw/sna-tts"
}
```

---

## CONVENTIONS — follow existing pipeline style exactly

- Modal app name format: `sna-<phase-name>`
- Volume: `modal.Volume.from_name("sna-data-vol", create_if_missing=True)`
- Secrets: `modal.Secret.from_dotenv()` for HF_TOKEN and HF_USERNAME
- Image: `modal.Image.debian_slim(python_version="3.10")` with `.uv_pip_install()`
- Always call `data_vol.commit()` after any write to the volume
- Always write an audit JSON to `/data/reports/` before finishing
- Use `tqdm` for all loops over clips
- Print `=` bordered section headers matching existing pipeline style
- Use tmp/backup pattern for safe dataset saves (write to \_tmp, promote)
- `@app.local_entrypoint()` calling `<function>.remote()`

## VOLUME STATE SUMMARY

```
/data/refined/              ← SOURCE — do not touch
/data/final/                ← DELETE at start of Phase A
/data/wav_cache/            ← Phase A reads audio from here
/data/wav_normalised/       ← Phase A writes normalised WAVs here
/data/relabel/              ← mapping and cluster CSVs from clustering run
/data/sna_annotated/        ← Phase A output DatasetDict
/data/sna_tts/              ← Phase B output WAVs + metadata JSONs
/data/reports/              ← all audit JSONs accumulate here
```

## RUN ORDER

```bash
modal run src/rebuild_annotated.py    # Phase A
modal run src/build_tts_subset.py     # Phase B — only after Phase A completes
```

## DOWNLOAD OUTPUTS AFTER EACH PHASE

```bash
# After Phase A:
modal volume get sna-data-vol /data/reports/rebuild_annotated_audit.json .

# After Phase B:
modal volume get sna-data-vol /data/reports/build_tts_subset_audit.json .
modal volume get sna-data-vol /data/sna_tts/metadata_train.json .
modal volume get sna-data-vol /data/sna_tts/metadata_val.json .
```
