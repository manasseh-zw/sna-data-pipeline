# Phase: Full-Dataset Speaker Classification and Relabeling

This document is the implementation plan for the next agent or session. It covers everything needed to run the speaker clustering and gender classification pipeline over the full 16,980-clip dataset on Modal and produce the relabel mapping for `sna-dataset-labeled`.

---

## Goal

Produce a deterministic, traceable mapping:

```
source_id  →  cluster_id  +  gender_predicted  +  gender_confidence  +  cluster_confidence
```

This mapping is the core artifact. It will be applied on top of `/data/refined/` to produce `sna-dataset-labeled` without reprocessing audio. Every clip gets an assignment — either a cluster ID or NOISE — and a confidence score. Low-confidence assignments are flagged, not dropped.

---

## What already exists (do not redo)

- `/data/refined/` — full cleaned dataset (HuggingFace DatasetDict, train/validation/test splits, 24kHz audio as float32 arrays)
- `src/tests/audio/audit_speaker/audit_speaker_clusters.py` — validated local clustering script (reference implementation)
- `src/tests/audio/audit_speaker/train_gender_classifier.py` — trains logistic regression gender classifier
- `src/tests/audio/audit_speaker/probe_gender_classifier.py` — validates classifier at scale
- `src/tests/audio/audit_speaker/gender_classifier_ecapa.pkl` — trained Shona-calibrated gender classifier (local)
- `src/tests/audio/audit_speaker/gender_training_data/` — ear-tested labeled WAVs used for training

The local audit scripts have been validated on 777 clips across 39 speakers. The `.pkl` classifier achieves 100% CV accuracy and 9.4% Unknown rate on the sample set. It supersedes the Wav2Vec2 gender model.

---

## Prerequisites before writing the Modal script

### 1. Upload the gender classifier to the Modal volume

The `.pkl` file must be accessible inside the Modal container. Upload it once:

```bash
modal volume put sna-data-vol \
  src/tests/audio/audit_speaker/gender_classifier_ecapa.pkl \
  /models/gender_classifier_ecapa.pkl
```

Verify with:
```bash
modal volume ls sna-data-vol /models/
```

### 2. Confirm `/data/refined/` is accessible

The dataset should already be there from Phase 4/5. Verify the split sizes match expectations (train ~14.1k, validation ~1.73k, test ~1.75k, total ~16,980 after ingest filtering).

---

## New volume paths (add to context)

```
/data/wav_cache/     — extracted WAV files, one per clip, named {source_id}.wav at 24kHz
                       needed by ECAPA (resampled in-memory to 16kHz) and by later
                       audio normalization passes
/data/models/        — model artifacts uploaded from local
                       gender_classifier_ecapa.pkl lives here
/data/reports/       — existing audit JSON folder; add speaker_relabel_audit.json here
/data/relabel/       — output of this phase
                       relabel_mapping.csv    — source_id → cluster assignment
                       cluster_report.csv     — per-cluster summary
                       params_used.json       — full run parameters for reproducibility
```

---

## New script

**Location:** `src/classify_speakers.py`

This is a Modal app, following the same pattern as all other pipeline scripts. It is not a test script — it is a production pipeline phase.

### Script responsibilities (in order)

1. **Load dataset** — load `/data/refined/` as a HuggingFace DatasetDict, concatenate train + validation + test into a flat list of records.

2. **Extract WAVs** — write each clip to `/data/wav_cache/{source_id}.wav` at 24kHz (the array is already 24kHz in the dataset). Skip if file already exists (idempotent). This serves two purposes: ECAPA can read from disk in parallel workers, and the files are available for the later audio normalization pass without re-extracting from the dataset.

3. **Load ECAPA-TDNN** — `speechbrain/spkrec-ecapa-voxceleb`, savedir `/tmp/ecapa_cache` (Modal will re-download if not warm, which is acceptable — it is a small model). Run on GPU.

4. **Load gender classifier** — unpickle `/data/models/gender_classifier_ecapa.pkl`. This is the logistic regression trained on Shona speaker samples.

5. **Batch ECAPA inference** — process clips in batches of 64. Pad clips in each batch to the length of the longest clip in that batch (SpeechBrain `encode_batch` requires uniform length within a batch). Collect: `source_id`, 192-d L2-normalised embedding, gender prediction + confidence from logistic regression.

6. **HDBSCAN clustering** — run on the full embedding matrix. Parameters to start with (tune if needed based on noise rate):
   - `min_cluster_size=50` (scaled up from 15 for local sample; adjust based on results)
   - `min_samples=10`
   - `metric="euclidean"` on L2-normalised embeddings
   - `cluster_selection_method="eom"`

7. **Noise rescue** — same logic as local script: compute cluster centroids, reassign noise points within cosine similarity threshold 0.75 to nearest centroid.

8. **Gender resolution per cluster** — majority vote with 0.75 threshold. Clusters below threshold are Mixed (flag, do not discard).

9. **Write relabel mapping** — one row per clip:
   ```
   source_id, source_speaker_id, cluster_id, cluster_gender,
   gender_predicted, gender_confidence, noise_rescued,
   cluster_size, unique_source_speakers_in_cluster
   ```
   Noise points (unrescued): `cluster_id = -1`.

10. **Write audit JSON** — full run summary to `/data/reports/speaker_relabel_audit.json`. Include: total clips, cluster count, noise rate, MIXED_GENDER count, gender distribution, HDBSCAN params, classifier pkl path and metadata, runtime.

---

## Compute specification

**GPU:** T4 (`gpu="T4"` in Modal decorator)
- ECAPA is convolutional — scales well with batch size on GPU
- Batch of 64 clips (avg ~8s, 16kHz) ≈ 82MB VRAM; T4 has 16GB — very comfortable
- Expected throughput: ~50 clips/second with batch_size=64 on T4
- Expected total ECAPA runtime: ~6 minutes for 16,980 clips

**CPU RAM:** 32GB (`memory=32768`)
- WAV extraction is the bottleneck for RAM: 16,980 clips × avg ~8s × 24kHz × 4 bytes (float32) ≈ 13GB if all loaded at once. With streaming batch extraction this is not an issue. 32GB gives comfortable headroom.
- HDBSCAN on 16,980 × 192-d float32 matrix: ~13MB matrix, trivial.

**Timeout:** Set to 30 minutes to be safe (`timeout=1800`).

**CPU vs T4 decision:** CPU is not viable. Without batched GPU inference, ECAPA at ~1s/clip = ~4.7 hours for 16,980 clips. T4 with batch_size=64 brings this to ~6 minutes. Use T4.

---

## HDBSCAN parameter guidance for full dataset scale

The local sample (777 clips, 39 speakers) used `min_cluster_size=15`. At full scale (16,980 clips, 168 speakers):

- `min_cluster_size=50` is a reasonable starting point (scales ~proportionally with dataset size)
- If noise > 15%: lower `min_cluster_size` to 30
- If clusters contain many source speakers: raise `min_samples` to 15
- Expected clusters at full scale: 150–300 (based on 31 clusters for 39 speakers in the sample)
- The extrapolation from v2 estimated ~677 clusters — this was based on raw clusters before merging; expect fewer after tuning

---

## Gender-separated clustering (recommended improvement over v2)

Rather than running HDBSCAN on all embeddings and checking gender post-hoc, run it within each gender partition:

1. Classify gender for all 16,980 clips → Female / Male / Unknown
2. Partition embedding matrix: Female embeddings, Male embeddings, Unknown embeddings
3. Run HDBSCAN independently on Female partition and Male partition
4. Offset cluster IDs so Female clusters = 0..N_f, Male clusters = N_f+1..N_f+N_m
5. For Unknown clips: post-hoc nearest centroid assignment across both cluster sets (same cosine sim threshold as noise rescue)

This eliminates MIXED_GENDER clusters by construction. The local v2 run had 4 MIXED_GENDER clusters out of 31 (13%); gender-separated clustering should bring this to zero for confident-gender clips.

---

## Relabel mapping output schema

`/data/relabel/relabel_mapping.csv`:

| Column | Type | Description |
|---|---|---|
| `source_id` | str | Original clip ID from WaxalNLP, never modified |
| `source_speaker_id` | str | Original speaker hash from WaxalNLP, never modified |
| `cluster_id` | int | HDBSCAN cluster assignment (-1 = unrescued noise) |
| `cluster_gender` | str | Female / Male / Mixed / Unknown |
| `gender_predicted` | str | Per-clip: Female / Male / Unknown |
| `gender_confidence` | float | Max probability from logistic regression (0.0–1.0) |
| `noise_rescued` | bool | True if clip was noise-rescued to a cluster |
| `cluster_size` | int | Number of clips in assigned cluster |
| `unique_source_speakers` | int | Number of distinct source_speaker_ids in cluster |
| `flag` | str | Pipe-separated flags: NOISE, MIXED_GENDER, MANY_SOURCES, RESCUED |

---

## After this phase completes

1. **Inspect the relabel mapping** locally — check cluster sizes, noise rate, MIXED_GENDER count.
2. **Apply mapping to `/data/refined/`** — join on `source_id`, add `cluster_id` as the new `speaker_idx` candidate, keep `source_speaker_id` untouched. Publish as `sna-dataset-labeled`.
3. **Validate gender classifier on full 168-speaker set** — run probe script against full dataset speaker samples. If Unknown rate holds below 12%, publish classifier as `{HF_USERNAME}/sna-gender-shona` on HuggingFace.

---

## Key design constraints (carry forward from pipeline)

- Never overwrite `source_id` or `source_speaker_id` — these are immutable provenance fields.
- Every script reads from one path and writes to one path. Never read and write the same path without temp → rename.
- Write a numbered audit JSON to `/data/reports/` for every run.
- All scripts run from the **repo root**.
- Noise points get `cluster_id = -1`. Do not force-assign low-confidence clips. Flag, keep, let consumers filter.
