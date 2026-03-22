# Pre- and post-classification audits (for reporting)

This document describes what to record **before** running the full speaker/gender classification pipeline, what to capture **after**, and which artifacts you need on disk so you can build figures (bar charts, UMAP, heatmaps) for a capstone report. No implementation details — only what to measure and what to save.

---

## Purpose

- **Pre-audit:** Establish a baseline of the *published* dataset state: source speaker IDs, source gender labels, clip counts, and basic demographics. This quantifies “what WaxalNLP gave us” before any relabeling.
- **Post-audit:** Record the same dimensions *after* clustering + your gender classifier, plus clustering-specific outputs. This supports **before/after** comparisons and contamination analysis in your report.
- **Shared artifacts:** Some outputs are needed for geometric plots (UMAP) and quantitative plots (heatmaps). If you do not persist them, you cannot reproduce figures without re-running heavy steps.

---

## 1. Pre-classification audit (baseline)

Run this **before** `classify_speakers` (or equivalent Modal job) changes any speaker or gender assignment used downstream.

### 1.1 What to measure (tabular)

For **every clip** in scope (concatenated train + validation + test from `/data/refined/`, or the same rows your classification script will process):

| Field | Purpose |
|-------|---------|
| `source_id` | Stable key for joins |
| `source_speaker_id` | Original WaxalNLP speaker hash |
| `gender` (as in dataset) | Original label after `annotate_metadata` normalization |
| `split` | train / validation / test |
| `duration` (or `duration_s`) | Optional: distribution plots |
| `quality_score`, `snr_db` | Optional: correlate with noise / unknown rate later |

### 1.2 Aggregates to log (for bar charts / one-pagers)

- **Clip count** total and per split.
- **Speaker count** (unique `source_speaker_id`).
- **Clips per speaker:** min, max, median, histogram or top-N table (long-tail is a story for TTS/ASR).
- **Gender distribution** (counts and %) using **source** `gender` column only — Male / Female / missing or invalid if any.
- **Hours of audio** (sum of `duration`) overall and optionally per gender / per split.

### 1.3 Audit artifact (machine-readable)

Write one JSON (or CSV summary + JSON) under `/data/reports/` with a clear name, e.g. `pre_classification_baseline.json`, including:

- Timestamp, dataset path, row count.
- All aggregates above.
- Optional: path to a **per-clip CSV** export if you want to re-plot without re-querying the dataset (`pre_classification_clips.csv` with the columns in §1.1).

This baseline is what you compare to post-classification gender and “effective speaker” counts.

---

## 2. Data you must persist for geometric / rigorous figures

These are **not** optional if you want UMAP and cosine-similarity heatmaps without re-running ECAPA on tens of thousands of clips.

### 2.1 Per-clip ECAPA embeddings

- **Shape:** one row per clip, 192-d float32 (L2-normalised as used for clustering).
- **Aligned with:** same row order as `source_id` list, or store `source_id` alongside each row.
- **Format:** `.npy` + sidecar `source_ids.json` / `.csv`, or a single `.npz` with keys `embeddings`, `source_ids`.

Used for: UMAP (2D), optional PCA preview, and any “same projection, two colourings” figure.

### 2.2 Per-clip labels (two layers)

Store **both** for every clip:

| Layer | Field | Used for |
|-------|--------|----------|
| Baseline | `source_speaker_id` | “Before” colouring (messy overlap = contamination story) |
| Post | `cluster_id` (or canonical speaker id) | “After” colouring (tight groups) |
| Post | `gender_predicted` + `gender_confidence` | Gender-separated analysis, Unknown rate |

Optional but valuable: `noise_rescued`, `hdbscan_label` before rescue.

### 2.3 Cluster centroids (for heatmaps)

After clustering (and optional noise rescue), save **L2-normalised centroid vectors** per final cluster (and optionally per gender partition if you use gender-separated HDBSCAN):

- **Shape:** `K × 192` for `K` clusters (+ row id → stable cluster id).
- Used for: **cosine similarity matrix between centroids** (K×K). Much more readable than a clip×clip matrix.

---

## 3. Post-classification audit

Run immediately after the classification job finishes successfully.

### 3.1 Per-clip export (for analysis and plots)

Minimum columns:

- `source_id`, `source_speaker_id` (unchanged provenance)
- `gender_predicted`, `gender_confidence`, optional `gender_original` for comparison
- `cluster_id` (or relabelled canonical id), `noise` / `noise_rescued` flags
- `split` (if you keep splits in the export)

### 3.2 Aggregates (mirror pre-audit)

- Total clips, speaker count **by source_speaker_id** (unchanged).
- **New:** cluster count, clips per cluster (distribution), noise clip count and %.
- **Gender:** distribution of `gender_predicted` (Female / Male / Unknown) — compare to pre-audit source gender counts.
- Optional: Unknown rate per split, per original speaker, or per cluster.

### 3.3 Contamination / relabelling story (high level)

- **Speaker–cluster confusion:** how many distinct `source_speaker_id` values appear in each cluster (histogram or table of “clusters with >1 source speaker”).
- **Cross-gender within cluster (if any):** should be **zero** or near-zero if you use **gender-first then cluster within gender**; if you still run unified HDBSCAN, report count of clusters with mixed predicted genders above a threshold.
- **Before/after tables:** e.g. for each original `source_speaker_id`, number of distinct `cluster_id` assignments (split speakers) vs number of other source speakers merged into same cluster.

### 3.4 Audit artifact

Write `/data/reports/post_classification_summary.json` (and optionally `post_classification_clips.csv`) with the same structure spirit as pre-audit so diffing is trivial.

---

## 4. Recommended figures and required inputs

| Figure | What it shows | Data you need saved |
|--------|----------------|---------------------|
| **UMAP 2D — coloured by cluster** | Tightness / separation of discovered groups | Embeddings + `cluster_id` |
| **UMAP 2D — coloured by source_speaker_id** | Overlap / fragmentation of original labels | Embeddings + `source_speaker_id` (same UMAP coords as below) |
| **Before / after (side by side)** | Contamination vs cleaned grouping | **One UMAP fit** on embeddings; two plots with **identical** 2D coordinates, different colours |
| **Cluster size bar chart** | Long tail, threshold justification | Table of `cluster_id` → count |
| **Cosine similarity heatmap (centroids)** | Inter- vs intra-cluster similarity (rigorous) | Centroid matrix `K×192` or precomputed `K×K` cosine matrix |
| **Gender UMAP** (optional) | Gender structure in embedding space | Embeddings + `gender_predicted` |

**Important:** For the before/after UMAP pair, fit UMAP **once** on the same embedding matrix, then only change the colour column. Otherwise the two panels are not comparable.

---

## 5. Optional extras (nice for a capstone)

- **Sankey or flow diagram** (source_speaker_id → cluster_id): needs an edge list or aggregated flow table (counts per source→cluster).
- **Per-speaker metrics CSV:** `source_speaker_id`, `n_clips`, `n_distinct_clusters`, `dominant_cluster_fraction` — supports tables and ranked “most contaminated” speakers.
- **Runtime and compute:** wall time, GPU type, batch size — one paragraph in methodology.

---

## 6. Checklist summary

**Before classification**

- [ ] Baseline JSON + optional per-clip CSV with source ids, source gender, split, duration/quality if needed.
- [ ] Logged aggregates: speakers, clips, gender %, hours.

**After classification**

- [ ] Post summary JSON + per-clip CSV with cluster + predicted gender + flags.
- [ ] Saved **embeddings** + **source_ids** (aligned).
- [ ] Saved **cluster centroids** (or full cosine matrix between centroids).
- [ ] Same UMAP projection reused for “before” (source speaker) and “after” (cluster) figures.

This is enough to support bar charts, UMAP story figures, and centroid heatmaps for the dissertation without prescribing any particular plotting library or script.
