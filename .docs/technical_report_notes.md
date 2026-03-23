# Technical Report Notes: Data Processing Techniques

This document summarizes the data processing techniques already implemented in this project and maps them to sections I should include in my technical report.

## 1) Data Source and Ingestion

### Implemented techniques
- Pulled labeled `google/WaxalNLP` subset `sna_asr` (train/validation/test).
- Preserved source provenance by renaming:
  - `id` -> `source_id`
  - `speaker_id` -> `source_speaker_id`
- Concatenated splits into one processing view.
- Enforced hard quality floor at ingest (`duration >= 5s`).
- Created stable speaker indexing (`speaker_idx`) sorted by speaker frequency.

### What to include in report
- Why provenance fields are critical for reproducibility and relabeling.
- Justification for early duration floor.
- Speaker distribution before/after ingest filtering.

## 2) Metadata Normalization

### Implemented techniques
- Standardized `gender` values (`Male`/`Female`).
- Standardized `language` to lowercase.
- Computed `speaker_clip_count` for each speaker.

### What to include in report
- Why schema normalization is needed before model-facing datasets.
- Error reduction from controlled categorical values.

## 3) Text Normalization

### Implemented techniques
- Normalized punctuation variants (smart quotes, em/en dashes).
- Preserved casing while cleaning unsafe characters.
- Applied regex-based canonicalization for spacing and punctuation.
- Added derived feature `has_punctuation`.

### What to include in report
- Text cleaning rules and rationale.
- Before/after examples.
- Trade-off: preserving linguistic structure vs over-normalization.

## 4) Audio Processing and Quality Metrics

### Implemented techniques
- Resampled to 24kHz mono.
- Applied VAD-based speech segmentation (WebRTC VAD).
- Trimmed leading/trailing silence with buffering.
- Reduced large internal pauses via bounded gap compression.
- Computed quality features:
  - `snr_db`
  - `speech_ratio`
  - `quality_score`
  - updated `duration`
- Hard-dropped clips with no speech or empty post-trim waveform.

### What to include in report
- End-to-end audio normalization flow diagram.
- Why VAD was selected.
- Quality-score design and limitations.
- Failure cases and how they were handled.

## 5) Dataset Cleanup Strategy

### Implemented techniques
- Removed short clips after normalization (`duration < 5s`).
- Removed singleton speakers for downstream stability.
- Refreshed `speaker_clip_count` after drops.
- Applied explicit speaker blacklist when justified.

### What to include in report
- Why singleton removal matters for speaker modeling.
- Blacklist governance: evidence-based, auditable decisions.
- Net retention impact.

## 6) Split and Publish Workflow

### Implemented techniques
- Created train/validation/test split from processed dataset.
- Published processed dataset to Hugging Face (`manassehzw/sna-dataset`).
- Maintained report artifacts and audit logs for each phase.

### What to include in report
- Release process and reproducibility guarantees.
- Dataset card/documentation strategy.
- Versioning and rollback considerations.

## 7) Speaker Quality and Identity Audit Techniques

### Implemented techniques
- Ranked speakers by cumulative effective speech duration.
- Pulled stratified speaker samples (low/mid/high quality bins).
- Generated per-speaker metadata and review sheets.
- Built robust archive workflows:
  - zip integrity checks
  - corruption handling
  - salvage path for partially corrupt artifacts

### What to include in report
- Why stratified sampling is better than random-only sampling.
- Operational lessons from remote artifact generation and transfer.
- Human audit constraints and scaling limitations.

## 8) Automated Speaker Relabeling — Clustering Audit (v2)

### Implemented techniques
- Built local clustering audit pipeline (`audit_speaker_clusters.py`) using:
  - ECAPA-TDNN speaker embeddings (SpeechBrain, 192-d)
  - HDBSCAN density clustering (`min_cluster_size=15`, `min_samples=5`, euclidean on L2-normalised embeddings)
  - Noise rescue: cosine similarity reassignment of outlier points to nearest cluster centroid (threshold 0.75)
  - Gender classification per clip via Wav2Vec2 model (`prithivMLmods/Common-Voice-Gender-Detection`)
  - Majority-vote gender resolution per cluster (threshold 0.75 for confident assignment)
- v2 run results on 777 sample clips (39 speakers):
  - 31 raw clusters, 4.6% noise after rescue
  - 4 MIXED_GENDER clusters (13% of clusters) — identified as a problem requiring investigation
- Produced outputs per run:
  - `cluster_report.csv` — per-cluster summary with gender votes, flags, source speaker IDs
  - `clip_assignments.csv` — per-clip cluster assignment with predicted gender
  - `cluster_summary.txt` — human-readable audit summary
  - `params_used.json` — full parameter record for reproducibility

### What to include in report
- Motivation: label consistency over perfect identity truth.
- Clustering outcome summary (clusters vs noise rate vs mixed-gender rate).
- Noise rescue rationale: conservative reassignment preserves more clips without forcing wrong cluster membership.
- Risks: over-merging (different speakers grouped) vs over-fragmentation (one speaker split across clusters).
- Why HDBSCAN was chosen over k-means: no fixed cluster count required, density-adaptive, explicit noise class.

## 9) Gender Classification — Source Label Contamination and Custom Classifier

### Problem discovered
The v2 audit produced 4 MIXED_GENDER clusters. Investigation revealed two compounding issues:
1. **Source label contamination**: gender labels in the WaxalNLP source dataset are unreliable. Multiple speakers are verifiably mislabelled (e.g., a speaker labelled Female whose ECAPA embedding and audio clearly indicate Male).
2. **Gender model distribution mismatch**: the off-the-shelf model (`prithivMLmods/Common-Voice-Gender-Detection`, a Wav2Vec2 fine-tune on Common Voice) was trained predominantly on English and European speech. It produced confident wrong predictions on Shona speakers — in one case calling 11 out of 18 clips from a known female speaker Male (61% wrong). This is not model error per se; it is a training distribution mismatch.

### Solution: Shona-calibrated logistic regression on ECAPA embeddings
Rather than substituting another off-the-shelf model, a lightweight custom classifier was built using the ECAPA embeddings already computed for clustering.

**Key insight**: ECAPA-TDNN (trained on VoxCeleb — millions of utterances, thousands of speakers, multilingual) encodes gender as one of the dominant axes of variation in its 192-d embedding space. Male and female voices form linearly separable regions independent of language. A logistic regression fit on verified Shona clips finds the boundary in that space and generalises because the separation is geometric, not linguistic.

**Implementation** (`train_gender_classifier.py`):
- Training data bootstrapped from clean v2 clusters: clusters with a single source speaker and ≥ 90% unanimous gender votes from the Wav2Vec2 model (before it was deprecated in the pipeline)
- Initial training set: 307 clips, 8 speakers (156 female, 151 male)
- Model: `sklearn.linear_model.LogisticRegression` on L2-normalised 192-d ECAPA vectors
- 5-fold stratified cross-validation accuracy: **100%** across all folds
- Saved as `gender_classifier_ecapa.pkl`; loaded automatically by the audit pipeline

**Active learning refinement loop** (`probe_gender_classifier.py`):
- Probe script runs the classifier over all 777 speaker sample clips
- Outputs clips into `female/`, `male/`, `unknown/` folders with speaker-prefixed filenames for ear-testing
- Cross-references previously Mixed clusters from v2 to show prediction changes directly
- Initial probe: 86 Unknown clips (11.1%) across 8 HIGH_UNKNOWN speakers
- Ear-tested all 86 Unknown clips; 84 were clearly identifiable as male or female by listening; 2 were genuinely ambiguous even by ear and excluded from training
- Added 84 ear-tested clips to training data; retrained classifier
- Post-retrain probe: **73 Unknown clips (9.4%)**, 6 HIGH_UNKNOWN speakers
- Specific improvement: speakers with previously high Unknown rates (e.g., speaker 08: 5U → 0U; speaker 19: 10U → 5U)
- Critical fix: the female speakers that Wav2Vec2 called Male now have **zero Male confident predictions**

**Comparison vs Wav2Vec2 gender model on known problem cases**:

| Cluster | Speaker truth | Wav2Vec2 (wrong) | New classifier |
|---|---|---|---|
| cluster_29_Mixed | Female | 11/18 Male (61%) | 0/18 Male |
| cluster_19_Mixed | Female | 6/19 Male (32%) | 0/19 Male |

### Integration into audit pipeline
- `audit_speaker_clusters.py` checks for `gender_classifier_ecapa.pkl` at startup
- If found: uses logistic regression on ECAPA embeddings; Wav2Vec2 model is not loaded (faster, more accurate)
- If not found: falls back to Wav2Vec2 behavior
- `params_used.json` records which classifier was used for reproducibility

### What to include in report
- Distribution mismatch as a systematic failure mode for off-the-shelf models on low-resource languages.
- The ECAPA embedding space gender separation property and why it generalises across languages.
- Active learning loop design: probe → ear-test → retrain → re-probe, and diminishing returns analysis.
- Comparison table: Wav2Vec2 vs logistic regression on known-label clips.
- Limitations: 9.4% Unknown rate expected at scale; will be slightly higher on unseen speakers.
- Why "Unknown" is the correct output for genuinely ambiguous voices rather than a forced label.
- This classifier as a publishable artifact: first Shona-calibrated gender classifier, trained on speaker embeddings.

## 10) Annotated Rebuild and Loudness Standardization

### Implemented techniques
- Added `rebuild_annotated.py` as a dedicated rebuild phase after speaker relabeling.
- Dropped all noise-assigned clips (`cluster_id == -1`) before expensive audio work.
- Rebuilt schema around relabel outputs:
  - `cluster_id` -> `speaker_id`
  - `cluster_gender` -> `gender`
  - `cluster_confidence` -> `speaker_assignment_confidence`
- Removed legacy identity fields from the annotated release (`source_speaker_id`, `speaker_idx`) and per-clip helper columns not needed in final schema.
- Recomputed `speaker_clip_count` from post-noise-cleaned relabelled speakers.
- Applied EBU R128 loudness normalization to **-23 LUFS** over `/data/wav_cache` into `/data/wav_normalised`.
- Used skip tolerance (`+/-1 LU`) to avoid unnecessary processing for already-stable clips.
- Tracked clipping pressure via `clips_peak_hit_0dbfs` audit counter.
- Saved the output as DatasetDict to `/data/sna_annotated` using tmp -> promote for atomic replacement.

### Report values to include (actual run)
- Input clips: `16,980`
- Noise dropped: `1,741`
- Final clips: `15,239`
- Unique speakers: `46`
- Total hours: `78.5`
- Loudness input mean/std: `-22.659 / 5.301 LUFS`
- Loudness output mean/std: `-22.999 / 0.243 LUFS`
- Clips normalized/skipped/failed: `12,458 / 2,781 / 0`

## 11) Annotated Dataset Publication

### Implemented techniques
- Decoupled rebuild from publish by introducing `upload_annotated.py`.
- Loaded `/data/sna_annotated` and pushed to Hugging Face as `manassehzw/sna-dataset-annotated`.
- Uploaded a method-rich dataset card documenting:
  - source contamination problem,
  - relabeling method,
  - gender classifier rationale,
  - loudness normalization policy,
  - schema and usage constraints.
- Wrote dedicated upload audit artifact (`upload_annotated_audit.json`).

### What to include in report
- Why release decoupling (rebuild vs upload) improves operational safety.
- How dataset cards act as reproducibility and governance artifacts.
- Versioning narrative: `sna-dataset` -> `sna-dataset-annotated`.

## 12) Local Validation Harness for Audio Decisions

### Implemented techniques
- Added local normalization sandbox under `src/tests/audio/normalization/`.
- `normalize_volume.py`: controlled input/output folder A/B testing before full-volume execution.
- `mic_pop_audit.py`: startup transient detection audit for clip-start pops.
- Used these local scripts to validate policy decisions before running full Modal phases.

### What to include in report
- Why small-sample local A/B validation reduces costly remote reruns.
- Decision outcome: global de-click/de-reverb not applied due to low prevalence and artifact risk.

## 13) Techniques Explored and Rejected

### Implemented techniques
- Tried heuristic artifact detection (kurtosis + high-frequency energy).
- Documented high false-positive behavior and dropped the method.
- Tried off-the-shelf gender classification (`prithivMLmods/Common-Voice-Gender-Detection`) and replaced with Shona-calibrated logistic regression after confirming distribution mismatch.
- Considered global enhancement (noise reduction/de-reverb) for full dataset and deferred it to task-specific TTS phase.

### What to include in report
- Why heuristic artifact detection failed (Shona plosives vs click signatures).
- Why off-the-shelf gender models can fail on low-resource languages.
- Why non-opinionated processing is preferable for a general-purpose release.
- Importance of documenting negative findings.

## 14) Suggested Technical Report Structure

1. Problem statement and dataset goals
2. System architecture (Modal + volume + phase pipeline)
3. Data schema and provenance strategy
4. Text and metadata normalization techniques
5. Audio normalization and quality scoring
6. Cleanup rules and retention analysis
7. Speaker contamination challenge and manual audit findings
8. Clustering-based speaker relabeling approach
9. Gender classification: contamination, distribution mismatch, and custom classifier
10. Annotated rebuild and loudness standardization
11. Publication pipeline and dataset-card governance
12. Evaluation, limitations, and future work

## 15) Metrics I Should Report

- Raw clips vs retained clips per phase
- Total hours retained per phase
- Speaker count and speaker-clip distribution per phase
- Quality metric distributions (`snr_db`, `speech_ratio`, `quality_score`)
- Clustering metrics:
  - number of clusters
  - noise percentage before and after rescue
  - rescued clip count
  - mixed-source and mixed-gender flags per run
- Gender classifier metrics:
  - training set size and composition (clips per gender, speakers per gender)
  - 5-fold CV accuracy
  - per-speaker Unknown rate before and after active learning loop
  - comparison table: Wav2Vec2 vs logistic regression on known-label speakers
- Rebuild/loudness metrics:
  - LUFS input vs output distribution (mean/std/min/max)
  - normalized vs skipped vs failed counts
  - clips hitting 0 dBFS ceiling after normalization
- Release metrics:
  - final clip count / total hours / unique speakers
  - split sizes
  - schema diff between `sna-dataset` and `sna-dataset-annotated`
