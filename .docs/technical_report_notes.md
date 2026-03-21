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

## 8) Automated Speaker Relabeling Direction (Current Priority)

### Implemented techniques
- Built local clustering audit pipeline (`audit_speaker`) using:
  - ECAPA embeddings (SpeechBrain)
  - HDBSCAN clustering
- Produced outputs:
  - `cluster_report.csv`
  - `clip_assignments.csv`
  - `cluster_summary.txt`
  - `params_used.json`
- Added progress instrumentation and metadata-safe parsing.

### What to include in report
- Motivation: label consistency over perfect identity truth.
- Clustering outcome summary (e.g., clusters vs noise rate).
- Noise handling policy (conservative assignment vs forced relabel).
- Risks: over-merging vs over-fragmentation.

## 9) Techniques Explored and Rejected

### Implemented techniques
- Tried heuristic artifact detection (kurtosis + high-frequency energy).
- Documented high false-positive behavior and dropped the method.

### What to include in report
- Why this failed on this dataset.
- Importance of reporting negative results in engineering research.

## 10) Suggested Technical Report Structure

1. Problem statement and dataset goals  
2. System architecture (Modal + volume + phase pipeline)  
3. Data schema and provenance strategy  
4. Text and metadata normalization techniques  
5. Audio normalization and quality scoring  
6. Cleanup rules and retention analysis  
7. Publication pipeline and reproducibility  
8. Speaker contamination challenge  
9. Clustering-based relabeling approach (`sna-dataset-labeled`)  
10. Evaluation, limitations, and future work  

## 11) Metrics I Should Report

- Raw clips vs retained clips per phase
- Total hours retained per phase
- Speaker count and speaker-clip distribution per phase
- Quality metric distributions (`snr_db`, `speech_ratio`, `quality_score`)
- Clustering metrics:
  - number of clusters
  - noise percentage
  - mixed-source and mixed-gender flags
- Final dataset release summary and intended use constraints

