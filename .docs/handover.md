# SNA Dataset — Current Handover (Post-Annotated Release)

# Agent: read this fully before writing any code.

# All scripts go in src/ and follow existing Modal pipeline conventions.

---

## CONTEXT

The annotated rebuild and publishing phase is complete.

Published datasets:

- `manassehzw/sna-dataset` (older cleaned release)
- `manassehzw/sna-dataset-annotated` (new speaker-relabelled release)

The active workstream is now **TTS subset planning and build** from the annotated dataset.

---

## CURRENT VOLUME STATE

| Path                                | Contents                                                              | Notes                                                           |
| ----------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------- |
| `/data/refined/`                    | Previous canonical flat dataset                                       | Keep for provenance                                              |
| `/data/relabel/relabel_mapping.csv` | source_id -> cluster_id + confidence + gender fields                 | Produced by classify phase                                       |
| `/data/relabel/cluster_report.csv`  | Per-cluster summary                                                   | Primary input for TTS speaker candidate analysis                |
| `/data/wav_cache/`                  | 16,980 WAV files at 24kHz mono (`<source_id>.wav`)                   | Source WAVs for rebuild and downstream audio ops                |
| `/data/wav_normalised/`             | Loudness-normalised WAV files from rebuild                            | Built at -23 LUFS                                                |
| `/data/sna_annotated/`              | Final annotated DatasetDict (`train`/`validation`/`test`)             | Primary source for next TTS phase                               |
| `/data/reports/`                    | Audit JSON files from all phases                                      | Includes relabel + rebuild + upload audits                      |

---

## WHAT WAS COMPLETED

### 1) Speaker relabeling + clustering

`src/classify_speakers.py` completed and wrote:

- `/data/relabel/relabel_mapping.csv`
- `/data/relabel/cluster_report.csv`
- `/data/reports/speaker_relabel_audit.json`

Key relabel audit values:

- total clips: `16,980`
- clusters: `46`
- noise after rescue: `1,741` (`10.2532%`)
- rescued clips: `794`
- embedding failures: `0`

### 2) Annotated rebuild + loudness normalization

`src/rebuild_annotated.py` completed and wrote:

- `/data/wav_normalised/`
- `/data/sna_annotated/`
- `/data/reports/rebuild_annotated_audit.json`

Key rebuild audit values:

- input clips: `16,980`
- noise dropped: `1,741`
- final clips: `15,239`
- unique speakers: `46`
- total hours: `78.5`
- clips normalised: `12,458`
- clips skipped (already near target): `2,781`
- loudness failures: `0`
- output LUFS mean/std: `-22.999 / 0.243`

### 3) Annotated dataset upload

`src/upload_annotated.py` completed and pushed:

- Hugging Face dataset: `manassehzw/sna-dataset-annotated`
- README dataset card uploaded
- upload audit saved to `/data/reports/upload_annotated_audit.json`

---

## ANNOTATED DATASET SCHEMA (CURRENT)

`/data/sna_annotated` has:

- `audio` — 24kHz mono, LUFS-normalised float audio
- `transcription`
- `source_id`
- `speaker_id` — acoustically-derived integer speaker class
- `speaker_clip_count`
- `language`
- `gender` — cluster-level resolved (`Female` / `Male` / `Unknown`)
- `has_punctuation`
- `snr_db`
- `speech_ratio`
- `quality_score`
- `duration`
- `speaker_assignment_confidence`

---

## LOCAL TEST HARNESS ADDED

Under `src/tests/audio/normalization/`:

- `normalize_volume.py` (local LUFS testing)
- `mic_pop_audit.py` (clip-start mic-pop audit)
- `README.md` + input/output/report scaffolding

Mic-pop prevalence was assessed as low and no global de-click/de-reverb processing was added to the full dataset.

---

## DESIGN DECISIONS TO KEEP

- Keep general-purpose dataset opinionation low.
- Do not apply global enhancement (noise reduction/de-reverb) to full annotated release.
- Keep relabel confidence in schema as `speaker_assignment_confidence`.
- Preserve `/data/refined` for provenance; use `/data/sna_annotated` for new work.
- Continue writing audits to `/data/reports/` for every new phase.

---

## NEXT WORKSTREAM — TTS SUBSET (NOT IMPLEMENTED YET)

Immediate task is **analysis first**, then subset build:

1. Write TTS speaker-candidate analysis script over `/data/sna_annotated` and `/data/relabel/cluster_report.csv`.
2. Produce selection metrics and shortlist criteria (speaker size, quality, SNR, duration, confidence, ear-check notes).
3. Finalize speaker ID list for TTS curation.
4. Implement `src/build_tts_subset.py` from final criteria.

Do not hard-code old top-15 speaker list without re-validating against the current annotated output.

---

## RECOMMENDED RUN ORDER FOR NEXT AGENT

```bash
# 1) Analysis phase (to be written next)
modal run src/analyze_tts_candidates.py

# 2) TTS subset build (after analysis criteria is agreed)
modal run src/build_tts_subset.py
```
