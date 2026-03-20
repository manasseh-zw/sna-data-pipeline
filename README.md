# SNA Data Pipeline

Modal-based data engineering pipeline for building a cleaned, metadata-rich Shona speech dataset from `google/WaxalNLP` (`sna_asr`).

## Published Dataset

- Hugging Face: [manassehzw/sna-dataset](https://huggingface.co/datasets/manassehzw/sna-dataset)

## Pipeline Phases

1. `src/ingest.py` - ingest labeled splits and preserve source provenance
2. `src/annotate_metadata.py` - normalize metadata fields
3. `src/normalize_text.py` - normalize transcripts and add text flags
4. `src/normalize_audio.py` - VAD-based audio cleanup + quality metrics
5. `src/cleanup_audio.py` - drop short/singleton-speaker leftovers
6. `src/split_and_upload.py` - stratified split and Hugging Face upload

## Run Order

Run all scripts from repo root:

```bash
uv run modal run src/ingest.py
uv run modal run src/annotate_metadata.py
uv run modal run src/normalize_text.py
uv run modal run src/normalize_audio.py
uv run modal run src/cleanup_audio.py
uv run modal run src/split_and_upload.py
```
