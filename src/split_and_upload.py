import modal

app = modal.App("sna-split-upload")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "datasets[audio]",
        "huggingface_hub",
        "numpy",
        "pandas",
    )
)

SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=10800,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_dotenv()],
)
def split_and_upload():
    import io
    import json
    import os
    import random
    import shutil
    from collections import defaultdict
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from datasets import DatasetDict, load_from_disk
    from huggingface_hub import HfApi, login

    print("=" * 60)
    print("SNA DATA PIPELINE - PHASE 6: SPLIT AND UPLOAD")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME")
    if not hf_token or not hf_username:
        raise ValueError("Missing HF_TOKEN or HF_USERNAME in environment.")

    login(token=hf_token)
    api = HfApi(token=hf_token)

    repo_id = f"{hf_username}/sna-dataset"
    print(f"\nTarget Hugging Face dataset repo: {repo_id}")

    print("\nLoading dataset from /data/refined ...")
    ds = load_from_disk("/data/refined")
    total_rows = len(ds)
    print(f"   Loaded {total_rows} rows")

    # Keep output schema order explicit for HF viewer readability.
    column_order = [
        "audio",
        "transcription",
        "source_id",
        "source_speaker_id",
        "speaker_idx",
        "speaker_clip_count",
        "language",
        "gender",
        "has_punctuation",
        "snr_db",
        "speech_ratio",
        "quality_score",
        "duration",
    ]
    missing = [c for c in column_order if c not in ds.column_names]
    if missing:
        raise RuntimeError(f"Missing expected columns before split: {missing}")
    ds = ds.select_columns(column_order)

    speaker_to_indices = defaultdict(list)
    for idx, spk in enumerate(ds["speaker_idx"]):
        speaker_to_indices[int(spk)].append(idx)

    rng = random.Random(SPLIT_SEED)
    train_idx, valid_idx, test_idx = [], [], []

    for _, idxs in speaker_to_indices.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)

        n_train = int(np.floor(n * TRAIN_RATIO))
        n_valid = int(np.floor(n * VALID_RATIO))
        n_test = n - n_train - n_valid

        # Ensure non-empty train split even for very small per-speaker groups.
        if n_train == 0 and n > 0:
            if n_test > 0:
                n_test -= 1
            elif n_valid > 0:
                n_valid -= 1
            n_train = 1

        train_idx.extend(idxs[:n_train])
        valid_idx.extend(idxs[n_train : n_train + n_valid])
        test_idx.extend(idxs[n_train + n_valid :])

    train_idx.sort()
    valid_idx.sort()
    test_idx.sort()

    train_ds = ds.select(train_idx)
    valid_ds = ds.select(valid_idx)
    test_ds = ds.select(test_idx)

    dataset_dict = DatasetDict(
        {
            "train": train_ds,
            "validation": valid_ds,
            "test": test_ds,
        }
    )

    print("\nSplit complete:")
    print(f"   train:      {len(train_ds)}")
    print(f"   validation: {len(valid_ds)}")
    print(f"   test:       {len(test_ds)}")

    train_speakers = set(train_ds["speaker_idx"])
    valid_speakers = set(valid_ds["speaker_idx"])
    test_speakers = set(test_ds["speaker_idx"])

    print(f"   unique speakers (train):      {len(train_speakers)}")
    print(f"   unique speakers (validation): {len(valid_speakers)}")
    print(f"   unique speakers (test):       {len(test_speakers)}")

    target_path = "/data/final"
    temp_path = "/data/final_tmp"
    backup_path = "/data/final_prev"

    for path in (temp_path, backup_path):
        if os.path.exists(path):
            print(f"Removing stale path -> {path}")
            shutil.rmtree(path)

    print(f"\nSaving DatasetDict to temporary path -> {temp_path}")
    dataset_dict.save_to_disk(temp_path)

    print("Promoting temporary dataset into place...")
    if os.path.exists(target_path):
        os.replace(target_path, backup_path)
    os.replace(temp_path, target_path)

    if os.path.exists(backup_path):
        print(f"Deleting previous dataset backup -> {backup_path}")
        shutil.rmtree(backup_path)

    print("\nPushing DatasetDict to Hugging Face Hub...")
    dataset_dict.push_to_hub(repo_id, token=hf_token)

    total_hours = float(pd.Series(ds["duration"]).sum()) / 3600.0
    card = f"""---
language:
- sna
task_categories:
- automatic-speech-recognition
license: other
pretty_name: Shona Speech Dataset (SNA)
source_datasets:
- google/WaxalNLP
tags:
- shona
- speech
- asr
- tts
---

# {repo_id}

## Summary

This dataset is a cleaned and metadata-annotated Shona (`sna`) speech corpus prepared for ASR and TTS downstream use.
It is published as a general-purpose dataset where consumers can filter clips by metadata columns instead of relying on hard-coded quality buckets.

## Source

- Original source dataset: [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) (`sna_asr` labeled splits)
- Curation and engineering pipeline: `sna-data-pipeline` (Modal-based processing)

## Processing Pipeline

1. Ingest and provenance-preserving column mapping
2. Metadata normalization
3. Text normalization
4. Audio normalization (VAD-based trimming + quality metrics)
5. Cleanup (drop short clips and singleton-speaker rows)
6. Speaker-stratified split into `train` / `validation` / `test`

## Splits

- Train: {len(train_ds)}
- Validation: {len(valid_ds)}
- Test: {len(test_ds)}
- Total clips: {len(ds)}
- Total hours: {round(total_hours, 3)}

## Columns

- `audio` (24kHz mono)
- `transcription`
- `source_id`
- `source_speaker_id`
- `speaker_idx`
- `speaker_clip_count`
- `language`
- `gender`
- `has_punctuation`
- `snr_db`
- `speech_ratio`
- `quality_score`
- `duration`

## Notes

This release is intentionally opinionation-free: clips are not pre-filtered by strict quality thresholds.
Users can apply task-specific filtering using `speaker_clip_count`, `snr_db`, `speech_ratio`, and `duration`.
"""

    print("Uploading dataset card README.md ...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    audit = {
        "phase": "split_and_upload",
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "input_rows": total_rows,
        "split_seed": SPLIT_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "splits": {
            "train": len(train_ds),
            "validation": len(valid_ds),
            "test": len(test_ds),
        },
        "speaker_coverage": {
            "train_unique_speakers": len(train_speakers),
            "validation_unique_speakers": len(valid_speakers),
            "test_unique_speakers": len(test_speakers),
        },
        "total_hours": round(total_hours, 3),
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/06_split_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit report saved -> {report_path}")

    data_vol.commit()

    print("\n" + "=" * 60)
    print("SPLIT AND UPLOAD COMPLETE")
    print(f"   Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    print(f"   Audit -> {report_path}")
    print("=" * 60)


@app.local_entrypoint()
def main():
    split_and_upload.remote()
