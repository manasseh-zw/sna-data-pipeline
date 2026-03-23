import modal

app = modal.App("sna-upload-annotated")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "datasets[audio]",
    "huggingface_hub",
    "numpy",
    "pandas",
)

ANNOTATED_PATH = "/data/sna_annotated"
REPORTS_DIR = "/data/reports"


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=10800,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_dotenv()],
)
def upload_annotated():
    import io
    import json
    import os
    from datetime import datetime

    import pandas as pd
    from datasets import DatasetDict, concatenate_datasets, load_from_disk
    from huggingface_hub import HfApi, login

    print("=" * 72)
    print("SNA DATA PIPELINE - UPLOAD ANNOTATED DATASET")
    print("=" * 72)

    hf_token = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME")
    if not hf_token or not hf_username:
        raise ValueError("Missing HF_TOKEN or HF_USERNAME in environment.")

    repo_id = f"{hf_username}/sna-dataset-annotated"
    print(f"Target repo: {repo_id}")

    ds_obj = load_from_disk(ANNOTATED_PATH)
    if not isinstance(ds_obj, DatasetDict):
        raise RuntimeError(
            f"Expected DatasetDict at {ANNOTATED_PATH}, got {type(ds_obj)}"
        )

    split_order = ["train", "validation", "test"]
    splits = [s for s in split_order if s in ds_obj]
    if not splits:
        raise RuntimeError(
            "No train/validation/test splits found in /data/sna_annotated"
        )

    flat_ds = concatenate_datasets([ds_obj[s] for s in splits])

    total_clips = len(flat_ds)
    total_hours = float(pd.Series(flat_ds["duration"]).sum()) / 3600.0
    unique_speakers = len(set(flat_ds["speaker_id"]))
    gender_counts = pd.Series(flat_ds["gender"]).value_counts().to_dict()

    rebuild_audit = {}
    relabel_audit = {}
    pre_audit = {}

    rebuild_path = os.path.join(REPORTS_DIR, "rebuild_annotated_audit.json")
    relabel_path = os.path.join(REPORTS_DIR, "speaker_relabel_audit.json")
    pre_path = os.path.join(REPORTS_DIR, "pre_classification_audit.json")

    if os.path.exists(rebuild_path):
        with open(rebuild_path) as f:
            rebuild_audit = json.load(f)
    if os.path.exists(relabel_path):
        with open(relabel_path) as f:
            relabel_audit = json.load(f)
    if os.path.exists(pre_path):
        with open(pre_path) as f:
            pre_audit = json.load(f)

    login(token=hf_token)
    api = HfApi(token=hf_token)

    print("Pushing DatasetDict to Hugging Face Hub...")
    ds_obj.push_to_hub(repo_id, token=hf_token)

    pre_total = pre_audit.get("total_clips")
    pre_speakers = pre_audit.get("unique_speakers")
    noise_after = relabel_audit.get("clusters", {}).get("noise_after_rescue")
    rescued = relabel_audit.get("clusters", {}).get("rescued_clips")
    relabel_speakers = relabel_audit.get("clusters", {}).get("count")
    gender_meta = relabel_audit.get("gender_classifier", {}).get("metadata", {})

    card = f"""---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: transcription
    dtype: string
  - name: source_id
    dtype: string
  - name: speaker_id
    dtype: int32
  - name: speaker_clip_count
    dtype: int32
  - name: language
    dtype: string
  - name: gender
    dtype: string
  - name: has_punctuation
    dtype: bool
  - name: snr_db
    dtype: float32
  - name: speech_ratio
    dtype: float32
  - name: quality_score
    dtype: float32
  - name: duration
    dtype: float32
  - name: speaker_assignment_confidence
    dtype: float32
  splits:
  - name: train
    num_examples: {len(ds_obj["train"]) if "train" in ds_obj else 0}
  - name: validation
    num_examples: {len(ds_obj["validation"]) if "validation" in ds_obj else 0}
  - name: test
    num_examples: {len(ds_obj["test"]) if "test" in ds_obj else 0}
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- text-to-speech
language:
- sna
tags:
- audio
- speech
- shona
- asr
- tts
- african-language
source_datasets:
- google/WaxalNLP
pretty_name: Shona Speech Dataset (SNA) - Annotated
size_categories:
- 10K<n<100K
---

# {repo_id}

An annotated, speaker-relabelled, and loudness-normalised Shona (`sna`) speech dataset prepared through a reproducible Modal-based data engineering pipeline.

This release addresses speaker label contamination in the original source labels by replacing identity columns with acoustically-derived speaker assignments.

## Dataset Details

- **Curated by:** [Manasseh Changachirere (Harare Institute of Technology)](https://www.manasseh.dev/)
- **Derived from:** [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP/)
- **Repository:** `{repo_id}`
- **Language:** Shona (`sna`)
- **Total Clips:** {total_clips}
- **Total Speech Hours:** {round(total_hours, 3)}
- **Unique Speakers (relabelled):** {unique_speakers}

## Why this annotated release exists

The original source speaker labels are contaminated (multiple voices assigned to the same identity). This release replaces source identity labels with programmatically derived speaker clusters and rederived gender labels.

### Pre/Post relabelling snapshot

- **Pre-classification clips:** {pre_total if pre_total is not None else "N/A"}
- **Pre-classification unique source speakers:** {pre_speakers if pre_speakers is not None else "N/A"}
- **Post-classification speaker clusters:** {relabel_speakers if relabel_speakers is not None else "N/A"}
- **Noise clips dropped after rescue:** {noise_after if noise_after is not None else "N/A"}
- **Noise clips rescued by centroid similarity:** {rescued if rescued is not None else "N/A"}

## Processing Summary

1. **Pre-classification audit** over `/data/refined`.
2. **Speaker relabelling** using ECAPA embeddings + HDBSCAN + noise rescue.
3. **Gender relabelling** using a Shona-calibrated Logistic Regression classifier trained on ECAPA embeddings.
4. **Noise drop** (`cluster_id == -1`) and schema rebuild.
5. **Loudness normalisation** to -23 LUFS (EBU R128) with clipping protection.
6. **Speaker-stratified split** into train/validation/test.

## Relabelling Method

- **Speaker embeddings:** `speechbrain/spkrec-ecapa-voxceleb` (192-d)
- **Clustering:** HDBSCAN (`min_cluster_size=50`, `min_samples=10`, metric `euclidean`, method `eom`)
- **Noise rescue:** cosine similarity threshold `0.75`
- **Gender model:** Logistic Regression on L2-normalised ECAPA embeddings
  - Training clips (female): {gender_meta.get("n_female_clips", "N/A")}
  - Training clips (male): {gender_meta.get("n_male_clips", "N/A")}
  - Train accuracy: {gender_meta.get("train_accuracy", "N/A")}
  - CV accuracy: {gender_meta.get("cv_accuracy", "N/A")}

## Loudness Normalisation

- **Target:** -23 LUFS
- **Skip tolerance:** +/-1 LU
- **Post-gain protection:** hard clip to [-1.0, 1.0]
- **Input LUFS mean/std:** {rebuild_audit.get("loudness_input_lufs", {}).get("mean", "N/A")} / {rebuild_audit.get("loudness_input_lufs", {}).get("std", "N/A")}
- **Output LUFS mean/std:** {rebuild_audit.get("loudness_output_lufs", {}).get("mean", "N/A")} / {rebuild_audit.get("loudness_output_lufs", {}).get("std", "N/A")}

## Splits

- Train: {len(ds_obj["train"]) if "train" in ds_obj else 0}
- Validation: {len(ds_obj["validation"]) if "validation" in ds_obj else 0}
- Test: {len(ds_obj["test"]) if "test" in ds_obj else 0}

Split strategy is speaker-stratified by clip proportion (not speaker-disjoint), preserving speaker distribution across splits.

## Data Fields

- **`audio`**: 24kHz mono float audio (loudness-normalised)
- **`transcription`**: normalized Shona transcription
- **`source_id`**: original clip identifier from source dataset
- **`speaker_id`**: acoustically-derived speaker class id
- **`speaker_clip_count`**: clip count for the assigned speaker_id
- **`language`**: language code (`sna`)
- **`gender`**: cluster-level resolved label (`Female` / `Male` / `Unknown`)
- **`has_punctuation`**: punctuation indicator from normalized transcript
- **`snr_db`**: signal-to-noise proxy metric
- **`speech_ratio`**: fraction of VAD frames classified as speech
- **`quality_score`**: composite quality metric
- **`duration`**: clip duration in seconds
- **`speaker_assignment_confidence`**: confidence for speaker assignment

## Uses

### Direct use

- Shona ASR model training and adaptation
- TTS subset construction by filtering on speaker and quality metadata
- Speech data quality analysis and dataset curation workflows

### Out-of-scope

- Identity verification / forensic use without additional validation
- Demographic representativeness claims without dedicated study

## Bias, Risks, and Limitations

- Inherits source demographic/dialect distribution.
- Relabelled speaker IDs are acoustic clusters, not identity-verified persons.
- Confidence values are useful for filtering, not absolute truth scores.
- Some residual label uncertainty can remain in ambiguous/noisy clips.

## Citation

If you use this dataset, cite both this release and the source dataset:

```bibtex
@inproceedings{{niang2024waxalnlp,
  title={{WaxalNLP: A Large Scale High Quality Speech Dataset for African Languages}},
  author={{Niang, El Hadj Mamadou and Dieng, Moustapha and Ba, Thierno Ibrahima and Ndiaye, Mamadou Boumedine and others}},
  booktitle={{Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)}},
  year={{2024}}
}}
```
"""

    print("Uploading README.md to dataset repo...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    upload_audit = {
        "phase": "upload_annotated",
        "timestamp": datetime.now().isoformat(),
        "source_path": ANNOTATED_PATH,
        "repo_id": repo_id,
        "total_clips": int(total_clips),
        "total_hours": round(float(total_hours), 3),
        "unique_speakers": int(unique_speakers),
        "splits": {s: int(len(ds_obj[s])) for s in splits},
        "gender_distribution": {
            "Female": int(gender_counts.get("Female", 0)),
            "Male": int(gender_counts.get("Male", 0)),
            "Unknown": int(gender_counts.get("Unknown", 0)),
        },
        "readme_uploaded": True,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "upload_annotated_audit.json")
    with open(out_path, "w") as f:
        json.dump(upload_audit, f, indent=2)

    data_vol.commit()

    print("=" * 72)
    print("UPLOAD ANNOTATED COMPLETE")
    print(f"Dataset pushed: https://huggingface.co/datasets/{repo_id}")
    print(f"Audit: {out_path}")
    print("=" * 72)


@app.local_entrypoint()
def main():
    upload_annotated.remote()
