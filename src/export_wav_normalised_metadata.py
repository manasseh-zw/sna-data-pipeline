import modal

app = modal.App("sna-export-wav-normalised-metadata")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "datasets[audio]",
        "numpy",
        "pandas",
    )
)

ANNOTATED_PATH = "/data/sna_annotated"
WAV_NORMALIZED_DIR = "/data/wav_normalised"
METADATA_CSV_PATH = f"{WAV_NORMALIZED_DIR}/metadata.csv"
REPORT_JSON_PATH = "/data/reports/export_wav_normalised_metadata_audit.json"


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/data": data_vol},
)
def export_wav_normalised_metadata(
    metadata_csv_path: str = METADATA_CSV_PATH,
    overwrite: bool = False,
    validate_wavs: bool = True,
):
    import json
    import os
    import shutil
    from datetime import datetime

    import pandas as pd
    from datasets import concatenate_datasets, load_from_disk

    print("=" * 72)
    print("SNA DATA PIPELINE - EXPORT WAV NORMALISED METADATA")
    print("=" * 72)

    if not os.path.exists(ANNOTATED_PATH):
        raise FileNotFoundError(f"Missing annotated dataset: {ANNOTATED_PATH}")

    if not os.path.isdir(WAV_NORMALIZED_DIR):
        raise FileNotFoundError(f"Missing WAV normalized directory: {WAV_NORMALIZED_DIR}")

    if os.path.exists(metadata_csv_path) and not overwrite:
        print(f"metadata.csv already exists: {metadata_csv_path}")
        print("Use --overwrite=true to regenerate.")
        return

    ds_dict = load_from_disk(ANNOTATED_PATH)
    if not ds_dict or not getattr(ds_dict, "keys", None):
        raise RuntimeError(f"Unexpected dataset object from {ANNOTATED_PATH}")

    ds_full = concatenate_datasets([ds_dict[s] for s in ds_dict.keys()])

    needed_cols = [
        "source_id",
        "speaker_id",
        "gender",
        "transcription",
        "quality_score",
        "duration",
    ]
    missing_cols = [c for c in needed_cols if c not in ds_full.column_names]
    if missing_cols:
        raise RuntimeError(f"Missing expected dataset columns: {missing_cols}")

    print("\nConverting dataset to dataframe ...")
    df = ds_full.select_columns(needed_cols).to_pandas()

    df["source_id"] = df["source_id"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype("int32")
    df["quality_score"] = df["quality_score"].astype("float32")
    df["duration"] = df["duration"].astype("float32")

    df["file_name"] = df["source_id"] + ".wav"

    before = len(df)
    if validate_wavs:
        wav_files = {
            fn for fn in os.listdir(WAV_NORMALIZED_DIR) if fn.endswith(".wav")
        }
        df = df[df["file_name"].isin(wav_files)].copy()
    after = len(df)

    os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)
    tmp_path = f"{metadata_csv_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    df.to_csv(tmp_path, index=False)
    shutil.move(tmp_path, metadata_csv_path)

    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "annotated_dataset_path": ANNOTATED_PATH,
        "wav_normalised_dir": WAV_NORMALIZED_DIR,
        "metadata_csv_path": metadata_csv_path,
        "total_rows_from_dataset": int(before),
        "rows_after_validation": int(after),
        "rows_removed_due_to_missing_wav": int(before - after) if validate_wavs else None,
        "columns": list(df.columns),
        "validate_wavs": validate_wavs,
        "overwrite": overwrite,
    }
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\nExport complete")
    print(f"  metadata.csv : {metadata_csv_path}")
    print(f"  rows         : {after:,}")
    print(f"  report       : {REPORT_JSON_PATH}")

    data_vol.commit()


@app.local_entrypoint()
def main(
    metadata_csv_path: str = METADATA_CSV_PATH,
    overwrite: bool = False,
    validate_wavs: bool = True,
):
    export_wav_normalised_metadata.remote(
        metadata_csv_path=metadata_csv_path,
        overwrite=overwrite,
        validate_wavs=validate_wavs,
    )

