import modal

app = modal.App("sna-cleanup-audio")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "datasets[audio]",
        "pandas",
    )
)

MIN_DURATION_SEC = 3.0


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    volumes={"/data": data_vol},
)
def cleanup_audio():
    import json
    import os
    import shutil
    from datetime import datetime

    import pandas as pd
    from datasets import Dataset, load_from_disk

    print("=" * 60)
    print("SNA DATA PIPELINE - PHASE 5: CLEANUP AUDIO")
    print("=" * 60)

    print("\nLoading dataset from /data/refined ...")
    ds = load_from_disk("/data/refined")
    total_rows = len(ds)
    print(f"   Loaded {total_rows} rows")

    df = ds.to_pandas()

    too_short_mask = df["duration"] < MIN_DURATION_SEC
    dropped_short_df = df.loc[too_short_mask, ["source_id", "source_speaker_id", "duration"]].copy()
    df = df.loc[~too_short_mask].copy()

    speaker_counts_after_duration = df["source_speaker_id"].value_counts()
    singleton_ids = set(speaker_counts_after_duration[speaker_counts_after_duration == 1].index.tolist())
    singleton_mask = df["source_speaker_id"].isin(singleton_ids)
    dropped_singleton_df = df.loc[singleton_mask, ["source_id", "source_speaker_id", "duration"]].copy()
    df = df.loc[~singleton_mask].copy()

    if len(df) == 0:
        raise RuntimeError("Cleanup removed all rows. Check filtering rules before proceeding.")

    refreshed_counts = df["source_speaker_id"].value_counts().to_dict()
    df["speaker_clip_count"] = df["source_speaker_id"].map(refreshed_counts).astype(int)

    kept_rows = len(df)
    dropped_rows = total_rows - kept_rows
    drop_rate_pct = round((dropped_rows / total_rows) * 100, 2) if total_rows else 0.0

    cleaned_ds = Dataset.from_pandas(df, preserve_index=False)

    unique_speakers = int(df["source_speaker_id"].nunique())
    duration_series = pd.Series(cleaned_ds["duration"])
    total_hours = float(duration_series.sum()) / 3600.0

    audit = {
        "phase": "cleanup_audio",
        "timestamp": datetime.now().isoformat(),
        "input_rows": int(total_rows),
        "kept_rows": int(kept_rows),
        "dropped_rows": int(dropped_rows),
        "drop_rate_pct": drop_rate_pct,
        "drop_reasons": {
            "under_5s": int(len(dropped_short_df)),
            "singleton_speaker": int(len(dropped_singleton_df)),
        },
        "min_duration_sec": MIN_DURATION_SEC,
        "post_cleanup": {
            "unique_speakers": unique_speakers,
            "total_hours": round(total_hours, 3),
            "min_speaker_clip_count": int(df["speaker_clip_count"].min()),
            "max_speaker_clip_count": int(df["speaker_clip_count"].max()),
        },
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/05_cleanup_audio_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit report saved -> {report_path}")

    target_path = "/data/refined"
    temp_path = "/data/refined_tmp"
    backup_path = "/data/refined_prev"

    for path in (temp_path, backup_path):
        if os.path.exists(path):
            print(f"Removing stale path -> {path}")
            shutil.rmtree(path)

    print(f"Saving cleaned dataset to temporary path -> {temp_path}")
    cleaned_ds.save_to_disk(temp_path)

    print("Promoting temporary dataset into place...")
    if os.path.exists(target_path):
        os.replace(target_path, backup_path)
    os.replace(temp_path, target_path)

    if os.path.exists(backup_path):
        print(f"Deleting previous dataset backup -> {backup_path}")
        shutil.rmtree(backup_path)

    data_vol.commit()

    print("\n" + "=" * 60)
    print("CLEANUP AUDIO COMPLETE")
    print(f"   {kept_rows} clips kept from {total_rows} input rows")
    print(f"   {round(total_hours, 2)} hours retained")
    print(f"   {unique_speakers} speakers retained")
    print(f"   Audit -> {report_path}")
    print("=" * 60)


@app.local_entrypoint()
def main():
    cleanup_audio.remote()
