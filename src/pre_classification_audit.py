import modal

app = modal.App("sna-pre-classification-audit")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "datasets[audio]",
        "soundfile",
        "numpy",
        "pandas",
    )
)

WAV_DIR         = "/data/wav_cache"
CHECKPOINT_FILE = "/data/wav_cache/.wav_checkpoint.json"
CHECKPOINT_EVERY = 500
SAMPLE_RATE     = 24_000


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/data": data_vol},
)
def pre_classification_audit():
    import json
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from datasets import load_from_disk

    print("=" * 60)
    print("SNA DATA PIPELINE — PRE-CLASSIFICATION AUDIT")
    print("WAV extraction + dataset snapshot")
    print("=" * 60)

    # ── 1. Load flat dataset ──────────────────────────────────────────────────
    print("\n📂 Loading dataset from /data/refined/ ...")
    ds = load_from_disk("/data/refined")
    total_clips = len(ds)
    print(f"   {total_clips} clips  |  columns: {ds.column_names}")

    # ── 2. Audit stats (always recompute — fast, metadata only) ──────────────
    print("\n📊 Computing pre-classification statistics ...")

    meta_cols = [
        "source_id", "source_speaker_id", "speaker_idx", "gender",
        "speaker_clip_count", "duration", "snr_db", "speech_ratio", "quality_score",
    ]
    missing = [c for c in meta_cols if c not in ds.column_names]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")

    df = ds.select_columns(meta_cols).to_pandas()

    total_hours = float(df["duration"].sum()) / 3600.0
    gender_dist = df["gender"].value_counts().to_dict()

    spk_df = (
        df.groupby("source_speaker_id")
        .agg(
            clip_count        = ("source_id", "count"),
            speaker_idx       = ("speaker_idx", "first"),
            gender_majority   = ("gender", lambda x: x.value_counts().index[0]),
            gender_labels     = ("gender", lambda x: sorted(x.unique().tolist())),
            total_hours       = ("duration", lambda x: x.sum() / 3600.0),
            mean_duration     = ("duration", "mean"),
            mean_snr_db       = ("snr_db", "mean"),
            mean_speech_ratio = ("speech_ratio", "mean"),
            mean_quality_score= ("quality_score", "mean"),
        )
        .reset_index()
        .sort_values("clip_count", ascending=False)
    )
    spk_df["gender_conflict"] = spk_df["gender_labels"].apply(lambda g: len(g) > 1)
    n_gender_conflict  = int(spk_df["gender_conflict"].sum())
    n_singleton        = int((spk_df["clip_count"] == 1).sum())
    spk_gender_dist    = spk_df["gender_majority"].value_counts().to_dict()

    print(f"   Total clips:              {total_clips}")
    print(f"   Total hours:              {total_hours:.2f}h")
    print(f"   Unique source_speaker_id: {len(spk_df)}")
    print(f"   Gender conflict speakers: {n_gender_conflict}")
    print(f"   Singleton speakers:       {n_singleton}")
    print(f"\n   Gender distribution (original labels):")
    for g, c in sorted(gender_dist.items()):
        print(f"     {g:<10} {c:>6} clips  ({c/total_clips*100:.1f}%)")
    print(f"\n   Gender by speaker majority:")
    for g, c in sorted(spk_gender_dist.items()):
        print(f"     {g:<10} {c:>4} speakers  ({c/len(spk_df)*100:.1f}%)")

    # ── 3. Write metadata CSV + audit JSON ────────────────────────────────────
    os.makedirs("/data/relabel", exist_ok=True)
    os.makedirs("/data/reports", exist_ok=True)
    os.makedirs(WAV_DIR, exist_ok=True)

    meta_csv = "/data/relabel/pre_audit_metadata.csv"
    df.to_csv(meta_csv, index=False)
    print(f"\n💾 Metadata CSV → {meta_csv}  ({len(df)} rows)")

    spk_records = [
        {
            "source_speaker_id":  str(r["source_speaker_id"]),
            "speaker_idx":        int(r["speaker_idx"]),
            "clip_count":         int(r["clip_count"]),
            "gender_majority":    str(r["gender_majority"]),
            "gender_labels":      r["gender_labels"],
            "gender_conflict":    bool(r["gender_conflict"]),
            "total_hours":        round(float(r["total_hours"]), 4),
            "mean_duration_s":    round(float(r["mean_duration"]), 3),
            "mean_snr_db":        round(float(r["mean_snr_db"]), 3),
            "mean_speech_ratio":  round(float(r["mean_speech_ratio"]), 4),
            "mean_quality_score": round(float(r["mean_quality_score"]), 3),
        }
        for _, r in spk_df.iterrows()
    ]

    audit = {
        "phase":            "pre_classification_audit",
        "timestamp":        datetime.now().isoformat(),
        "source_path":      "/data/refined",
        "total_clips":      int(total_clips),
        "total_hours":      round(float(total_hours), 3),
        "unique_speakers":  int(len(spk_df)),
        "gender_distribution_original": {str(k): int(v) for k, v in gender_dist.items()},
        "gender_distribution_by_speaker_majority": {str(k): int(v) for k, v in spk_gender_dist.items()},
        "speakers_with_gender_conflict": n_gender_conflict,
        "singleton_speakers_remaining":  n_singleton,
        "duration_stats": {
            "mean_s":   round(float(df["duration"].mean()), 3),
            "median_s": round(float(df["duration"].median()), 3),
            "min_s":    round(float(df["duration"].min()), 3),
            "max_s":    round(float(df["duration"].max()), 3),
        },
        "snr_stats": {
            "mean_db":   round(float(df["snr_db"].mean()), 3),
            "median_db": round(float(df["snr_db"].median()), 3),
            "min_db":    round(float(df["snr_db"].min()), 3),
            "max_db":    round(float(df["snr_db"].max()), 3),
        },
        "per_speaker": spk_records,
    }

    report_path = "/data/reports/pre_classification_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"💾 Audit JSON     → {report_path}")

    # ── 4. WAV extraction with checkpointing ──────────────────────────────────
    # Read checkpoint — resume from last completed index if present
    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        last = int(ckpt.get("last_completed_idx", -1))
        if last >= total_clips - 1:
            print(f"\n✅ WAV extraction already complete per checkpoint ({last + 1} clips).")
            data_vol.commit()
            return
        start_idx = last + 1
        print(f"\n🔖 Resuming WAV extraction from index {start_idx} ({total_clips - start_idx} remaining)")
    else:
        print(f"\n🔊 Starting WAV extraction from index 0 ({total_clips} clips) ...")

    # Select only the remaining slice — never iterates already-done clips
    ds_slice = ds.select(range(start_idx, total_clips))

    n_written = 0
    n_failed  = 0

    for i, example in enumerate(ds_slice):
        global_idx = start_idx + i
        source_id  = example["source_id"]
        wav_path   = f"{WAV_DIR}/{source_id}.wav"

        try:
            arr = np.array(example["audio"]["array"], dtype=np.float32)
            sf.write(wav_path, arr, SAMPLE_RATE, subtype="FLOAT")
            n_written += 1
        except Exception as e:
            n_failed += 1
            if n_failed <= 10:
                print(f"   ⚠️  FAILED [{source_id}]: {e}")

        # Checkpoint: write index + commit volume every N clips
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"last_completed_idx": global_idx}, f)
            data_vol.commit()
            pct = (global_idx + 1) / total_clips * 100
            print(
                f"   [{global_idx + 1}/{total_clips}] {pct:.1f}%  "
                f"written={n_written}  failed={n_failed}  ✓ checkpoint"
            )

    # Final checkpoint + commit
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_completed_idx": total_clips - 1}, f)

    # Append WAV summary to audit JSON
    audit["wav_extraction"] = {
        "output_path":    WAV_DIR,
        "sample_rate_hz": SAMPLE_RATE,
        "start_idx":      start_idx,
        "written":        n_written,
        "failed":         n_failed,
        "total_in_cache": start_idx + n_written,
    }
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)

    data_vol.commit()

    print("\n" + "=" * 60)
    print("✅ PRE-CLASSIFICATION AUDIT COMPLETE")
    print(f"   {total_clips} clips  |  {total_hours:.2f}h  |  {len(spk_df)} speakers")
    print(f"   WAV written: {n_written}  |  failed: {n_failed}  |  total: {start_idx + n_written}")
    print(f"   Metadata CSV  → {meta_csv}")
    print(f"   Audit JSON    → {report_path}")
    print(f"   WAV files     → {WAV_DIR}/{{source_id}}.wav")
    print("=" * 60)


@app.local_entrypoint()
def main():
    pre_classification_audit.remote()
