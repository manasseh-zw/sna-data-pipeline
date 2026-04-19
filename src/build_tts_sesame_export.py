import modal

app = modal.App("sna-build-tts-sesame-export")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "numpy",
    "pandas",
)

ANNOTATED_PATH = "/data/sna_annotated"
WAV_NORMALIZED_DIR = "/data/wav_normalised"
METADATA_CSV_PATH = f"{WAV_NORMALIZED_DIR}/metadata.csv"

OUTPUT_ROOT = "/speakers"
OUTPUT_AUDIO_DIR = f"{OUTPUT_ROOT}/audio"
OUTPUT_METADATA_CSV = f"{OUTPUT_ROOT}/metadata.csv"
OUTPUT_SUMMARY_CSV = f"{OUTPUT_ROOT}/speaker_summary.csv"

REPORT_PATH = "/data/reports/build_tts_sesame_export_audit.json"

# Optional: if provided at runtime, read config CSV from this path in volume.
DEFAULT_CONFIG_PATH = ""

# Programmatic default config from current ear-test decisions.
SPEAKER_RULES = [
    {
        "rank": 2,
        "speaker_id": 0,
        "gender": "Female",
        "inclusion_status": "primary",
        "quality_setting": "strict",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 3,
        "speaker_id": 29,
        "gender": "Male",
        "inclusion_status": "primary",
        "quality_setting": "standard",
        "quality_percentile_range": "p50-p100",
        "quality_score_floor": 70.72,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 4,
        "speaker_id": 1,
        "gender": "Female",
        "inclusion_status": "primary",
        "quality_setting": "lenient",
        "quality_percentile_range": "p50-p100",
        "quality_score_floor": 70.72,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 6,
        "speaker_id": 2,
        "gender": "Female",
        "inclusion_status": "primary",
        "quality_setting": "strict_plus",
        "quality_percentile_range": "p70-p100",
        "quality_score_floor": 75.22,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 8,
        "speaker_id": 31,
        "gender": "Male",
        "inclusion_status": "primary",
        "quality_setting": "standard",
        "quality_percentile_range": "p50-p100",
        "quality_score_floor": 70.72,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 10,
        "speaker_id": 3,
        "gender": "Female",
        "inclusion_status": "primary",
        "quality_setting": "standard",
        "quality_percentile_range": "p50-p100",
        "quality_score_floor": 70.72,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 15,
        "speaker_id": 7,
        "gender": "Female",
        "inclusion_status": "primary",
        "quality_setting": "standard_high",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 19,
        "speaker_id": 11,
        "gender": "Female",
        "inclusion_status": "primary",
        "quality_setting": "standard",
        "quality_percentile_range": "p50-p100",
        "quality_score_floor": 70.72,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 1,
        "speaker_id": 28,
        "gender": "Male",
        "inclusion_status": "conditional",
        "quality_setting": "very_strict",
        "quality_percentile_range": "p70-p100",
        "quality_score_floor": 75.22,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 5,
        "speaker_id": 32,
        "gender": "Male",
        "inclusion_status": "conditional",
        "quality_setting": "very_strict",
        "quality_percentile_range": "p70-p100",
        "quality_score_floor": 75.22,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 7,
        "speaker_id": 30,
        "gender": "Male",
        "inclusion_status": "conditional",
        "quality_setting": "strict",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 9,
        "speaker_id": 4,
        "gender": "Female",
        "inclusion_status": "conditional",
        "quality_setting": "strict",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 11,
        "speaker_id": 6,
        "gender": "Female",
        "inclusion_status": "conditional",
        "quality_setting": "strict",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 16,
        "speaker_id": 8,
        "gender": "Female",
        "inclusion_status": "conditional",
        "quality_setting": "strict",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
    {
        "rank": 17,
        "speaker_id": 35,
        "gender": "Male",
        "inclusion_status": "conditional",
        "quality_setting": "strict",
        "quality_percentile_range": "p60-p100",
        "quality_score_floor": 73.24,
        "duration_cap_s": 20.0,
    },
]


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/data": data_vol},
)
def build_tts_sesame_export(
    include_conditional: bool = False,
    config_path: str = DEFAULT_CONFIG_PATH,
    metadata_csv_path: str = METADATA_CSV_PATH,
):
    import json
    import os
    import shutil
    from datetime import datetime

    import pandas as pd

    print("=" * 72)
    print("SNA DATA PIPELINE - BUILD TTS SESAME EXPORT")
    print("=" * 72)

    if config_path and config_path.strip():
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Config CSV path was provided but not found in Modal volume: "
                f"{config_path}"
            )
        print(f"Config source: CSV ({config_path})")
        cfg = pd.read_csv(config_path)
    else:
        print("Config source: in-script SPEAKER_RULES object")
        cfg = pd.DataFrame(SPEAKER_RULES)

    required_cfg = {
        "speaker_id",
        "gender",
        "inclusion_status",
        "quality_score_floor",
        "duration_cap_s",
        "quality_setting",
    }
    missing_cfg = sorted(required_cfg - set(cfg.columns))
    if missing_cfg:
        raise RuntimeError(f"Missing config columns: {missing_cfg}")

    cfg = cfg.copy()
    cfg["speaker_id"] = cfg["speaker_id"].astype("int32")
    cfg["quality_score_floor"] = cfg["quality_score_floor"].astype("float32")
    cfg["duration_cap_s"] = cfg["duration_cap_s"].astype("float32")
    cfg["inclusion_status"] = (
        cfg["inclusion_status"].astype(str).str.lower().str.strip()
    )

    allowed_status = {"primary", "conditional"} if include_conditional else {"primary"}
    cfg = cfg[cfg["inclusion_status"].isin(allowed_status)].copy()
    if cfg.empty:
        raise RuntimeError("No speakers left after inclusion_status filter.")

    cfg_map = {
        int(row.speaker_id): {
            "gender": str(row.gender),
            "inclusion_status": str(row.inclusion_status),
            "quality_setting": str(row.quality_setting),
            "quality_score_floor": float(row.quality_score_floor),
            "duration_cap_s": float(row.duration_cap_s),
        }
        for row in cfg.itertuples(index=False)
    }

    print(f"Selected speakers from config: {len(cfg_map)}")
    print(f"Including conditional speakers: {include_conditional}")

    print("\nLoading wav-normalised metadata ...")
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(
            f"Missing metadata.csv at: {metadata_csv_path}\n"
            "Run: modal run src/export_wav_normalised_metadata.py"
        )

    needed_cols = [
        "file_name",
        "source_id",
        "speaker_id",
        "gender",
        "transcription",
        "quality_score",
        "duration",
    ]
    df = pd.read_csv(metadata_csv_path)
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing expected metadata.csv columns: {missing_cols}")

    df["speaker_id"] = df["speaker_id"].astype("int32")
    df["quality_score"] = df["quality_score"].astype("float32")
    df["duration"] = df["duration"].astype("float32")
    df["source_id"] = df["source_id"].astype(str)
    df["file_name"] = df["source_id"] + ".wav"

    print(f"Metadata clips available: {len(df):,}")

    def row_passes(row):
        sid = int(row["speaker_id"])
        rule = cfg_map.get(sid)
        if rule is None:
            return False
        if float(row["quality_score"]) < rule["quality_score_floor"]:
            return False
        if float(row["duration"]) > rule["duration_cap_s"]:
            return False
        return True

    keep_mask = df.apply(row_passes, axis=1)
    kept = df[keep_mask].copy()
    if kept.empty:
        raise RuntimeError("No records passed the configured speaker filters.")

    kept["file_name"] = kept["source_id"].astype(str) + ".wav"
    kept["inclusion_status"] = kept["speaker_id"].map(
        lambda s: cfg_map[int(s)]["inclusion_status"]
    )
    kept["quality_setting"] = kept["speaker_id"].map(
        lambda s: cfg_map[int(s)]["quality_setting"]
    )

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    os.makedirs("/data/reports", exist_ok=True)

    print(f"\nExporting audio files to: {OUTPUT_AUDIO_DIR}")
    copied = 0
    missing_wav = 0
    for row in kept.itertuples(index=False):
        src = os.path.join(WAV_NORMALIZED_DIR, f"{row.source_id}.wav")
        dst = os.path.join(OUTPUT_AUDIO_DIR, row.file_name)
        if not os.path.exists(src):
            missing_wav += 1
            continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    if missing_wav > 0:
        print(f"WARNING: Missing normalized wav files: {missing_wav}")

    metadata = kept[
        [
            "file_name",
            "speaker_id",
            "gender",
            "transcription",
            "source_id",
            "quality_score",
            "duration",
            "quality_setting",
            "inclusion_status",
        ]
    ].copy()
    metadata.to_csv(OUTPUT_METADATA_CSV, index=False)

    summary = (
        kept.groupby(["speaker_id", "gender", "inclusion_status", "quality_setting"])
        .agg(
            clips=("source_id", "count"),
            hours=("duration", lambda x: float(x.sum()) / 3600.0),
            mean_quality=("quality_score", "mean"),
            mean_duration=("duration", "mean"),
        )
        .reset_index()
        .sort_values(["clips", "hours"], ascending=False)
    )
    summary["hours"] = summary["hours"].round(3)
    summary["mean_quality"] = summary["mean_quality"].round(2)
    summary["mean_duration"] = summary["mean_duration"].round(2)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    total_clips = int(len(kept))
    total_hours = float(kept["duration"].sum()) / 3600.0
    unique_speakers = int(kept["speaker_id"].nunique())

    report = {
        "timestamp": datetime.now().isoformat(),
        "source_metadata_csv": metadata_csv_path,
        "config_source": "csv" if (config_path and config_path.strip()) else "object",
        "config_csv": config_path,
        "include_conditional": include_conditional,
        "output_root": OUTPUT_ROOT,
        "output_audio_dir": OUTPUT_AUDIO_DIR,
        "output_metadata_csv": OUTPUT_METADATA_CSV,
        "output_summary_csv": OUTPUT_SUMMARY_CSV,
        "totals": {
            "clips": total_clips,
            "hours": round(total_hours, 3),
            "unique_speakers": unique_speakers,
            "copied_audio_files": copied,
            "missing_normalized_wavs": int(missing_wav),
        },
        "speakers": summary.to_dict(orient="records"),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    data_vol.commit()

    print("\nBuild complete")
    print(f"  Clips:   {total_clips:,}")
    print(f"  Hours:   {total_hours:.3f}")
    print(f"  Speakers:{unique_speakers}")
    print(f"  Metadata:{OUTPUT_METADATA_CSV}")
    print(f"  Audio:   {OUTPUT_AUDIO_DIR}")
    print(f"  Report:  {REPORT_PATH}")


@app.local_entrypoint()
def main(include_conditional: bool = False, config_path: str = ""):
    build_tts_sesame_export.remote(
        include_conditional=include_conditional,
        config_path=config_path,
    )
