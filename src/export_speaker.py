"""
export_speaker.py
-----------------
Isolate and export clips + metadata for a single speaker from the SNA
annotated dataset.  Gender is inferred from the data — you only need to
know the numeric speaker ID.

Usage (via Modal CLI):
  modal run src/export_speaker.py \\
      --speaker-id 0 \\
      [--quality-floor 73.24] \\
      [--quality-percentile 60] \\
      [--duration-cap 20.0]

Arguments
---------
  --speaker-id         (required) Integer speaker ID to isolate.
  --quality-floor      (optional) Minimum quality_score to accept.
                       If omitted, no lower-bound quality filter is applied.
  --quality-percentile (optional) Keep only clips whose quality_score is at or
                       above this percentile *within the speaker's own clips*.
                       Applied AFTER --quality-floor (if provided).
                       Range 0-100.  If omitted, percentile filter is skipped.
  --duration-cap       (optional) Maximum clip duration in seconds.
                       If omitted, no duration cap is applied.

Output layout
-------------
  /speakers/<speaker_id>_<gender>/   (gender resolved automatically from data)
      audio/               WAV files (copied from wav_normalised volume path)
      metadata.csv         file_name, transcription, quality_score, duration, source_id
      export_report.json   run summary / audit trail
"""

import modal

app = modal.App("sna-export-speaker")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "numpy",
    "pandas",
)

# ── Volume paths ────────────────────────────────────────────────────────────
ANNOTATED_PATH = "/data/sna_annotated"
WAV_NORMALIZED_DIR = "/data/wav_normalised"
METADATA_CSV_PATH = f"{WAV_NORMALIZED_DIR}/metadata.csv"
SPEAKERS_ROOT = "/data/speakers"


# ── Modal function ───────────────────────────────────────────────────────────
@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/data": data_vol},
)
def export_speaker(
    speaker_id: int,
    quality_floor: float = -1.0,          # sentinel: -1.0 → no floor
    quality_percentile: float = -1.0,     # sentinel: -1.0 → no percentile filter
    duration_cap: float = -1.0,           # sentinel: -1.0 → no cap
    output_suffix: str = "",
    metadata_csv_path: str = METADATA_CSV_PATH,
):
    import json
    import os
    import shutil
    from datetime import datetime

    import numpy as np
    import pandas as pd

    print("=" * 72)
    print("SNA DATA PIPELINE - SPEAKER ISOLATION EXPORT")
    print("=" * 72)
    print(f"  Speaker ID        : {speaker_id}")
    print(f"  Quality floor     : {quality_floor if quality_floor >= 0 else 'none'}")
    print(
        f"  Quality percentile: "
        f"{quality_percentile if quality_percentile >= 0 else 'none'}"
    )
    print(f"  Duration cap (s)  : {duration_cap if duration_cap >= 0 else 'none'}")
    print(f"  Output suffix     : {output_suffix.strip() if output_suffix.strip() else 'none'}")
    print()

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading wav-normalised metadata ...")
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

    print(f"Total wav-normalised clips: {len(df):,}")

    # ── Filter to target speaker ─────────────────────────────────────────────
    df = df[df["speaker_id"] == int(speaker_id)].copy()
    if df.empty:
        raise RuntimeError(f"No clips found for speaker_id={speaker_id}.")

    # ── Infer gender from data ───────────────────────────────────────────────
    gender = str(df["gender"].mode().iloc[0]).strip()
    print(f"Inferred gender for speaker {speaker_id}: {gender}")
    print(f"Clips for speaker {speaker_id}: {len(df):,}")

    # ── Derived output paths (gender now known) ──────────────────────────────
    clean_suffix = output_suffix.strip().strip("_")
    folder_name = f"{speaker_id}_{gender}"
    if clean_suffix:
        folder_name = f"{folder_name}_{clean_suffix}"
    speaker_out = os.path.join(SPEAKERS_ROOT, folder_name)
    audio_out = os.path.join(speaker_out, "audio")
    metadata_csv = os.path.join(speaker_out, "metadata.csv")
    report_json = os.path.join(speaker_out, "export_report.json")
    print(f"Output directory    : {speaker_out}")

    # ── Optional: absolute quality floor ────────────────────────────────────
    if quality_floor >= 0:
        before = len(df)
        df = df[df["quality_score"] >= quality_floor].copy()
        print(f"After quality floor  ({quality_floor:.2f}): {len(df):,}  "
              f"(dropped {before - len(df):,})")
        if df.empty:
            raise RuntimeError("No clips remain after quality_floor filter.")

    # ── Optional: within-speaker percentile gate ─────────────────────────────
    if quality_percentile >= 0:
        threshold = float(np.percentile(df["quality_score"].values, quality_percentile))
        before = len(df)
        df = df[df["quality_score"] >= threshold].copy()
        print(
            f"After p{quality_percentile:.0f} percentile (>={threshold:.2f}): "
            f"{len(df):,}  (dropped {before - len(df):,})"
        )
        if df.empty:
            raise RuntimeError("No clips remain after quality_percentile filter.")

    # ── Optional: duration cap ───────────────────────────────────────────────
    if duration_cap >= 0:
        before = len(df)
        df = df[df["duration"] <= duration_cap].copy()
        print(f"After duration cap   (<={duration_cap:.1f}s): {len(df):,}  "
              f"(dropped {before - len(df):,})")
        if df.empty:
            raise RuntimeError("No clips remain after duration_cap filter.")

    # ── Prepare output directories ────────────────────────────────────────────
    os.makedirs(audio_out, exist_ok=True)

    # ── Copy WAV files ────────────────────────────────────────────────────────
    print(f"\nCopying audio to: {audio_out}")
    df["file_name"] = df["source_id"].astype(str) + ".wav"
    copied = 0
    missing_wav = 0
    for row in df.itertuples(index=False):
        src = os.path.join(WAV_NORMALIZED_DIR, f"{row.source_id}.wav")
        dst = os.path.join(audio_out, row.file_name)
        if not os.path.exists(src):
            missing_wav += 1
            continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1

    if missing_wav:
        print(f"WARNING: Missing normalized WAVs: {missing_wav}")

    # ── Write metadata CSV ────────────────────────────────────────────────────
    metadata = df[["file_name", "transcription", "quality_score",
                   "duration", "source_id"]].copy()
    metadata.to_csv(metadata_csv, index=False)
    print(f"Metadata CSV written: {metadata_csv}  ({len(metadata):,} rows)")

    # ── Build report ──────────────────────────────────────────────────────────
    total_clips = len(df)
    total_hours = float(df["duration"].sum()) / 3600.0
    mean_quality = float(df["quality_score"].mean())
    mean_duration = float(df["duration"].mean())

    report = {
        "timestamp": datetime.now().isoformat(),
        "speaker_id": speaker_id,
        "gender": gender,   # inferred from data
        "source_metadata_csv": metadata_csv_path,
        "filters": {
            "quality_floor": quality_floor if quality_floor >= 0 else None,
            "quality_percentile": quality_percentile if quality_percentile >= 0 else None,
            "duration_cap_s": duration_cap if duration_cap >= 0 else None,
        },
        "output": {
            "folder": speaker_out,
            "audio_dir": audio_out,
            "metadata_csv": metadata_csv,
        },
        "totals": {
            "clips": total_clips,
            "hours": round(total_hours, 4),
            "mean_quality_score": round(mean_quality, 2),
            "mean_duration_s": round(mean_duration, 2),
            "copied_audio_files": copied,
            "missing_normalized_wavs": missing_wav,
        },
    }

    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    data_vol.commit()

    print("\nExport complete")
    print(f"  Clips   : {total_clips:,}")
    print(f"  Hours   : {total_hours:.4f}")
    print(f"  Quality : {mean_quality:.2f} mean")
    print(f"  Audio   : {audio_out}")
    print(f"  Metadata: {metadata_csv}")
    print(f"  Report  : {report_json}")


# ── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    speaker_id: int,
    quality_floor: float = -1.0,
    quality_percentile: float = -1.0,
    duration_cap: float = -1.0,
    output_suffix: str = "",
):
    """
    CLI wrapper.  Pass -1 (the default) to skip any optional filter.
    Gender is inferred automatically from the dataset.

    Examples
    --------
    # All clips for speaker 0, no filters:
    modal run src/export_speaker.py --speaker-id 0

    # Top 40 % quality clips (within speaker), no longer than 15 s:
    modal run src/export_speaker.py --speaker-id 0 \\
        --quality-percentile 60 --duration-cap 15

    # Absolute quality floor only:
    modal run src/export_speaker.py --speaker-id 29 --quality-floor 73.5

    # Both floor and percentile:
    modal run src/export_speaker.py --speaker-id 2 \\
        --quality-floor 70.0 --quality-percentile 70 --duration-cap 20
    """
    export_speaker.remote(
        speaker_id=speaker_id,
        quality_floor=quality_floor,
        quality_percentile=quality_percentile,
        duration_cap=duration_cap,
        output_suffix=output_suffix,
    )
