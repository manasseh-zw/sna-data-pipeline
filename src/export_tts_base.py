"""
export_tts_base.py
------------------
Build a Phase-1 (phonetics-first) TTS export from matrix-style bands.

Default profile targets your current plan:
  - speaker pool: top_20
  - quality band: p50 (global floor from audit report)
  - duration cap: 25s

Usage (via Modal CLI):
  modal run src/export_tts_base.py

  modal run src/export_tts_base.py \
      --speaker-pool top_20 \
      --quality-label p50 \
      --duration-cap 25

  # Override quality floor directly (ignores quality_label):
  modal run src/export_tts_base.py \
      --speaker-pool top_20 \
      --quality-floor 70.72 \
      --duration-cap 25

Output layout:
  /data/exports/<band_name>/
      audio/
      metadata.csv
      speaker_summary.csv
      export_report.json
"""

import modal

app = modal.App("sna-export-tts-base")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "numpy",
    "pandas",
)

ANNOTATED_PATH = "/data/sna_annotated"
WAV_NORMALIZED_DIR = "/data/wav_normalised"
METADATA_CSV_PATH = f"{WAV_NORMALIZED_DIR}/metadata.csv"
EXPORTS_ROOT = "/data/exports"

DEFAULT_AUDIT_JSON = "/data/reports/tts_expansion_audit.json"


def _format_duration_tag(duration_cap: float) -> str:
    if float(duration_cap).is_integer():
        return f"D{int(duration_cap)}"
    return f"D{str(duration_cap).replace('.', 'p')}"


def _default_band_name(speaker_pool: str, quality_tag: str, duration_cap: float) -> str:
    pool = speaker_pool.strip().lower()
    if pool.startswith("top_"):
        n = pool.split("_", 1)[1]
        pool_tag = f"T{n}"
    else:
        pool_tag = pool.upper().replace("-", "_")
    return f"{pool_tag}_{quality_tag}_{_format_duration_tag(duration_cap)}"


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/data": data_vol},
)
def export_tts_base(
    speaker_pool: str = "top_20",
    quality_label: str = "p50",
    duration_cap: float = 25.0,
    quality_floor: float = -1.0,
    output_name: str = "",
    audit_json_path: str = DEFAULT_AUDIT_JSON,
    output_root: str = EXPORTS_ROOT,
    metadata_csv_path: str = METADATA_CSV_PATH,
):
    import json
    import os
    import shutil
    from datetime import datetime

    import pandas as pd

    print("=" * 72)
    print("SNA DATA PIPELINE - EXPORT TTS BASE SET")
    print("=" * 72)

    if not os.path.exists(audit_json_path):
        raise FileNotFoundError(
            "Audit report not found. Run src/audit_tts_expansion_options.py first, "
            f"or pass --audit-json-path. Missing: {audit_json_path}"
        )

    with open(audit_json_path, "r") as f:
        audit = json.load(f)

    quality_map = audit.get("quality_percentiles", {})
    speaker_sets = audit.get("speaker_sets", {})
    top_n_sets = speaker_sets.get("top_n", {})

    available_pools = {
        "primary_only": speaker_sets.get("primary_only", []),
        "primary_plus_conditional": speaker_sets.get("primary_plus_conditional", []),
        **top_n_sets,
    }

    if speaker_pool not in available_pools:
        raise RuntimeError(
            f"Unknown speaker pool '{speaker_pool}'. "
            f"Available pools: {sorted(available_pools.keys())}"
        )

    selected_speakers = [int(x) for x in available_pools[speaker_pool]]
    if not selected_speakers:
        raise RuntimeError(f"Speaker pool '{speaker_pool}' is empty.")

    q_label = quality_label.strip().lower()
    if quality_floor >= 0:
        resolved_quality_floor = float(quality_floor)
        quality_tag = f"Q{str(round(resolved_quality_floor, 2)).replace('.', 'p')}"
    else:
        if q_label not in quality_map:
            raise RuntimeError(
                f"quality_label '{quality_label}' not found in audit file. "
                f"Available labels: {sorted(quality_map.keys())}"
            )
        resolved_quality_floor = float(quality_map[q_label])
        quality_tag = q_label.upper()

    band_name = (
        output_name.strip()
        if output_name.strip()
        else _default_band_name(
            speaker_pool=speaker_pool,
            quality_tag=quality_tag,
            duration_cap=duration_cap,
        )
    )

    export_root = os.path.join(output_root, band_name)
    audio_out = os.path.join(export_root, "audio")
    metadata_csv = os.path.join(export_root, "metadata.csv")
    summary_csv = os.path.join(export_root, "speaker_summary.csv")
    report_json = os.path.join(export_root, "export_report.json")

    print(f"Speaker pool        : {speaker_pool} ({len(selected_speakers)} speakers)")
    print(f"Quality label       : {quality_label}")
    print(f"Quality floor       : {resolved_quality_floor:.4f}")
    print(f"Duration cap (s)    : {duration_cap}")
    print(f"Output band         : {band_name}")
    print(f"Output root         : {export_root}")

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

    before = len(df)
    kept = df[
        (df["speaker_id"].isin(selected_speakers))
        & (df["quality_score"] >= resolved_quality_floor)
        & (df["duration"] <= float(duration_cap))
    ].copy()

    if kept.empty:
        raise RuntimeError("No records passed the selected matrix band filters.")

    kept["file_name"] = kept["source_id"].astype(str) + ".wav"

    os.makedirs(audio_out, exist_ok=True)

    print(f"\nRows before filter  : {before:,}")
    print(f"Rows after filter   : {len(kept):,}")
    print(f"Copying audio files : {audio_out}")

    copied_new = 0
    audio_present = 0
    missing_wav = 0
    for row in kept.itertuples(index=False):
        src = os.path.join(WAV_NORMALIZED_DIR, f"{row.source_id}.wav")
        dst = os.path.join(audio_out, row.file_name)
        if not os.path.exists(src):
            missing_wav += 1
            continue
        audio_present += 1
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied_new += 1

    metadata = kept[
        [
            "file_name",
            "speaker_id",
            "gender",
            "transcription",
            "quality_score",
            "duration",
            "source_id",
        ]
    ].copy()
    metadata.to_csv(metadata_csv, index=False)

    summary = (
        kept.groupby(["speaker_id", "gender"], as_index=False)
        .agg(
            clips=("source_id", "count"),
            hours=("duration", lambda x: float(x.sum()) / 3600.0),
            mean_quality=("quality_score", "mean"),
            mean_duration=("duration", "mean"),
        )
        .sort_values(["clips", "hours"], ascending=False)
    )
    summary["hours"] = summary["hours"].round(3)
    summary["mean_quality"] = summary["mean_quality"].round(2)
    summary["mean_duration"] = summary["mean_duration"].round(2)
    summary.to_csv(summary_csv, index=False)

    total_hours = float(kept["duration"].sum()) / 3600.0
    unique_speakers = int(kept["speaker_id"].nunique())

    hours_by_speaker = (
        kept.groupby("speaker_id")["duration"].sum().sort_values(ascending=False)
        / 3600.0
    )
    top1_share = float(hours_by_speaker.head(1).sum() / total_hours)
    top5_share = float(hours_by_speaker.head(5).sum() / total_hours)

    # Optional trace-back to matrix row in the audit report.
    matrix_match = None
    matrix_csv_path = audit.get("paths", {}).get("matrix_csv", "")
    if matrix_csv_path and os.path.exists(matrix_csv_path):
        matrix_df = pd.read_csv(matrix_csv_path)
        match_df = matrix_df[
            (matrix_df["speaker_pool"] == speaker_pool)
            & (matrix_df["quality_label"].astype(str).str.lower() == q_label)
            & (matrix_df["duration_ceiling"].astype(float) == float(duration_cap))
        ].copy()
        if not match_df.empty:
            matrix_match = match_df.iloc[0].to_dict()

    report = {
        "timestamp": datetime.now().isoformat(),
        "audit_json_path": audit_json_path,
        "source_metadata_csv": metadata_csv_path,
        "filters": {
            "speaker_pool": speaker_pool,
            "speaker_ids": selected_speakers,
            "quality_label": quality_label,
            "quality_floor": round(resolved_quality_floor, 4),
            "duration_cap_s": float(duration_cap),
        },
        "output": {
            "band_name": band_name,
            "root": export_root,
            "audio_dir": audio_out,
            "metadata_csv": metadata_csv,
            "speaker_summary_csv": summary_csv,
        },
        "totals": {
            "clips": int(len(kept)),
            "hours": round(total_hours, 3),
            "unique_speakers": unique_speakers,
            "mean_quality": round(float(kept["quality_score"].mean()), 2),
            "mean_duration_s": round(float(kept["duration"].mean()), 2),
            "top1_hours_share": round(top1_share, 3),
            "top5_hours_share": round(top5_share, 3),
            "copied_new_audio_files": int(copied_new),
            "audio_files_present": int(audio_present),
            "missing_normalized_wavs": int(missing_wav),
        },
        "audit_matrix_match": matrix_match,
        "speakers": summary.to_dict(orient="records"),
    }

    os.makedirs(export_root, exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    data_vol.commit()

    print("\nExport complete")
    print(f"  Clips    : {len(kept):,}")
    print(f"  Hours    : {total_hours:.3f}")
    print(f"  Speakers : {unique_speakers}")
    print(f"  Metadata : {metadata_csv}")
    print(f"  Summary  : {summary_csv}")
    print(f"  Report   : {report_json}")


@app.local_entrypoint()
def main(
    speaker_pool: str = "top_20",
    quality_label: str = "p50",
    duration_cap: float = 25.0,
    quality_floor: float = -1.0,
    output_name: str = "",
    audit_json_path: str = DEFAULT_AUDIT_JSON,
    output_root: str = EXPORTS_ROOT,
):
    export_tts_base.remote(
        speaker_pool=speaker_pool,
        quality_label=quality_label,
        duration_cap=duration_cap,
        quality_floor=quality_floor,
        output_name=output_name,
        audit_json_path=audit_json_path,
        output_root=output_root,
    )
