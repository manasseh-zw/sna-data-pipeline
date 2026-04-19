"""
Run a small Resemble Enhance audition pass for one exported speaker folder.

The speaker folder is expected to be mounted inside the Modal job as:
  /data/speakers/<speaker_export>/

Expected layout:
  /data/speakers/1_Female_full/
    audio/
    metadata.csv
    export_report.json

This script:
  1. Reads metadata.csv
  2. Deterministically samples N rows stratified across low/mid/high quality
  3. Runs Resemble Enhance on those files only
  4. Writes processed WAVs to audio_enhanced/ with the same filenames
  5. Writes a manifest CSV, failures CSV, and run report JSON

Example:
  modal run src/resemble_enhance_bakeoff.py \
      --speaker-dir /data/speakers/1_Female_full \
      --num-samples 50 \
      --seed 42
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("sna-resemble-enhance-bakeoff")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "numpy",
        "pandas",
        "resemble-enhance",
        "soundfile",
    )
)

DEFAULT_SPEAKER_DIR = "/data/speakers/1_Female_full"
DEFAULT_OUTPUT_SUBDIR = "audio_enhanced"
DEFAULT_NUM_SAMPLES = 50
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cuda"
DEFAULT_LAMBD = 0.4
DEFAULT_TAU = 0.5
DEFAULT_SOLVER = "midpoint"
DEFAULT_NFE = 64

MANIFEST_CSV_NAME = "resemble_enhance_manifest.csv"
FAILURES_CSV_NAME = "resemble_enhance_failures.csv"
REPORT_JSON_NAME = "resemble_enhance_report.json"

METADATA_COLUMNS = [
    "file_name",
    "transcription",
    "quality_score",
    "duration",
    "source_id",
]

SUCCESS_STATUSES = {"success", "reused_existing"}
MOUNT_PREFIX = "/data"


@dataclass(frozen=True)
class EnhanceConfig:
    speaker_dir: str
    output_subdir: str
    num_samples: int
    seed: int
    overwrite: bool
    device: str
    lambd: float
    tau: float
    solver: str
    nfe: int
    denoise_only: bool


def mounted_to_volume_path(path_str: str) -> str:
    path = Path(path_str)
    mount_root = Path(MOUNT_PREFIX)
    try:
        relative = path.relative_to(mount_root)
    except ValueError:
        return path_str
    return "/" + relative.as_posix()


def validate_metadata_columns(frame) -> None:
    missing = [column for column in METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing expected metadata columns: {missing}")


def _tier_counts(num_samples: int) -> dict[str, int]:
    base = num_samples // 3
    counts = {"low": base, "mid": base, "high": base}
    remainder = num_samples - (base * 3)
    for tier in ("low", "high", "mid"):
        if remainder <= 0:
            break
        counts[tier] += 1
        remainder -= 1
    return counts


def build_stratified_sample(frame, num_samples: int, seed: int):
    import pandas as pd

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if len(frame) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples but only {len(frame)} rows are available"
        )

    ordered = frame.sort_values(
        by=["quality_score", "duration", "file_name"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    first_cut = len(ordered) // 3
    second_cut = (2 * len(ordered)) // 3

    tiered = []
    slices = (
        ("low", ordered.iloc[:first_cut].copy()),
        ("mid", ordered.iloc[first_cut:second_cut].copy()),
        ("high", ordered.iloc[second_cut:].copy()),
    )
    counts = _tier_counts(num_samples)

    for offset, (tier, subset) in enumerate(slices):
        if subset.empty:
            raise RuntimeError(f"Quality tier '{tier}' is empty; cannot stratify sample.")
        take = counts[tier]
        if len(subset) < take:
            raise RuntimeError(
                f"Quality tier '{tier}' has only {len(subset)} rows; "
                f"cannot draw requested {take} samples."
            )

        sampled = subset.sample(n=take, random_state=seed + offset).copy()
        sampled["quality_tier"] = tier
        tiered.append(sampled)

    sampled_frame = pd.concat(tiered, ignore_index=True)
    sampled_frame = sampled_frame.sort_values(
        by=["file_name"],
        ascending=[True],
        kind="mergesort",
    ).reset_index(drop=True)
    sampled_frame["sample_id"] = range(1, len(sampled_frame) + 1)
    return sampled_frame


@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=14400,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_dotenv()],
)
def run_resemble_enhance_bakeoff(
    speaker_dir: str = DEFAULT_SPEAKER_DIR,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    overwrite: bool = False,
    device: str = DEFAULT_DEVICE,
    lambd: float = DEFAULT_LAMBD,
    tau: float = DEFAULT_TAU,
    solver: str = DEFAULT_SOLVER,
    nfe: int = DEFAULT_NFE,
    denoise_only: bool = False,
):
    import os
    import subprocess
    import tempfile
    import time

    import pandas as pd
    import soundfile as sf

    started_at = time.perf_counter()
    config = EnhanceConfig(
        speaker_dir=speaker_dir,
        output_subdir=output_subdir,
        num_samples=num_samples,
        seed=seed,
        overwrite=overwrite,
        device=device,
        lambd=lambd,
        tau=tau,
        solver=solver,
        nfe=nfe,
        denoise_only=denoise_only,
    )

    speaker_root = Path(speaker_dir)
    audio_dir = speaker_root / "audio"
    metadata_csv = speaker_root / "metadata.csv"
    output_dir = speaker_root / output_subdir
    manifest_csv = speaker_root / MANIFEST_CSV_NAME
    failures_csv = speaker_root / FAILURES_CSV_NAME
    report_json = speaker_root / REPORT_JSON_NAME

    print("=" * 72)
    print("SNA DATA PIPELINE - RESEMBLE ENHANCE BAKEOFF")
    print("=" * 72)
    print(f"  Speaker dir      : {speaker_root}")
    print(f"  Audio dir        : {audio_dir}")
    print(f"  Output dir       : {output_dir}")
    print(f"  Samples          : {num_samples}")
    print(f"  Seed             : {seed}")
    print(f"  Device           : {device}")
    print(f"  Denoise only     : {denoise_only}")
    print(f"  lambd / tau      : {lambd} / {tau}")
    print(f"  Solver / NFE     : {solver} / {nfe}")
    print(f"  Overwrite        : {overwrite}")
    print()

    if not speaker_root.is_dir():
        raise FileNotFoundError(f"Speaker folder not found: {speaker_root}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Missing audio directory: {audio_dir}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_csv}")

    os.environ.setdefault("HF_HOME", "/data/models/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/models/huggingface/hub")

    metadata = pd.read_csv(metadata_csv)
    validate_metadata_columns(metadata)
    metadata = metadata[METADATA_COLUMNS].copy()
    metadata["file_name"] = metadata["file_name"].astype(str)
    metadata["source_id"] = metadata["source_id"].astype(str)
    metadata["transcription"] = metadata["transcription"].fillna("").astype(str)
    metadata["quality_score"] = metadata["quality_score"].astype(float)
    metadata["duration"] = metadata["duration"].astype(float)

    sampled = build_stratified_sample(metadata, num_samples=num_samples, seed=seed)

    missing_audio = [
        file_name for file_name in sampled["file_name"].tolist() if not (audio_dir / file_name).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(
            "Sampled files missing from audio directory: "
            + ", ".join(missing_audio[:10])
            + (" ..." if len(missing_audio) > 10 else "")
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    total_audio_seconds = 0.0

    for index, row in enumerate(sampled.itertuples(index=False), start=1):
        source_path = audio_dir / row.file_name
        output_path = output_dir / row.file_name
        original_info = sf.info(str(source_path))
        original_duration = float(original_info.duration)
        total_audio_seconds += original_duration

        print(
            f"[{index:02d}/{len(sampled)}] {row.file_name} "
            f"(tier={row.quality_tier}, q={row.quality_score:.2f}, dur={original_duration:.2f}s)"
        )

        status = "success"
        error_message = ""
        process_stdout = ""
        process_stderr = ""
        enhanced_sample_rate = None
        enhanced_duration = None
        processing_seconds = 0.0

        run_started_at = time.perf_counter()
        try:
            if output_path.exists() and not overwrite:
                status = "reused_existing"
            else:
                with tempfile.TemporaryDirectory(prefix="resemble_bakeoff_") as temp_dir:
                    temp_root = Path(temp_dir)
                    in_dir = temp_root / "input"
                    out_dir = temp_root / "output"
                    in_dir.mkdir(parents=True, exist_ok=True)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    temp_input = in_dir / row.file_name
                    shutil.copy2(source_path, temp_input)

                    command = [
                        "resemble-enhance",
                        str(in_dir),
                        str(out_dir),
                        "--device",
                        device,
                        "--suffix",
                        ".wav",
                        "--lambd",
                        str(lambd),
                        "--tau",
                        str(tau),
                        "--solver",
                        solver,
                        "--nfe",
                        str(nfe),
                    ]
                    if denoise_only:
                        command.append("--denoise_only")

                    completed = subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    process_stdout = completed.stdout.strip()
                    process_stderr = completed.stderr.strip()

                    produced_path = out_dir / row.file_name
                    if not produced_path.is_file():
                        raise FileNotFoundError(
                            f"Resemble Enhance finished but did not produce {produced_path}"
                        )

                    shutil.copy2(produced_path, output_path)

            enhanced_info = sf.info(str(output_path))
            enhanced_sample_rate = int(enhanced_info.samplerate)
            enhanced_duration = float(enhanced_info.duration)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
        finally:
            processing_seconds = time.perf_counter() - run_started_at

        realtime_factor = (
            processing_seconds / original_duration if original_duration > 0 else None
        )

        records.append(
            {
                "sample_id": int(row.sample_id),
                "source_id": row.source_id,
                "file_name": row.file_name,
                "quality_tier": row.quality_tier,
                "quality_score": round(float(row.quality_score), 5),
                "duration_sec": round(float(row.duration), 5),
                "transcription": row.transcription,
                "input_path": str(source_path),
                "output_path": str(output_path),
                "status": status,
                "error": error_message,
                "original_sample_rate": int(original_info.samplerate),
                "original_duration_sec": round(original_duration, 5),
                "enhanced_sample_rate": enhanced_sample_rate,
                "enhanced_duration_sec": round(enhanced_duration, 5)
                if enhanced_duration is not None
                else None,
                "processing_seconds": round(processing_seconds, 5),
                "realtime_factor": round(realtime_factor, 5)
                if realtime_factor is not None
                else None,
                "denoise_only": denoise_only,
                "lambd": lambd,
                "tau": tau,
                "solver": solver,
                "nfe": nfe,
                "process_stdout": process_stdout,
                "process_stderr": process_stderr,
            }
        )

    results = pd.DataFrame(records)
    failures = results[results["status"] == "failed"].copy()

    manifest_tmp = manifest_csv.with_suffix(".csv.tmp")
    failures_tmp = failures_csv.with_suffix(".csv.tmp")
    report_tmp = report_json.with_suffix(".json.tmp")

    results.to_csv(manifest_tmp, index=False)
    failures.to_csv(failures_tmp, index=False)

    total_elapsed = time.perf_counter() - started_at
    successful = int(results["status"].isin(SUCCESS_STATUSES).sum())
    failed = int((results["status"] == "failed").sum())
    total_processed_audio = float(
        results.loc[results["status"].isin(SUCCESS_STATUSES), "original_duration_sec"].sum()
    )
    approximate_rtf = (
        float(results.loc[results["status"].isin(SUCCESS_STATUSES), "processing_seconds"].sum())
        / total_processed_audio
        if total_processed_audio > 0
        else None
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "speaker_dir_mounted": str(speaker_root),
        "speaker_dir_volume": mounted_to_volume_path(str(speaker_root)),
        "audio_dir_mounted": str(audio_dir),
        "audio_dir_volume": mounted_to_volume_path(str(audio_dir)),
        "output_dir_mounted": str(output_dir),
        "output_dir_volume": mounted_to_volume_path(str(output_dir)),
        "manifest_csv_mounted": str(manifest_csv),
        "manifest_csv_volume": mounted_to_volume_path(str(manifest_csv)),
        "failures_csv_mounted": str(failures_csv),
        "failures_csv_volume": mounted_to_volume_path(str(failures_csv)),
        "sampled_count": int(len(results)),
        "successful_count": successful,
        "failed_count": failed,
        "requested_audio_seconds": round(total_audio_seconds, 5),
        "processed_audio_seconds": round(total_processed_audio, 5),
        "total_processing_seconds": round(total_elapsed, 5),
        "approximate_realtime_factor": round(approximate_rtf, 5)
        if approximate_rtf is not None
        else None,
        "notes": {
            "conservative_defaults": {
                "lambd": lambd,
                "tau": tau,
                "solver": solver,
                "nfe": nfe,
                "denoise_only": denoise_only,
            },
            "path_mapping": "The job mounts the volume at /data, but Modal volume commands use the path without the /data prefix.",
        },
    }

    with report_tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    shutil.move(manifest_tmp, manifest_csv)
    shutil.move(failures_tmp, failures_csv)
    shutil.move(report_tmp, report_json)

    data_vol.commit()

    print()
    print("=" * 72)
    print("BAKEOFF COMPLETE")
    print("=" * 72)
    print(f"  Enhanced output dir : {output_dir}")
    print(f"  Manifest CSV        : {manifest_csv}")
    print(f"  Failures CSV        : {failures_csv}")
    print(f"  Report JSON         : {report_json}")
    print(f"  Successes           : {successful}")
    print(f"  Failures            : {failed}")
    print(f"  Total time (s)      : {total_elapsed:.2f}")
    if approximate_rtf is not None:
        print(f"  Approx. RTF         : {approximate_rtf:.3f}x")


@app.local_entrypoint()
def main(
    speaker_dir: str = DEFAULT_SPEAKER_DIR,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    overwrite: bool = False,
    device: str = DEFAULT_DEVICE,
    lambd: float = DEFAULT_LAMBD,
    tau: float = DEFAULT_TAU,
    solver: str = DEFAULT_SOLVER,
    nfe: int = DEFAULT_NFE,
    denoise_only: bool = False,
):
    run_resemble_enhance_bakeoff.remote(
        speaker_dir=speaker_dir,
        num_samples=num_samples,
        seed=seed,
        output_subdir=output_subdir,
        overwrite=overwrite,
        device=device,
        lambd=lambd,
        tau=tau,
        solver=solver,
        nfe=nfe,
        denoise_only=denoise_only,
    )
