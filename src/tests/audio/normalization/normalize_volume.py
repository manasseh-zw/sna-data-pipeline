"""
Local volume-normalization test harness.

Drop test clips in:
    src/tests/audio/normalization/input/

Run:
    python src/tests/audio/normalization/normalize_volume.py

Outputs:
    src/tests/audio/normalization/output/*.wav
    src/tests/audio/normalization/normalization_report.csv
    src/tests/audio/normalization/normalization_summary.json

Default target is -23 LUFS (EBU R128) to match the annotated rebuild plan.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_CSV = BASE_DIR / "normalization_report.csv"
SUMMARY_JSON = BASE_DIR / "normalization_summary.json"
SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".aif", ".aiff"}


@dataclass
class ClipResult:
    source_file: str
    output_file: Optional[str]
    sample_rate: Optional[int]
    duration_s: Optional[float]
    status: str
    input_lufs: Optional[float]
    output_lufs: Optional[float]
    gain_db: Optional[float]
    peak_in_dbfs: Optional[float]
    peak_out_dbfs: Optional[float]
    error: str


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return -120.0
    return float(20.0 * np.log10(peak))


def integrated_loudness(meter: pyln.Meter, audio: np.ndarray) -> Optional[float]:
    try:
        loudness = float(meter.integrated_loudness(audio))
    except Exception:
        return None
    if not np.isfinite(loudness):
        return None
    return loudness


def write_report_csv(rows: List[ClipResult], path: Path) -> None:
    import csv

    fieldnames = (
        list(asdict(rows[0]).keys())
        if rows
        else list(
            asdict(
                ClipResult(
                    source_file="",
                    output_file=None,
                    sample_rate=None,
                    duration_s=None,
                    status="",
                    input_lufs=None,
                    output_lufs=None,
                    gain_db=None,
                    peak_in_dbfs=None,
                    peak_out_dbfs=None,
                    error="",
                )
            ).keys()
        )
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(
    rows: List[ClipResult], target_lufs: float, tolerance_lu: float
) -> Dict[str, object]:
    normalized = [r for r in rows if r.status == "normalized"]
    skipped = [r for r in rows if r.status == "skipped_within_tolerance"]
    failed = [r for r in rows if r.status.startswith("failed")]

    output_lufs = [r.output_lufs for r in rows if r.output_lufs is not None]
    input_lufs = [r.input_lufs for r in rows if r.input_lufs is not None]

    return {
        "target_lufs": target_lufs,
        "tolerance_lu": tolerance_lu,
        "total_files": len(rows),
        "n_normalized": len(normalized),
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "input_lufs": {
            "mean": round(float(np.mean(input_lufs)), 2) if input_lufs else None,
            "min": round(float(np.min(input_lufs)), 2) if input_lufs else None,
            "max": round(float(np.max(input_lufs)), 2) if input_lufs else None,
        },
        "output_lufs": {
            "mean": round(float(np.mean(output_lufs)), 2) if output_lufs else None,
            "min": round(float(np.min(output_lufs)), 2) if output_lufs else None,
            "max": round(float(np.max(output_lufs)), 2) if output_lufs else None,
        },
    }


def run(target_lufs: float, tolerance_lu: float) -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        print(f"No supported audio files found in {INPUT_DIR}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}")
        return

    print("=" * 60)
    print("SNA LOCAL VOLUME NORMALIZATION TEST")
    print("=" * 60)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target loudness:  {target_lufs:.1f} LUFS")
    print(f"Skip tolerance:   +/-{tolerance_lu:.1f} LU")
    print(f"Files detected:   {len(files)}")

    meter_cache: Dict[int, pyln.Meter] = {}
    rows: List[ClipResult] = []

    iterator = tqdm(files, desc="Normalizing") if tqdm else files
    for in_path in iterator:
        output_name = f"{in_path.stem}_norm.wav"
        out_path = OUTPUT_DIR / output_name

        try:
            audio, sr = sf.read(str(in_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.nan_to_num(audio, copy=False)

            if audio.size == 0:
                raise ValueError("empty_audio")

            meter = meter_cache.setdefault(sr, pyln.Meter(sr))
            in_lufs = integrated_loudness(meter, audio)
            if in_lufs is None:
                sf.write(str(out_path), audio, sr)
                rows.append(
                    ClipResult(
                        source_file=in_path.name,
                        output_file=out_path.name,
                        sample_rate=sr,
                        duration_s=round(float(len(audio) / sr), 3),
                        status="failed_loudness_measurement",
                        input_lufs=None,
                        output_lufs=None,
                        gain_db=None,
                        peak_in_dbfs=round(peak_dbfs(audio), 2),
                        peak_out_dbfs=round(peak_dbfs(audio), 2),
                        error="integrated_loudness_not_finite",
                    )
                )
                continue

            gain_db = target_lufs - in_lufs
            if abs(gain_db) <= tolerance_lu:
                out_audio = audio
                status = "skipped_within_tolerance"
            else:
                out_audio = pyln.normalize.loudness(audio, in_lufs, target_lufs)
                out_audio = np.clip(out_audio, -1.0, 1.0).astype(np.float32)
                status = "normalized"

            out_lufs = integrated_loudness(meter, out_audio)
            sf.write(str(out_path), out_audio, sr)

            rows.append(
                ClipResult(
                    source_file=in_path.name,
                    output_file=out_path.name,
                    sample_rate=sr,
                    duration_s=round(float(len(audio) / sr), 3),
                    status=status,
                    input_lufs=round(in_lufs, 2),
                    output_lufs=round(out_lufs, 2) if out_lufs is not None else None,
                    gain_db=round(gain_db, 2),
                    peak_in_dbfs=round(peak_dbfs(audio), 2),
                    peak_out_dbfs=round(peak_dbfs(out_audio), 2),
                    error="",
                )
            )
        except Exception as exc:
            rows.append(
                ClipResult(
                    source_file=in_path.name,
                    output_file=None,
                    sample_rate=None,
                    duration_s=None,
                    status="failed_read_or_process",
                    input_lufs=None,
                    output_lufs=None,
                    gain_db=None,
                    peak_in_dbfs=None,
                    peak_out_dbfs=None,
                    error=str(exc),
                )
            )

    write_report_csv(rows, REPORT_CSV)
    summary = summarize(rows, target_lufs, tolerance_lu)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("Use EBU R128 integrated loudness normalization at -23 LUFS.")
    print("Keep clipping protection with np.clip(..., -1.0, 1.0) after gain.")
    print("For production rebuild, keep +/-1 LU skip tolerance for stable clips.")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  total_files   : {summary['total_files']}")
    print(f"  n_normalized  : {summary['n_normalized']}")
    print(f"  n_skipped     : {summary['n_skipped']}")
    print(f"  n_failed      : {summary['n_failed']}")
    print(f"  report_csv    : {REPORT_CSV}")
    print(f"  summary_json  : {SUMMARY_JSON}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local loudness normalization tests."
    )
    parser.add_argument(
        "--target-lufs", type=float, default=-23.0, help="Target integrated loudness."
    )
    parser.add_argument(
        "--tolerance-lu",
        type=float,
        default=1.0,
        help="Skip normalization when within +/- this range from target.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(target_lufs=args.target_lufs, tolerance_lu=args.tolerance_lu)
