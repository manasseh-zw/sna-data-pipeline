"""
Local mic-pop audit over normalization test inputs.

Scans the start of each clip for transient spike patterns typical of
mic-on pops/clicks and writes per-file + aggregate reports.

Run from repo root:
    python src/tests/audio/normalization/mic_pop_audit.py

Outputs:
    src/tests/audio/normalization/mic_pop_report.csv
    src/tests/audio/normalization/mic_pop_summary.json
    src/tests/audio/normalization/mic_pop_flagged.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
REPORT_CSV = BASE_DIR / "mic_pop_report.csv"
SUMMARY_JSON = BASE_DIR / "mic_pop_summary.json"
FLAGGED_TXT = BASE_DIR / "mic_pop_flagged.txt"
SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".aif", ".aiff"}


@dataclass
class AuditRow:
    source_file: str
    sample_rate: int
    duration_s: float
    status: str
    flagged_mic_pop: bool
    peak_start_dbfs: float
    peak_pos_ms: float
    peak_to_rms_db: float
    decay_db: float
    reason: str


def amp_to_db(x: float, floor: float = 1e-8) -> float:
    return float(20.0 * np.log10(max(float(x), floor)))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def ms_to_samples(ms: float, sr: int) -> int:
    return max(1, int(round(ms / 1000.0 * sr)))


def detect_mic_pop(
    audio: np.ndarray,
    sr: int,
    peak_window_ms: float,
    early_peak_max_ms: float,
    follow_start_ms: float,
    follow_end_ms: float,
    decay_early_ms: float,
    decay_late_start_ms: float,
    decay_late_end_ms: float,
    peak_threshold_dbfs: float,
    peak_to_rms_threshold_db: float,
    decay_threshold_db: float,
) -> Dict[str, object]:
    n_peak = ms_to_samples(peak_window_ms, sr)
    n_early_peak_max = ms_to_samples(early_peak_max_ms, sr)

    i_follow_start = ms_to_samples(follow_start_ms, sr)
    i_follow_end = ms_to_samples(follow_end_ms, sr)

    i_decay_early = ms_to_samples(decay_early_ms, sr)
    i_decay_late_start = ms_to_samples(decay_late_start_ms, sr)
    i_decay_late_end = ms_to_samples(decay_late_end_ms, sr)

    needed = max(n_peak, i_follow_end, i_decay_late_end)
    if audio.size < needed:
        return {
            "status": "failed_too_short",
            "flagged": False,
            "peak_start_dbfs": -120.0,
            "peak_pos_ms": 0.0,
            "peak_to_rms_db": 0.0,
            "decay_db": 0.0,
            "reason": "clip_too_short_for_windows",
        }

    head = np.abs(audio[:n_peak])
    peak_idx = int(np.argmax(head))
    peak_amp = float(head[peak_idx])
    peak_start_dbfs = amp_to_db(peak_amp)
    peak_pos_ms = float(peak_idx / sr * 1000.0)

    follow = audio[i_follow_start:i_follow_end]
    follow_rms = rms(follow)
    peak_to_rms_db = peak_start_dbfs - amp_to_db(follow_rms)

    early = audio[:i_decay_early]
    late = audio[i_decay_late_start:i_decay_late_end]
    early_rms_db = amp_to_db(rms(early))
    late_rms_db = amp_to_db(rms(late))
    decay_db = early_rms_db - late_rms_db

    c1 = peak_start_dbfs >= peak_threshold_dbfs
    c2 = peak_idx <= n_early_peak_max
    c3 = peak_to_rms_db >= peak_to_rms_threshold_db
    c4 = decay_db >= decay_threshold_db

    flagged = bool(c1 and c2 and c3 and c4)

    reasons = []
    if not c1:
        reasons.append("peak_below_threshold")
    if not c2:
        reasons.append("peak_not_early")
    if not c3:
        reasons.append("insufficient_peak_to_rms")
    if not c4:
        reasons.append("insufficient_decay")

    return {
        "status": "ok",
        "flagged": flagged,
        "peak_start_dbfs": round(peak_start_dbfs, 2),
        "peak_pos_ms": round(peak_pos_ms, 3),
        "peak_to_rms_db": round(peak_to_rms_db, 2),
        "decay_db": round(decay_db, 2),
        "reason": "|".join(reasons) if reasons else "all_rules_passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit likely mic pops at clip starts."
    )
    parser.add_argument("--peak-window-ms", type=float, default=25.0)
    parser.add_argument("--early-peak-max-ms", type=float, default=12.0)
    parser.add_argument("--follow-start-ms", type=float, default=20.0)
    parser.add_argument("--follow-end-ms", type=float, default=120.0)
    parser.add_argument("--decay-early-ms", type=float, default=10.0)
    parser.add_argument("--decay-late-start-ms", type=float, default=30.0)
    parser.add_argument("--decay-late-end-ms", type=float, default=100.0)
    parser.add_argument("--peak-threshold-dbfs", type=float, default=-3.0)
    parser.add_argument("--peak-to-rms-threshold-db", type=float, default=12.0)
    parser.add_argument("--decay-threshold-db", type=float, default=8.0)
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    if not files:
        print(f"No supported audio files found in {INPUT_DIR}")
        return

    print("=" * 60)
    print("SNA MIC-POP AUDIT")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Files detected : {len(files)}")
    print("Rules:")
    print(
        "  peak>=%.1f dBFS, early_peak<=%.1f ms, peak_to_rms>=%.1f dB, decay>=%.1f dB"
        % (
            args.peak_threshold_dbfs,
            args.early_peak_max_ms,
            args.peak_to_rms_threshold_db,
            args.decay_threshold_db,
        )
    )

    rows: List[AuditRow] = []
    iterator = tqdm(files, desc="Auditing") if tqdm else files

    for path in iterator:
        try:
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.nan_to_num(audio, copy=False)

            result = detect_mic_pop(
                audio=audio,
                sr=sr,
                peak_window_ms=args.peak_window_ms,
                early_peak_max_ms=args.early_peak_max_ms,
                follow_start_ms=args.follow_start_ms,
                follow_end_ms=args.follow_end_ms,
                decay_early_ms=args.decay_early_ms,
                decay_late_start_ms=args.decay_late_start_ms,
                decay_late_end_ms=args.decay_late_end_ms,
                peak_threshold_dbfs=args.peak_threshold_dbfs,
                peak_to_rms_threshold_db=args.peak_to_rms_threshold_db,
                decay_threshold_db=args.decay_threshold_db,
            )

            rows.append(
                AuditRow(
                    source_file=path.name,
                    sample_rate=int(sr),
                    duration_s=round(float(audio.size / sr), 3),
                    status=str(result["status"]),
                    flagged_mic_pop=bool(result["flagged"]),
                    peak_start_dbfs=float(result["peak_start_dbfs"]),
                    peak_pos_ms=float(result["peak_pos_ms"]),
                    peak_to_rms_db=float(result["peak_to_rms_db"]),
                    decay_db=float(result["decay_db"]),
                    reason=str(result["reason"]),
                )
            )
        except Exception as exc:
            rows.append(
                AuditRow(
                    source_file=path.name,
                    sample_rate=0,
                    duration_s=0.0,
                    status="failed_read_or_process",
                    flagged_mic_pop=False,
                    peak_start_dbfs=-120.0,
                    peak_pos_ms=0.0,
                    peak_to_rms_db=0.0,
                    decay_db=0.0,
                    reason=str(exc),
                )
            )

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    valid = [r for r in rows if r.status == "ok"]
    flagged = [r for r in valid if r.flagged_mic_pop]
    failed = [r for r in rows if r.status != "ok"]

    flagged_pct = (len(flagged) / len(valid) * 100.0) if valid else 0.0

    summary = {
        "phase": "mic_pop_audit_local",
        "input_dir": str(INPUT_DIR),
        "total_files": len(rows),
        "valid_files": len(valid),
        "failed_files": len(failed),
        "flagged_files": len(flagged),
        "flagged_pct_of_valid": round(flagged_pct, 2),
        "thresholds": {
            "peak_window_ms": args.peak_window_ms,
            "early_peak_max_ms": args.early_peak_max_ms,
            "follow_start_ms": args.follow_start_ms,
            "follow_end_ms": args.follow_end_ms,
            "decay_early_ms": args.decay_early_ms,
            "decay_late_start_ms": args.decay_late_start_ms,
            "decay_late_end_ms": args.decay_late_end_ms,
            "peak_threshold_dbfs": args.peak_threshold_dbfs,
            "peak_to_rms_threshold_db": args.peak_to_rms_threshold_db,
            "decay_threshold_db": args.decay_threshold_db,
        },
        "suggestion": (
            "leave_as_is"
            if flagged_pct < 5.0
            else "consider_conditional_fade_for_flagged"
            if flagged_pct < 10.0
            else "consider_global_fade_in"
        ),
        "flagged_files": [r.source_file for r in flagged],
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with FLAGGED_TXT.open("w", encoding="utf-8") as f:
        for row in flagged:
            f.write(f"{row.source_file}\n")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  total_files : {summary['total_files']}")
    print(f"  valid_files : {summary['valid_files']}")
    print(f"  failed_files: {summary['failed_files']}")
    print(
        f"  flagged     : {summary['flagged_files']} ({summary['flagged_pct_of_valid']}%)"
    )
    print(f"  suggestion  : {summary['suggestion']}")
    print(f"  report_csv  : {REPORT_CSV}")
    print(f"  summary_json: {SUMMARY_JSON}")
    print(f"  flagged_txt : {FLAGGED_TXT}")


if __name__ == "__main__":
    main()
