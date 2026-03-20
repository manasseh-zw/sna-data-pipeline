"""
Local audio-phase test — runs the same VAD + trimming + gap-trimming logic
as normalize_audio.py over the sampled raw WAVs in src/tests/audio/samples/.

Outputs:
  samples/audit.csv           — per-clip metrics
  samples/audit_summary.json  — aggregate stats

Run from repo root:
    pip install librosa webrtcvad soundfile numpy pandas
    python src/tests/audio/test_curate.py
"""

import csv
import json
import os
import re

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import webrtcvad  # installed via webrtcvad-wheels

# ── Config (keep in sync with normalize_audio.py) ─────────────────────────────
TARGET_SR               = 24_000
FRAME_MS                = 30
FRAME_LEN               = int(TARGET_SR * FRAME_MS / 1000)
BUFFER_SEC              = 0.4
MAX_INTERNAL_GAP_MS     = 150
TARGET_GAP_MS           = 80
MAX_INTERNAL_GAP_FRAMES = int(MAX_INTERNAL_GAP_MS / FRAME_MS)
TARGET_GAP_FRAMES       = int(TARGET_GAP_MS / FRAME_MS)

SAMPLES_DIR  = "src/tests/audio/samples"
METADATA_CSV = os.path.join(SAMPLES_DIR, "metadata.csv")
AUDIT_CSV    = os.path.join(SAMPLES_DIR, "audit.csv")
AUDIT_JSON   = os.path.join(SAMPLES_DIR, "audit_summary.json")


# ── VAD helpers (identical to normalize_audio.py) ─────────────────────────────
def smooth_vad_mask(mask, min_speech_frames=3, bridge_gap_frames=2):
    mask = mask.copy()
    if mask.size == 0:
        return mask

    start = None
    runs = []
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))

    for rs, re in runs:
        if (re - rs) < min_speech_frames:
            mask[rs:re] = False

    idx = np.where(mask)[0]
    if idx.size == 0:
        return mask
    for i in range(len(idx) - 1):
        gap = idx[i + 1] - idx[i] - 1
        if 0 < gap <= bridge_gap_frames:
            mask[idx[i] + 1 : idx[i + 1]] = True

    return mask


def trim_internal_gaps(audio_16k, speech_mask):
    n_frames = len(speech_mask)
    chunks = []
    i = 0

    while i < n_frames:
        if speech_mask[i]:
            run_start = i
            while i < n_frames and speech_mask[i]:
                i += 1
            chunks.append(audio_16k[run_start * FRAME_LEN : min(i * FRAME_LEN, len(audio_16k))])
        else:
            gap_start = i
            while i < n_frames and not speech_mask[i]:
                i += 1
            gap_frames = i - gap_start
            keep_frames = TARGET_GAP_FRAMES if gap_frames > MAX_INTERNAL_GAP_FRAMES else gap_frames
            chunks.append(audio_16k[gap_start * FRAME_LEN : min((gap_start + keep_frames) * FRAME_LEN, len(audio_16k))])

    return np.concatenate(chunks) if chunks else audio_16k


def compute_metrics(audio_16k, speech_mask):
    n_frames  = len(speech_mask)
    usable    = n_frames * FRAME_LEN
    audio_16k = audio_16k[:usable]

    frames       = audio_16k.reshape(n_frames, FRAME_LEN)
    frame_powers = np.mean(frames.astype(np.float32) ** 2, axis=1)

    speech_powers     = frame_powers[speech_mask]
    non_speech_powers = frame_powers[~speech_mask]

    if speech_powers.size == 0:
        return None

    duration_s   = float(len(audio_16k)) / TARGET_SR
    speech_ratio = float(speech_powers.size / n_frames)

    if non_speech_powers.size >= 4:
        quiet_n     = max(1, int(np.ceil(non_speech_powers.size * 0.2)))
        noise_floor = np.mean(np.partition(non_speech_powers, quiet_n - 1)[:quiet_n])
        penalty     = 0.0
    else:
        quiet_n     = max(1, int(np.ceil(n_frames * 0.1)))
        noise_floor = np.mean(np.partition(frame_powers, quiet_n - 1)[:quiet_n])
        penalty     = 3.0

    signal_power  = np.mean(speech_powers)
    snr_db        = float(10.0 * np.log10((signal_power + 1e-10) / (noise_floor + 1e-10)))

    if speech_ratio < 0.35:
        penalty += (0.35 - speech_ratio) * 20.0
    if speech_ratio > 0.95:
        penalty += (speech_ratio - 0.95) * 40.0

    quality_score = float(snr_db - penalty)
    return quality_score, snr_db, speech_ratio, duration_s


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    metadata = pd.read_csv(METADATA_CSV)

    raw_files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith("_raw.wav"))
    total = len(raw_files)

    print("=" * 60)
    print("SNA NORMALIZE AUDIO TEST — LOCAL RUN")
    print("=" * 60)
    print(f"\n   {total} raw clips found in {SAMPLES_DIR}")

    vad = webrtcvad.Vad(2)

    audit_rows = []
    drop_reasons = {"no_speech_detected": 0, "empty_after_trim": 0}
    kept = 0

    for i, fname in enumerate(raw_files):
        source_id = re.search(r"sna_(.+)_raw\.wav", fname).group(1)
        raw_path  = os.path.join(SAMPLES_DIR, fname)

        audio, sr = sf.read(raw_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.nan_to_num(audio, copy=False)
        duration_raw = len(audio) / sr

        audio_24k = (
            librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            if sr != TARGET_SR else audio.copy()
        )

        usable    = (len(audio_24k) // FRAME_LEN) * FRAME_LEN
        audio_24k = np.clip(audio_24k[:usable], -1.0, 1.0)

        drop_reason = None

        if usable < FRAME_LEN:
            drop_reason = "empty_after_trim"
        else:
            pcm      = (audio_24k * 32767).astype(np.int16).reshape(-1, FRAME_LEN)
            raw_mask = np.array([vad.is_speech(f.tobytes(), TARGET_SR) for f in pcm], dtype=bool)
            mask     = smooth_vad_mask(raw_mask)

            if not mask.any():
                drop_reason = "no_speech_detected"
            else:
                speech_idx = np.where(mask)[0]
                frame_sec  = FRAME_MS / 1000.0

                start_sec     = max(0.0, speech_idx[0] * frame_sec - BUFFER_SEC)
                end_sec       = min(len(audio_24k) / TARGET_SR, (speech_idx[-1] + 1) * frame_sec + BUFFER_SEC)
                audio_trimmed = audio_24k[int(start_sec * TARGET_SR) : int(end_sec * TARGET_SR)]

                if len(audio_trimmed) == 0:
                    drop_reason = "empty_after_trim"
                else:
                    start_frame  = int(start_sec / frame_sec)
                    end_frame    = min(len(mask), int(end_sec / frame_sec) + 1)
                    mask_trimmed = mask[start_frame:end_frame]
                    audio_trimmed = trim_internal_gaps(audio_trimmed, mask_trimmed)

                    usable2     = (len(audio_trimmed) // FRAME_LEN) * FRAME_LEN
                    audio_final = np.clip(audio_trimmed[:usable2], -1.0, 1.0)

                    if usable2 < FRAME_LEN:
                        drop_reason = "empty_after_trim"
                    else:
                        pcm2      = (audio_final * 32767).astype(np.int16).reshape(-1, FRAME_LEN)
                        raw_mask2 = np.array([vad.is_speech(f.tobytes(), TARGET_SR) for f in pcm2], dtype=bool)
                        mask_final = smooth_vad_mask(raw_mask2)

                        metrics = compute_metrics(audio_final, mask_final)
                        if metrics is None:
                            drop_reason = "no_speech_detected"
                        else:
                            quality_score, snr_db, speech_ratio, duration_refined = metrics

                            kept += 1

                            duration_change_pct = round((duration_refined - duration_raw) / duration_raw * 100, 1)
                            audit_rows.append({
                                "source_id":           source_id,
                                "status":              "kept",
                                "duration_raw_s":      round(duration_raw, 3),
                                "duration_refined_s":  round(duration_refined, 3),
                                "duration_change_pct": duration_change_pct,
                                "snr_db":              round(snr_db, 2),
                                "speech_ratio":        round(speech_ratio, 3),
                                "quality_score":       round(quality_score, 2),
                            })

        if drop_reason:
            drop_reasons[drop_reason] += 1
            audit_rows.append({
                "source_id":           source_id,
                "status":              f"dropped:{drop_reason}",
                "duration_raw_s":      round(duration_raw, 3),
                "duration_refined_s":  None,
                "duration_change_pct": None,
                "snr_db":              None,
                "speech_ratio":        None,
                "quality_score":       None,
            })

        if (i + 1) % 50 == 0:
            print(f"   [{i+1}/{total}]  kept={kept}  dropped={sum(drop_reasons.values())}")

    # ── Write audit CSV ───────────────────────────────────────────────────────
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)

    # ── Compute summary ───────────────────────────────────────────────────────
    kept_rows    = [r for r in audit_rows if r["status"] == "kept"]
    dropped_rows = [r for r in audit_rows if r["status"] != "kept"]

    dur_raw_arr      = np.array([r["duration_raw_s"]     for r in kept_rows])
    dur_refined_arr  = np.array([r["duration_refined_s"] for r in kept_rows])
    snr_arr          = np.array([r["snr_db"]             for r in kept_rows])
    ratio_arr        = np.array([r["speech_ratio"]       for r in kept_rows])
    change_arr       = np.array([r["duration_change_pct"]for r in kept_rows])

    summary = {
        "total_sampled":        total,
        "kept":                 len(kept_rows),
        "dropped":              len(dropped_rows),
        "drop_reasons":         drop_reasons,
        "duration_raw_s":       {"mean": round(float(dur_raw_arr.mean()), 2),     "min": round(float(dur_raw_arr.min()), 2),     "max": round(float(dur_raw_arr.max()), 2)},
        "duration_refined_s":   {"mean": round(float(dur_refined_arr.mean()), 2), "min": round(float(dur_refined_arr.min()), 2), "max": round(float(dur_refined_arr.max()), 2)},
        "duration_change_pct":  {"mean": round(float(change_arr.mean()), 1),      "min": round(float(change_arr.min()), 1),      "max": round(float(change_arr.max()), 1)},
        "snr_db":               {"mean": round(float(snr_arr.mean()), 2),         "min": round(float(snr_arr.min()), 2),         "max": round(float(snr_arr.max()), 2)},
        "speech_ratio":         {"mean": round(float(ratio_arr.mean()), 3),       "min": round(float(ratio_arr.min()), 3),       "max": round(float(ratio_arr.max()), 3)},
    }

    with open(AUDIT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Kept:    {len(kept_rows)} / {total}")
    print(f"  Dropped: {len(dropped_rows)}")
    for r, c in drop_reasons.items():
        print(f"    {r:<28} {c}")
    print(f"\n  Duration (raw)     mean={summary['duration_raw_s']['mean']}s  min={summary['duration_raw_s']['min']}s  max={summary['duration_raw_s']['max']}s")
    print(f"  Duration (refined) mean={summary['duration_refined_s']['mean']}s  min={summary['duration_refined_s']['min']}s  max={summary['duration_refined_s']['max']}s")
    print(f"  Duration change    mean={summary['duration_change_pct']['mean']}%  min={summary['duration_change_pct']['min']}%  max={summary['duration_change_pct']['max']}%")
    print(f"\n  SNR       mean={summary['snr_db']['mean']} dB")
    print(f"  Speech ratio mean={summary['speech_ratio']['mean']}")
    print(f"  Audit CSV  → {AUDIT_CSV}")
    print(f"  Audit JSON → {AUDIT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
