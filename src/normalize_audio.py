import modal

app = modal.App("sna-normalize-audio")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "datasets[audio]",
        "librosa",
        "webrtcvad",
        "soundfile",
        "numpy",
        "pandas",
    )
)

TARGET_SR  = 24_000
VAD_SR     = 16_000
FRAME_MS   = 30
FRAME_LEN  = int(TARGET_SR * FRAME_MS / 1000)   # 720 samples
VAD_FRAME_LEN = int(VAD_SR * FRAME_MS / 1000)   # 480 samples
BUFFER_SEC = 0.4

MAX_INTERNAL_GAP_MS     = 150
TARGET_GAP_MS           = 90
MAX_INTERNAL_GAP_FRAMES = int(MAX_INTERNAL_GAP_MS / FRAME_MS)
TARGET_GAP_FRAMES       = int(TARGET_GAP_MS / FRAME_MS)

# Speakers whose entire contribution is blacklisted due to consistent audio quality issues.
# DVRNxPvJnmebFbLnQhG9VSCLhdf2: 185 clips, all distorted/mumbled - confirmed by manual review.
BLACKLISTED_SPEAKER_IDS = {
    "DVRNxPvJnmebFbLnQhG9VSCLhdf2",
}


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=7200,
    volumes={"/data": data_vol},
)
def normalize_audio():
    import json
    import os
    import shutil
    from datetime import datetime

    import librosa
    import numpy as np
    import pandas as pd
    import webrtcvad
    from datasets import Dataset, load_from_disk

    print("=" * 60)
    print("SNA DATA PIPELINE - PHASE 4: NORMALIZE AUDIO")
    print("=" * 60)

    print("\nLoading normalised dataset from /data/refined/...")
    ds = load_from_disk("/data/refined")
    print(f"   Loaded {len(ds)} rows")

    vad = webrtcvad.Vad(2)

    def compute_vad_mask(audio_24k):
        audio_vad = librosa.resample(audio_24k, orig_sr=TARGET_SR, target_sr=VAD_SR)
        usable_vad = (len(audio_vad) // VAD_FRAME_LEN) * VAD_FRAME_LEN
        if usable_vad < VAD_FRAME_LEN:
            return np.array([], dtype=bool)

        audio_vad = np.clip(audio_vad[:usable_vad], -1.0, 1.0)
        pcm_vad = (audio_vad * 32767).astype(np.int16).reshape(-1, VAD_FRAME_LEN)
        raw_mask = np.array(
            [vad.is_speech(frame.tobytes(), VAD_SR) for frame in pcm_vad],
            dtype=bool,
        )
        mask = smooth_vad_mask(raw_mask)

        target_frames = len(audio_24k) // FRAME_LEN
        if target_frames == 0:
            return np.array([], dtype=bool)
        if len(mask) > target_frames:
            return mask[:target_frames]
        if len(mask) < target_frames:
            return np.pad(mask, (0, target_frames - len(mask)), constant_values=False)
        return mask

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

    def trim_internal_gaps(audio_24k, speech_mask):
        n_frames = len(speech_mask)
        chunks = []
        i = 0

        while i < n_frames:
            if speech_mask[i]:
                run_start = i
                while i < n_frames and speech_mask[i]:
                    i += 1
                sample_start = run_start * FRAME_LEN
                sample_end = min(i * FRAME_LEN, len(audio_24k))
                chunks.append(audio_24k[sample_start:sample_end])
            else:
                gap_start = i
                while i < n_frames and not speech_mask[i]:
                    i += 1
                gap_frames = i - gap_start
                keep_frames = (
                    TARGET_GAP_FRAMES if gap_frames > MAX_INTERNAL_GAP_FRAMES else gap_frames
                )
                sample_start = gap_start * FRAME_LEN
                sample_end = min((gap_start + keep_frames) * FRAME_LEN, len(audio_24k))
                chunks.append(audio_24k[sample_start:sample_end])

        return np.concatenate(chunks) if chunks else audio_24k

    def compute_metrics(audio_24k, speech_mask):
        n_frames = len(speech_mask)
        usable = n_frames * FRAME_LEN
        audio_24k = audio_24k[:usable]

        frames = audio_24k.reshape(n_frames, FRAME_LEN)
        frame_powers = np.mean(frames.astype(np.float32) ** 2, axis=1)

        speech_powers = frame_powers[speech_mask]
        non_speech_powers = frame_powers[~speech_mask]

        if speech_powers.size == 0:
            return None

        duration_s = float(len(audio_24k)) / TARGET_SR
        speech_ratio = float(speech_powers.size / n_frames)

        if non_speech_powers.size >= 4:
            quiet_n = max(1, int(np.ceil(non_speech_powers.size * 0.2)))
            noise_floor = np.mean(np.partition(non_speech_powers, quiet_n - 1)[:quiet_n])
            penalty = 0.0
        else:
            quiet_n = max(1, int(np.ceil(n_frames * 0.1)))
            noise_floor = np.mean(np.partition(frame_powers, quiet_n - 1)[:quiet_n])
            penalty = 3.0

        signal_power = np.mean(speech_powers)
        snr_db = float(10.0 * np.log10((signal_power + 1e-10) / (noise_floor + 1e-10)))

        if speech_ratio < 0.35:
            penalty += (0.35 - speech_ratio) * 20.0
        if speech_ratio > 0.95:
            penalty += (speech_ratio - 0.95) * 40.0

        quality_score = float(snr_db - penalty)
        return quality_score, snr_db, speech_ratio, duration_s

    kept = []
    dropped = []
    drop_reasons = {
        "no_speech_detected": 0,
        "empty_after_trim": 0,
    }

    total = len(ds)
    print(f"\nProcessing {total} clips...\n")

    for idx_row, item in enumerate(ds):
        if idx_row % 500 == 0:
            print(f"   [{idx_row:>5}/{total}] kept={len(kept)}  dropped={len(dropped)}")

        if item.get("source_speaker_id") in BLACKLISTED_SPEAKER_IDS:
            drop_reasons.setdefault("blacklisted_speaker", 0)
            drop_reasons["blacklisted_speaker"] += 1
            dropped.append({"source_id": item["source_id"], "drop_reason": "blacklisted_speaker"})
            continue

        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.nan_to_num(audio, copy=False)

        audio_24k = (
            librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            if sr != TARGET_SR
            else audio.copy()
        )

        usable = (len(audio_24k) // FRAME_LEN) * FRAME_LEN
        audio_24k = np.clip(audio_24k[:usable], -1.0, 1.0)
        if usable < FRAME_LEN:
            drop_reasons["empty_after_trim"] += 1
            dropped.append(item["source_id"])
            continue

        mask = compute_vad_mask(audio_24k)

        if not mask.any():
            drop_reasons["no_speech_detected"] += 1
            dropped.append(item["source_id"])
            continue

        speech_idx = np.where(mask)[0]
        frame_sec = FRAME_MS / 1000.0

        start_sec = max(0.0, speech_idx[0] * frame_sec - BUFFER_SEC)
        end_sec = min(
            len(audio_24k) / TARGET_SR,
            (speech_idx[-1] + 1) * frame_sec + BUFFER_SEC,
        )

        start_sample = int(start_sec * TARGET_SR)
        end_sample = int(end_sec * TARGET_SR)
        audio_trimmed = audio_24k[start_sample:end_sample]

        if len(audio_trimmed) == 0:
            drop_reasons["empty_after_trim"] += 1
            dropped.append(item["source_id"])
            continue

        start_frame = int(start_sec / frame_sec)
        end_frame = min(len(mask), int(end_sec / frame_sec) + 1)
        mask_trimmed = mask[start_frame:end_frame]

        audio_trimmed = trim_internal_gaps(audio_trimmed, mask_trimmed)

        usable2 = (len(audio_trimmed) // FRAME_LEN) * FRAME_LEN
        audio_final = np.clip(audio_trimmed[:usable2], -1.0, 1.0)
        if usable2 < FRAME_LEN:
            drop_reasons["empty_after_trim"] += 1
            dropped.append(item["source_id"])
            continue

        mask_final = compute_vad_mask(audio_final)

        metrics = compute_metrics(audio_final, mask_final)
        if metrics is None:
            drop_reasons["no_speech_detected"] += 1
            dropped.append(item["source_id"])
            continue

        quality_score, snr_db, speech_ratio, duration_s = metrics

        kept.append(
            {
                "audio": {"array": audio_final, "sampling_rate": TARGET_SR},
                "transcription": item["transcription"],
                "source_id": item["source_id"],
                "source_speaker_id": item["source_speaker_id"],
                "speaker_idx": item["speaker_idx"],
                "speaker_clip_count": item["speaker_clip_count"],
                "language": item["language"],
                "gender": item["gender"],
                "has_punctuation": item["has_punctuation"],
                "snr_db": snr_db,
                "speech_ratio": speech_ratio,
                "quality_score": quality_score,
                "duration": duration_s,
            }
        )

    print("\nProcessing complete.")
    print(f"   Kept:    {len(kept)}")
    print(f"   Dropped: {len(dropped)}")
    for reason, count in drop_reasons.items():
        print(f"     {reason:<30} {count}")

    print("\nBuilding Dataset object...")
    refined_ds = Dataset.from_list(kept)

    snr_arr = np.array(refined_ds["snr_db"])
    ratio_arr = np.array(refined_ds["speech_ratio"])
    duration_arr = np.array(refined_ds["duration"])

    total_speech_hours = float(duration_arr.sum()) / 3600.0

    dur_buckets = {
        "under_5s": int((duration_arr < 5).sum()),
        "5s_to_10s": int(((duration_arr >= 5) & (duration_arr < 10)).sum()),
        "10s_to_15s": int(((duration_arr >= 10) & (duration_arr < 15)).sum()),
        "15s_to_20s": int(((duration_arr >= 15) & (duration_arr < 20)).sum()),
        "over_20s": int((duration_arr >= 20).sum()),
    }

    audit = {
        "phase": "normalize_audio",
        "timestamp": datetime.now().isoformat(),
        "input_rows": total,
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "drop_rate_pct": round(len(dropped) / total * 100, 2),
        "drop_reasons": drop_reasons,
        "total_hours": round(total_speech_hours, 3),
        "vad_config": {
            "aggressiveness": 2,
            "frame_ms": FRAME_MS,
            "buffer_sec": BUFFER_SEC,
            "max_internal_gap_ms": MAX_INTERNAL_GAP_MS,
            "target_gap_ms": TARGET_GAP_MS,
        },
        "snr_db": {
            "mean": round(float(snr_arr.mean()), 2),
            "std": round(float(snr_arr.std()), 2),
            "min": round(float(snr_arr.min()), 2),
            "max": round(float(snr_arr.max()), 2),
        },
        "speech_ratio": {
            "mean": round(float(ratio_arr.mean()), 3),
            "std": round(float(ratio_arr.std()), 3),
            "min": round(float(ratio_arr.min()), 3),
            "max": round(float(ratio_arr.max()), 3),
        },
        "duration_distribution": dur_buckets,
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/04_normalize_audio_audit.json"
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

    print(f"Saving normalized dataset to temporary path -> {temp_path}")
    refined_ds.save_to_disk(temp_path)

    print("Promoting temporary dataset into place...")
    if os.path.exists(target_path):
        os.replace(target_path, backup_path)
    os.replace(temp_path, target_path)

    if os.path.exists(backup_path):
        print(f"Deleting previous dataset backup -> {backup_path}")
        shutil.rmtree(backup_path)

    data_vol.commit()

    print("\n" + "=" * 60)
    print("NORMALIZE AUDIO COMPLETE")
    print(f"   {len(kept)} clips kept from {total} input rows")
    print(f"   {round(total_speech_hours, 2)} hours retained")
    print(f"   Audit -> /data/reports/04_normalize_audio_audit.json")
    print("=" * 60)


@app.local_entrypoint()
def main():
    normalize_audio.remote()
