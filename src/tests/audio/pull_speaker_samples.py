"""
Modal script — pulls stratified quality samples for the top N speakers
by total effective speech time from /data/final/ (DatasetDict with
train / validation / test splits).

All three splits are concatenated before ranking and sampling so that
speaker totals reflect their true representation across the full dataset.

Stratification splits each speaker's clips into bottom / middle / top
thirds by quality_score, then samples evenly across the three strata so
the sampled clips are representative of the *full* quality distribution
for that speaker — not just their best or worst material.

Folder layout written to the volume:
  /data/speaker_samples/
    <speaker_id>/
      metadata.csv          — per-clip stats for all sampled clips
      sna_<source_id>.wav   — decoded 24 kHz mono WAV
    summary.csv             — flat file across all speakers and clips

Archive:
  /data/speaker_samples.zip

Download after running (from repo root):
  modal volume get sna-data-vol /data/speaker_samples.zip src/tests/audio/speaker_samples.zip
  cd src/tests/audio && unzip -o speaker_samples.zip -d speaker_samples/
"""

import modal

app = modal.App("sna-pull-speaker-samples")

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

TOP_N_SPEAKERS    = 39   # how many speakers to sample from
CLIPS_PER_SPEAKER = 20   # total clips pulled per speaker
BOTTOM_SHARE      = 6    # clips from bottom third  (quality floor)
MIDDLE_SHARE      = 8    # clips from middle third  (typical quality)
TOP_SHARE         = 6    # clips from top third     (quality ceiling)
# BOTTOM_SHARE + MIDDLE_SHARE + TOP_SHARE must equal CLIPS_PER_SPEAKER
assert BOTTOM_SHARE + MIDDLE_SHARE + TOP_SHARE == CLIPS_PER_SPEAKER

RANDOM_SEED = 42


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    timeout=3600,
    volumes={"/data": data_vol},
)
def pull_speaker_samples():
    import io
    import os
    import pickle
    import random
    import shutil
    import time
    import zipfile
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from datasets import concatenate_datasets, load_from_disk

    print("=" * 60)
    print("SNA — PULL STRATIFIED SPEAKER SAMPLES")
    print("=" * 60)

    # ── 1. Load DatasetDict and concatenate all splits ────────────────────────
    print("\n📂 Loading DatasetDict from /data/final/ ...")
    dataset_dict = load_from_disk("/data/final")
    print(f"   Splits found: {list(dataset_dict.keys())}")
    for split, ds in dataset_dict.items():
        print(f"   {split:<12} {len(ds):>6} rows")

    # Concatenate all splits — we want speaker totals across the full dataset
    ds_full = concatenate_datasets([dataset_dict[s] for s in dataset_dict.keys()])
    total_rows = len(ds_full)
    print(f"\n   Combined: {total_rows} rows total")

    def decode_audio_field(audio_value):
        if audio_value is None:
            raise ValueError("audio field is None")

        if isinstance(audio_value, dict):
            arr = audio_value.get("array", None)
            sr = audio_value.get("sampling_rate", None)
            if arr is not None and sr:
                return np.asarray(arr), int(sr)

            raw = audio_value.get("bytes", None)
            if raw is not None:
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                elif isinstance(raw, bytearray):
                    raw = bytes(raw)
                return sf.read(io.BytesIO(raw))

            path = audio_value.get("path", None)
            if path:
                return sf.read(path)

        if isinstance(audio_value, (str, os.PathLike)):
            return sf.read(audio_value)

        raise ValueError(f"Unsupported audio field type: {type(audio_value)}")

    def audio_duration_seconds(audio_value):
        if audio_value is None:
            return 0.0

        try:
            if isinstance(audio_value, dict):
                arr = audio_value.get("array", None)
                sr = audio_value.get("sampling_rate", None)
                if arr is not None and sr:
                    return float(len(arr) / int(sr))

                raw = audio_value.get("bytes", None)
                if raw is not None:
                    if isinstance(raw, memoryview):
                        raw = raw.tobytes()
                    elif isinstance(raw, bytearray):
                        raw = bytes(raw)
                    return float(sf.info(io.BytesIO(raw)).duration)

                path = audio_value.get("path", None)
                if path:
                    return float(sf.info(path).duration)

            if isinstance(audio_value, (str, os.PathLike)):
                return float(sf.info(audio_value).duration)
        except Exception:
            pass

        try:
            arr, sr = decode_audio_field(audio_value)
            return float(len(arr) / sr)
        except Exception:
            return 0.0

    # ── 2. Build per-speaker index in one pass ────────────────────────────────
    print("\n📊 Indexing speakers ...")
    speaker_clips = defaultdict(list)
    indexing_started = time.time()
    checkpoint_path = "/data/speaker_samples_index_checkpoint.pkl"
    checkpoint_every_rows = 500
    start_idx = 0

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            start_idx = int(ckpt.get("next_idx", 0))
            speaker_clips = defaultdict(list, ckpt.get("speaker_clips", {}))
            print(
                f"   ↻ Resuming from checkpoint at row {start_idx}/{total_rows} "
                f"({len(speaker_clips)} speakers indexed)"
            )
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint, starting fresh: {e}")
            start_idx = 0
            speaker_clips = defaultdict(list)

    for i in range(start_idx, total_rows):
        item = ds_full[i]
        if i % 1000 == 0:
            elapsed = time.time() - indexing_started
            rows_done = (i - start_idx) + 1
            rate = rows_done / elapsed if elapsed > 0 else 0.0
            rows_left = total_rows - (i + 1)
            eta = rows_left / rate if rate > 0 else 0.0
            print(f"   [{i:>6}/{total_rows}]  {rate:>7.1f} rows/s  eta={eta:>6.1f}s")

        spk = item["source_speaker_id"]

        # Prefer stored duration column — avoids decoding every clip
        stored_dur = item.get("duration", None)
        if stored_dur and float(stored_dur) > 0:
            duration_s = float(stored_dur)
        else:
            duration_s = audio_duration_seconds(item.get("audio"))

        speaker_clips[spk].append({
            "ds_idx":        i,
            "source_id":     item["source_id"],
            "duration_s":    duration_s,
            "quality_score": float(item.get("quality_score", 0.0)),
            "snr_db":        float(item.get("snr_db", 0.0)),
            "speech_ratio":  float(item.get("speech_ratio", 0.0)),
            "gender":        item.get("gender", ""),
            "transcription": item.get("transcription", ""),
        })

        if i > start_idx and i % checkpoint_every_rows == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(
                    {
                        "next_idx": i + 1,
                        "speaker_clips": dict(speaker_clips),
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            data_vol.commit()

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        data_vol.commit()

    # ── 3. Rank speakers by total effective speech time ───────────────────────
    speaker_totals = []
    for spk, clips in speaker_clips.items():
        total_dur = sum(c["duration_s"] for c in clips)
        speaker_totals.append((spk, total_dur, len(clips)))

    speaker_totals.sort(key=lambda x: x[1], reverse=True)
    top_speakers = speaker_totals[:TOP_N_SPEAKERS]

    print(f"\n🏆 Top {TOP_N_SPEAKERS} speakers by total duration:")
    print(f"   {'Rank':<5} {'Clips':>6} {'Total min':>10}  Speaker ID")
    print("   " + "-" * 58)
    for rank, (spk, dur, count) in enumerate(top_speakers, 1):
        print(f"   {rank:<5} {count:>6} {dur/60:>10.2f}  {spk}")

    # ── 4. Stratified sampling ────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)

    out_root = "/data/speaker_samples"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []

    print(f"\n🎧 Stratified sampling — {CLIPS_PER_SPEAKER} clips/speaker ...")
    print(f"   Strata: bottom={BOTTOM_SHARE} | middle={MIDDLE_SHARE} | top={TOP_SHARE}")

    for rank, (spk, total_dur, clip_count) in enumerate(top_speakers, 1):
        clips = speaker_clips[spk]

        # Sort ascending so index 0 = worst quality, -1 = best
        clips_sorted = sorted(clips, key=lambda c: c["quality_score"])
        n = len(clips_sorted)

        # Stratum boundaries
        b_end   = max(1, n // 3)
        t_start = max(b_end + 1, n - n // 3)

        bottom = clips_sorted[:b_end]
        middle = clips_sorted[b_end:t_start]
        top    = clips_sorted[t_start:]

        def safe_sample(pool, k):
            k = min(k, len(pool))
            return rng.sample(pool, k) if k > 0 else []

        sampled = (
            safe_sample(bottom, BOTTOM_SHARE)
            + safe_sample(middle, MIDDLE_SHARE)
            + safe_sample(top,    TOP_SHARE)
        )

        # Top up if any stratum was smaller than its share
        if len(sampled) < CLIPS_PER_SPEAKER:
            sampled_ids = {c["source_id"] for c in sampled}
            remainder   = [c for c in clips_sorted if c["source_id"] not in sampled_ids]
            extra       = safe_sample(remainder, CLIPS_PER_SPEAKER - len(sampled))
            sampled.extend(extra)

        # Shuffle to avoid listener anchoring bias during the ear test
        rng.shuffle(sampled)

        # ── Write WAVs ────────────────────────────────────────────────────────
        spk_dir = os.path.join(out_root, spk)
        os.makedirs(spk_dir, exist_ok=True)

        meta_rows = []

        for clip in sampled:
            item = ds_full[clip["ds_idx"]]
            try:
                arr, sr = decode_audio_field(item.get("audio"))
                arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
            except Exception as e:
                print(f"   ⚠️  Could not decode {clip['source_id']}: {e}")
                continue

            wav_name = f"sna_{clip['source_id']}.wav"
            sf.write(os.path.join(spk_dir, wav_name), arr, sr)

            meta_rows.append({
                "source_id":     clip["source_id"],
                "filename":      wav_name,
                "duration_s":    round(clip["duration_s"], 3),
                "quality_score": round(clip["quality_score"], 3),
                "snr_db":        round(clip["snr_db"], 3),
                "speech_ratio":  round(clip["speech_ratio"], 3),
                "gender":        clip["gender"],
                "transcription": clip["transcription"],
            })

            summary_rows.append({
                "rank":          rank,
                "speaker_id":    spk,
                "total_dur_min": round(total_dur / 60, 2),
                "speaker_clips": clip_count,
                **meta_rows[-1],
            })

        # Per-speaker metadata CSV sorted best→worst for easy scanning
        meta_df = pd.DataFrame(meta_rows).sort_values("quality_score", ascending=False)
        meta_df.to_csv(os.path.join(spk_dir, "metadata.csv"), index=False)

        q_scores = [c["quality_score"] for c in sampled]
        print(
            f"   [{rank:>2}] {spk[:28]:<28} "
            f"sampled={len(sampled):>2}  "
            f"q min={min(q_scores):>5.1f}  "
            f"mean={sum(q_scores)/len(q_scores):>5.1f}  "
            f"max={max(q_scores):>5.1f}"
        )

    # ── 5. Top-level summary CSV ──────────────────────────────────────────────
    summary_path = os.path.join(out_root, "summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n📊 Summary CSV → {summary_path}")

    # ── 6. Zip everything ─────────────────────────────────────────────────────
    zip_path = "/data/speaker_samples.zip"
    zip_tmp_path = f"{zip_path}.tmp"
    print(f"\n🗜  Zipping to {zip_path} ...")

    with zipfile.ZipFile(zip_tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_path, "summary.csv")
        for spk_folder in os.listdir(out_root):
            spk_path = os.path.join(out_root, spk_folder)
            if not os.path.isdir(spk_path):
                continue
            for fname in sorted(os.listdir(spk_path)):
                zf.write(
                    os.path.join(spk_path, fname),
                    os.path.join(spk_folder, fname),
                )

    bad_member = None
    with zipfile.ZipFile(zip_tmp_path, "r") as zf:
        bad_member = zf.testzip()
    if bad_member is not None:
        raise RuntimeError(f"ZIP integrity check failed at member: {bad_member}")

    os.replace(zip_tmp_path, zip_path)

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   Zip size: {zip_mb:.1f} MB")

    data_vol.commit()

    print("\n" + "=" * 60)
    print("✅ DONE")
    print(f"\n   {TOP_N_SPEAKERS} speakers × {CLIPS_PER_SPEAKER} clips = "
          f"{TOP_N_SPEAKERS * CLIPS_PER_SPEAKER} clips total")
    print("\nDownload with:")
    print("  modal volume get sna-data-vol /data/speaker_samples.zip \\")
    print("    src/tests/audio/speaker_samples.zip")
    print("  cd src/tests/audio && unzip -o speaker_samples.zip -d speaker_samples/")
    print("=" * 60)


@app.local_entrypoint()
def main():
    pull_speaker_samples.remote()