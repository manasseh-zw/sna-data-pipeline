"""
Modal script — audits speaker representation in /data/refined/.

For each speaker, computes:
  - clip_count       : number of clips
  - total_duration_s : effective talk time (sum of clip durations)
  - mean_duration_s  : average clip length

Outputs:
  /data/speaker_audit/speaker_report.csv  — full speaker ranking
  /data/speaker_audit/samples/<speaker_id>/  — 3 random raw WAVs per top-N speaker
  /data/speaker_audit.zip  — download-ready archive

Download after running (from repo root):
  modal volume get sna-data-vol /data/speaker_audit.zip src/tests/audio/speaker_audit.zip
  cd src/tests/audio && unzip -o speaker_audit.zip -d speaker_audit/
"""

import modal

app = modal.App("sna-speaker-audit")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "datasets",
        "soundfile",
        "numpy",
        "pandas",
    )
)

TOP_N_SPEAKERS  = 20   # speakers to pull sample clips for
CLIPS_PER_SPEAKER = 3  # sample clips per speaker


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    volumes={"/data": data_vol},
)
def speaker_audit():
    import io
    import os
    import random
    import zipfile
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from datasets import Audio, load_from_disk

    print("=" * 60)
    print("SNA SPEAKER AUDIT")
    print("=" * 60)

    print("\n📂 Loading dataset from /data/refined/...")
    ds = load_from_disk("/data/refined")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"   {len(ds)} rows loaded")

    print("\n📊 Computing per-speaker stats...")
    speaker_clips = defaultdict(list)

    for i, item in enumerate(ds):
        if i % 2000 == 0:
            print(f"   [{i}/{len(ds)}]")
        spk = item["source_speaker_id"]
        audio_bytes = item["audio"]["bytes"]
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes))
            duration = len(arr) / sr
        except Exception:
            duration = 0.0
        speaker_clips[spk].append({
            "source_id": item["source_id"],
            "duration_s": duration,
            "idx": i,
        })

    rows = []
    for spk, clips in speaker_clips.items():
        total = sum(c["duration_s"] for c in clips)
        rows.append({
            "speaker_id":       spk,
            "clip_count":       len(clips),
            "total_duration_s": round(total, 2),
            "total_duration_min": round(total / 60, 2),
            "mean_duration_s":  round(total / len(clips), 2),
        })

    report = pd.DataFrame(rows).sort_values("total_duration_s", ascending=False).reset_index(drop=True)
    report.insert(0, "rank", range(1, len(report) + 1))

    audit_dir = "/data/speaker_audit"
    os.makedirs(audit_dir, exist_ok=True)
    report_path = os.path.join(audit_dir, "speaker_report.csv")
    report.to_csv(report_path, index=False)

    print(f"\n   Saved report → {report_path}")
    print(f"\n{'rank':<6} {'clip_count':<12} {'total_min':<12} {'speaker_id'}")
    print("-" * 70)
    for _, row in report.head(TOP_N_SPEAKERS).iterrows():
        print(f"  {int(row['rank']):<4} {int(row['clip_count']):<12} {row['total_duration_min']:<12} {row['speaker_id']}")

    print(f"\n🎧 Pulling {CLIPS_PER_SPEAKER} sample clips for top {TOP_N_SPEAKERS} speakers...")
    top_speaker_ids = set(report.head(TOP_N_SPEAKERS)["speaker_id"])

    samples_dir = os.path.join(audit_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for spk in top_speaker_ids:
        clips = speaker_clips[spk]
        random.seed(42)
        sample = random.sample(clips, min(CLIPS_PER_SPEAKER, len(clips)))
        spk_dir = os.path.join(samples_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)

        for clip in sample:
            item = ds[clip["idx"]]
            arr, sr = sf.read(io.BytesIO(item["audio"]["bytes"]))
            arr = arr.astype(np.float32)
            out_path = os.path.join(spk_dir, f"sna_{clip['source_id']}_raw.wav")
            sf.write(out_path, arr, sr)

    print(f"\n🗜  Zipping to /data/speaker_audit.zip...")
    zip_path = "/data/speaker_audit.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(report_path, "speaker_report.csv")
        for spk in os.listdir(samples_dir):
            spk_dir = os.path.join(samples_dir, spk)
            for fname in os.listdir(spk_dir):
                zf.write(os.path.join(spk_dir, fname), f"samples/{spk}/{fname}")

    data_vol.commit()

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   Zip size: {zip_mb:.1f} MB")
    print("\n" + "=" * 60)
    print("✅ DONE")
    print("\nDownload with:")
    print("  modal volume get sna-data-vol /data/speaker_audit.zip src/tests/audio/speaker_audit.zip")
    print("  cd src/tests/audio && unzip -o speaker_audit.zip -d speaker_audit/")
    print("=" * 60)


@app.local_entrypoint()
def main():
    speaker_audit.remote()
