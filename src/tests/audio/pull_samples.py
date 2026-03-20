"""
Modal script — pulls 500 random clips from /data/refined/ on the volume,
saves raw WAVs + metadata CSV, zips everything, and writes the zip back
to the volume for local download.

After running:
    modal volume get sna-data-vol /data/curate_test.zip src/tests/audio/curate_test.zip
    cd src/tests/audio && unzip -o curate_test.zip -d samples/
"""

import modal

app = modal.App("sna-pull-samples")

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

SAMPLE_SIZE = 500


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=1800,
    volumes={"/data": data_vol},
)
def pull_samples():
    import os
    import random
    import zipfile

    import io

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from datasets import Audio, load_from_disk

    print("=" * 60)
    print("SNA CURATE TEST — PULL SAMPLES")
    print("=" * 60)

    print("\n📂 Loading dataset from /data/refined/...")
    ds = load_from_disk("/data/refined")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"   {len(ds)} rows loaded")

    random.seed(42)
    indices = random.sample(range(len(ds)), min(SAMPLE_SIZE, len(ds)))
    print(f"   Sampling {len(indices)} clips (seed=42 for reproducibility)")

    samples_dir = "/data/curate_test/samples"
    os.makedirs(samples_dir, exist_ok=True)

    metadata = []

    print(f"\n💾 Saving raw WAVs to {samples_dir}...")
    for i, idx in enumerate(indices):
        item = ds[idx]
        source_id = item["source_id"]
        audio_array, sr = sf.read(io.BytesIO(item["audio"]["bytes"]))
        audio_array = audio_array.astype(np.float32)

        filename = f"sna_{source_id}_raw.wav"
        sf.write(os.path.join(samples_dir, filename), audio_array, sr)

        metadata.append({
            "source_id":          source_id,
            "source_speaker_id":  item["source_speaker_id"],
            "speaker_idx":        item["speaker_idx"],
            "speaker_clip_count": item["speaker_clip_count"],
            "gender":             item["gender"],
            "language":           item["language"],
            "has_punctuation":    item["has_punctuation"],
            "transcription":      item["transcription"],
            "sampling_rate":      sr,
            "duration_raw_s":     round(len(audio_array) / sr, 3),
        })

        if (i + 1) % 100 == 0:
            print(f"   [{i+1}/{len(indices)}]")

    meta_path = "/data/curate_test/samples/metadata.csv"
    pd.DataFrame(metadata).to_csv(meta_path, index=False)
    print(f"\n📊 Metadata CSV saved → {meta_path}")

    zip_path = "/data/curate_test.zip"
    print(f"\n🗜  Zipping to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(samples_dir)):
            zf.write(os.path.join(samples_dir, fname), f"samples/{fname}")

    import os as _os
    zip_mb = _os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   Zip size: {zip_mb:.1f} MB")

    data_vol.commit()

    print("\n" + "=" * 60)
    print("✅ DONE")
    print("\nDownload with:")
    print("  modal volume get sna-data-vol /data/curate_test.zip src/tests/audio/curate_test.zip")
    print("  cd src/tests/audio && unzip -o curate_test.zip -d .")
    print("=" * 60)


@app.local_entrypoint()
def main():
    pull_samples.remote()
