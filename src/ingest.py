import modal

app = modal.App("sna-ingest")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "datasets[audio]",
        "huggingface_hub",
        "soundfile",
        "pandas",
    )
)

@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=7200,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_dotenv()],
)
def ingest():
    import os
    import json
    from datetime import datetime
    from datasets import load_dataset, concatenate_datasets
    from huggingface_hub import login
    import pandas as pd

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")
    login(token=hf_token)

    print("=" * 60)
    print("SNA DATA PIPELINE — PHASE 1: INGEST")
    print("=" * 60)

    labeled_splits = ["train", "validation", "test"]

    def load_one_split(split_name):
        print(f"\n📥 Pulling split '{split_name}'...")
        ds = load_dataset("google/WaxalNLP", "sna_asr", split=split_name)
        print(f"   {split_name}: {len(ds)} rows")
        return ds

    split_datasets = [load_one_split(split_name) for split_name in labeled_splits]
    print("\n🔗 Concatenating labeled splits...")
    dataset = concatenate_datasets(split_datasets)
    print(f"   total rows: {len(dataset)}")
    print(f"   columns: {dataset.column_names}")

    # --- Rename speaker_id to source_speaker_id to preserve provenance ---
    print("\n🏷️  Renaming speaker_id → source_speaker_id...")
    dataset = dataset.rename_column("speaker_id", "source_speaker_id")

    # --- Rename id → source_id ---
    if "id" in dataset.column_names:
        dataset = dataset.rename_column("id", "source_id")

    # --- Speaker frequency audit ---
    print("\n📊 SPEAKER AUDIT")
    print("-" * 40)
    speaker_counts = pd.Series(dataset["source_speaker_id"]).value_counts()
    total_speakers = len(speaker_counts)

    print(f"   Total unique speakers:      {total_speakers}")
    print(f"\n   Top 10 speakers by clip count:")
    for spk, count in speaker_counts.head(10).items():
        bar = "█" * min(count // 10, 40)
        print(f"   {spk[:30]:<32} {count:>4} clips  {bar}")
    print(f"\n   Bottom 5 speakers by clip count:")
    for spk, count in speaker_counts.tail(5).items():
        print(f"   {spk[:30]:<32} {count:>4} clips")

    # --- Gender distribution ---
    print("\n👤 GENDER DISTRIBUTION")
    print("-" * 40)
    gender_counts = pd.Series(dataset["gender"]).value_counts()
    for gender, count in gender_counts.items():
        pct = count / len(dataset) * 100
        print(f"   {gender:<12} {count:>5} clips  ({pct:.1f}%)")

    # --- Build speaker_idx mapping ---
    print("\n🗂️  Building speaker_idx mapping...")
    # Sort by frequency descending so idx=0 is the most represented speaker
    speaker_to_idx = {
        spk: idx for idx, spk in enumerate(speaker_counts.index)
    }

    def add_speaker_idx(example):
        spk = example["source_speaker_id"]
        example["speaker_idx"] = speaker_to_idx[spk]
        return example

    print("   Mapping source_speaker_id → speaker_idx...")
    map_workers = max(1, min(8, (os.cpu_count() or 1)))
    dataset = dataset.map(
        add_speaker_idx,
        num_proc=map_workers,
        desc=f"Assigning speaker indices ({map_workers} workers)",
    )

    # --- Transcription length audit ---
    print("\n📝 TRANSCRIPTION AUDIT")
    print("-" * 40)
    transcript_lengths = pd.Series([len(t) for t in dataset["transcription"]])
    print(f"   Mean chars:   {transcript_lengths.mean():.1f}")
    print(f"   Median chars: {transcript_lengths.median():.1f}")
    print(f"   Min chars:    {transcript_lengths.min()}")
    print(f"   Max chars:    {transcript_lengths.max()}")
    empty = (transcript_lengths == 0).sum()
    print(f"   Empty transcriptions: {empty}")

    # --- Save audit report ---
    audit = {
        "phase": "ingest",
        "timestamp": datetime.now().isoformat(),
        "source": "google/WaxalNLP sna_asr (train + validation + test)",
        "total_rows": len(dataset),
        "columns": dataset.column_names,
        "speakers": {
            "total": int(total_speakers),
            "top10": {
                str(k): int(v)
                for k, v in speaker_counts.head(10).items()
            },
        },
        "gender_distribution": {
            str(k): int(v) for k, v in gender_counts.items()
        },
        "transcription": {
            "mean_chars": round(float(transcript_lengths.mean()), 1),
            "median_chars": round(float(transcript_lengths.median()), 1),
            "min_chars": int(transcript_lengths.min()),
            "max_chars": int(transcript_lengths.max()),
            "empty_count": int(empty),
        },
        "speaker_idx_mapping": {
            str(k): int(v) for k, v in speaker_to_idx.items()
        },
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/01_ingest_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n💾 Audit report saved → {report_path}")

    # --- Save dataset to volume ---
    print("\n💾 Saving raw dataset to volume at /data/raw/...")
    dataset.save_to_disk("/data/raw/")
    data_vol.commit()

    print("\n" + "=" * 60)
    print("✅ INGEST COMPLETE")
    print(f"   {len(dataset)} rows saved to /data/raw/")
    print(f"   {total_speakers} speakers indexed")
    print(f"   Audit report → /data/reports/01_ingest_audit.json")
    print("=" * 60)


@app.local_entrypoint()
def main():
    ingest.remote()