import modal

app = modal.App("sna-annotate-metadata")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "datasets[audio]",
        "pandas",
    )
)

@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    volumes={"/data": data_vol},
)
def annotate_metadata():
    import json
    import os
    import shutil
    from datetime import datetime
    from datasets import load_from_disk
    import pandas as pd

    print("=" * 60)
    print("SNA DATA PIPELINE — ANNOTATE METADATA")
    print("=" * 60)

    print("\n📂 Loading dataset from /data/raw/...")
    dataset = load_from_disk("/data/raw/")
    print(f"   {len(dataset)} rows loaded")
    print(f"   columns: {dataset.column_names}")

    # --- Gender normalisation ---
    print("\n👤 Normalising gender column...")
    raw_gender_counts = pd.Series(dataset["gender"]).value_counts().to_dict()
    print(f"   Raw values: {raw_gender_counts}")

    def normalize_gender(example):
        g = example["gender"]
        if isinstance(g, str):
            example["gender"] = g.strip().lower().capitalize()
        return example

    dataset = dataset.map(normalize_gender, desc="Normalising gender")
    normalised_gender_counts = pd.Series(dataset["gender"]).value_counts().to_dict()
    print(f"   Normalised values: {normalised_gender_counts}")

    # --- Language normalisation ---
    print("\n🌐 Normalising language column...")
    raw_lang_counts = pd.Series(dataset["language"]).value_counts().to_dict()
    print(f"   Raw values: {raw_lang_counts}")

    def normalize_language(example):
        lang = example["language"]
        if isinstance(lang, str):
            example["language"] = lang.strip().lower()
        return example

    dataset = dataset.map(normalize_language, desc="Normalising language")
    normalised_lang_counts = pd.Series(dataset["language"]).value_counts().to_dict()
    print(f"   Normalised values: {normalised_lang_counts}")

    # --- Speaker clip count ---
    print("\n🔢 Adding speaker_clip_count column...")
    speaker_counts = pd.Series(dataset["source_speaker_id"]).value_counts().to_dict()

    def add_clip_count(example):
        example["speaker_clip_count"] = speaker_counts[example["source_speaker_id"]]
        return example

    map_workers = max(1, min(4, (os.cpu_count() or 1)))
    dataset = dataset.map(
        add_clip_count,
        num_proc=map_workers,
        desc=f"Adding speaker_clip_count ({map_workers} workers)",
    )

    clip_counts = pd.Series(dataset["speaker_clip_count"])
    print(f"   Min speaker clip count:    {clip_counts.min()}")
    print(f"   Max speaker clip count:    {clip_counts.max()}")
    print(f"   Median speaker clip count: {clip_counts.median()}")

    # --- Save audit report ---
    audit = {
        "phase": "annotate_metadata",
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(dataset),
        "columns": dataset.column_names,
        "gender": {
            "before": {str(k): int(v) for k, v in raw_gender_counts.items()},
            "after": {str(k): int(v) for k, v in normalised_gender_counts.items()},
        },
        "language": {
            "before": {str(k): int(v) for k, v in raw_lang_counts.items()},
            "after": {str(k): int(v) for k, v in normalised_lang_counts.items()},
        },
        "speaker_clip_count": {
            "min": int(clip_counts.min()),
            "max": int(clip_counts.max()),
            "median": float(clip_counts.median()),
            "mean": round(float(clip_counts.mean()), 1),
        },
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/annotate_metadata_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n💾 Audit report saved → {report_path}")

    # --- Write to temp path then rename for safety ---
    print("\n💾 Saving annotated dataset...")
    tmp_path = "/data/raw_annotated_tmp"
    final_path = "/data/raw"

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    dataset.save_to_disk(tmp_path)

    shutil.rmtree(final_path)
    shutil.move(tmp_path, final_path)

    data_vol.commit()

    print(f"   Dataset saved → {final_path}")
    print("\n" + "=" * 60)
    print("✅ ANNOTATE METADATA COMPLETE")
    print(f"   {len(dataset)} rows")
    print(f"   columns: {dataset.column_names}")
    print("=" * 60)


@app.local_entrypoint()
def main():
    annotate_metadata.remote()
