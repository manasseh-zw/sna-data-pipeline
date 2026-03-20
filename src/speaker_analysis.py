import modal

app = modal.App("sna-speaker-analysis")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
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
def speaker_analysis():
    import json
    import os
    from datetime import datetime
    from datasets import load_from_disk
    import pandas as pd

    print("=" * 60)
    print("SNA DATA PIPELINE — SPEAKER ANALYSIS")
    print("=" * 60)

    print("\n📂 Loading dataset from /data/raw/...")
    dataset = load_from_disk("/data/raw/")
    print(f"   {len(dataset)} rows loaded")

    df = dataset.to_pandas()

    # --- Per-speaker aggregation ---
    speaker_stats = (
        df.groupby("source_speaker_id")
        .agg(
            clip_count=("source_speaker_id", "count"),
            speaker_idx=("speaker_idx", "first"),
            gender=("gender", lambda x: x.str.strip().str.lower().str.capitalize().mode().iloc[0]),
        )
        .reset_index()
        .sort_values("clip_count", ascending=False)
        .reset_index(drop=True)
    )

    total_clips = len(df)
    total_speakers = len(speaker_stats)

    print(f"\n   Total speakers: {total_speakers}")
    print(f"   Total clips:    {total_clips}")

    # --- Clip count histogram buckets ---
    buckets = [
        ("1–9",    1,   9),
        ("10–24",  10,  24),
        ("25–49",  25,  49),
        ("50–99",  50,  99),
        ("100–199",100, 199),
        ("200–499",200, 499),
        ("500+",   500, 99999),
    ]

    print("\n📊 SPEAKER CLIP COUNT DISTRIBUTION")
    print("-" * 55)
    print(f"   {'Bucket':<12} {'Speakers':>8} {'Clips':>8} {'% Clips':>8}")
    print("   " + "-" * 42)

    histogram = []
    for label, lo, hi in buckets:
        mask = (speaker_stats["clip_count"] >= lo) & (speaker_stats["clip_count"] <= hi)
        spk_count = mask.sum()
        clip_sum = speaker_stats.loc[mask, "clip_count"].sum()
        pct = clip_sum / total_clips * 100
        print(f"   {label:<12} {spk_count:>8} {clip_sum:>8} {pct:>7.1f}%")
        histogram.append({
            "bucket": label,
            "min_clips": lo,
            "max_clips": hi if hi != 99999 else None,
            "speaker_count": int(spk_count),
            "clip_count": int(clip_sum),
            "pct_of_total_clips": round(pct, 2),
        })

    # --- Cumulative coverage at thresholds ---
    thresholds = [10, 20, 30, 50, 75, 100, 150, 200]

    print("\n📈 CUMULATIVE COVERAGE AT CLIP THRESHOLDS")
    print("-" * 65)
    print(f"   {'Min clips':>10} {'Speakers kept':>14} {'Clips kept':>11} {'% Clips':>8} {'% Speakers':>11}")
    print("   " + "-" * 58)

    coverage = []
    for threshold in thresholds:
        qualified = speaker_stats[speaker_stats["clip_count"] >= threshold]
        spk_kept = len(qualified)
        clips_kept = qualified["clip_count"].sum()
        pct_clips = clips_kept / total_clips * 100
        pct_spk = spk_kept / total_speakers * 100
        print(f"   {threshold:>10} {spk_kept:>14} {clips_kept:>11} {pct_clips:>7.1f}% {pct_spk:>10.1f}%")
        coverage.append({
            "min_clips_threshold": threshold,
            "speakers_kept": int(spk_kept),
            "clips_kept": int(clips_kept),
            "pct_clips_kept": round(pct_clips, 2),
            "pct_speakers_kept": round(pct_spk, 2),
        })

    # --- Gender breakdown ---
    gender_norm = df["gender"].str.strip().str.lower().str.capitalize()
    gender_counts = gender_norm.value_counts()

    print("\n👤 GENDER DISTRIBUTION (normalised)")
    print("-" * 40)
    for g, cnt in gender_counts.items():
        pct = cnt / total_clips * 100
        print(f"   {g:<12} {cnt:>5} clips  ({pct:.1f}%)")

    # --- Gender breakdown by speaker ---
    gender_by_speaker = (
        speaker_stats.groupby("gender")["clip_count"]
        .agg(speaker_count="count", total_clips="sum")
        .reset_index()
    )

    print("\n👤 GENDER BREAKDOWN BY SPEAKER")
    print("-" * 40)
    for _, row in gender_by_speaker.iterrows():
        print(f"   {row['gender']:<12} {row['speaker_count']:>4} speakers   {row['total_clips']:>6} clips")

    # --- Top 20 speakers table ---
    print("\n🏆 TOP 20 SPEAKERS")
    print("-" * 65)
    print(f"   {'Rank':>4} {'speaker_idx':>11} {'clips':>6} {'gender':<10} {'speaker_id (truncated)'}")
    print("   " + "-" * 60)
    for rank, row in speaker_stats.head(20).iterrows():
        bar = "█" * min(row["clip_count"] // 30, 30)
        print(f"   {rank+1:>4} {row['speaker_idx']:>11} {row['clip_count']:>6} {row['gender']:<10} {row['source_speaker_id'][:28]}  {bar}")

    # --- Long tail summary ---
    tail_mask = speaker_stats["clip_count"] < 50
    tail_speakers = tail_mask.sum()
    tail_clips = speaker_stats.loc[tail_mask, "clip_count"].sum()
    print(f"\n⚠️  LONG TAIL SUMMARY (< 50 clips)")
    print(f"   {tail_speakers} speakers ({tail_speakers/total_speakers*100:.1f}% of speakers) "
          f"account for {tail_clips} clips ({tail_clips/total_clips*100:.1f}% of total)")

    # --- Full per-speaker table ---
    per_speaker_records = speaker_stats.rename(
        columns={"source_speaker_id": "speaker_id"}
    )[["speaker_idx", "speaker_id", "clip_count", "gender"]].to_dict(orient="records")

    # --- Save report ---
    report = {
        "phase": "speaker_analysis",
        "timestamp": datetime.now().isoformat(),
        "total_speakers": total_speakers,
        "total_clips": total_clips,
        "gender_distribution_normalised": {
            str(k): int(v) for k, v in gender_counts.items()
        },
        "gender_breakdown_by_speaker": [
            {
                "gender": row["gender"],
                "speaker_count": int(row["speaker_count"]),
                "total_clips": int(row["total_clips"]),
            }
            for _, row in gender_by_speaker.iterrows()
        ],
        "clip_count_histogram": histogram,
        "cumulative_coverage": coverage,
        "long_tail": {
            "threshold": 50,
            "tail_speakers": int(tail_speakers),
            "tail_clips": int(tail_clips),
            "pct_speakers": round(tail_speakers / total_speakers * 100, 2),
            "pct_clips": round(tail_clips / total_clips * 100, 2),
        },
        "per_speaker": [
            {k: (int(v) if isinstance(v, (int, float)) and k != "speaker_id" else v)
             for k, v in r.items()}
            for r in per_speaker_records
        ],
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/speaker_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved → {report_path}")

    print("\n" + "=" * 60)
    print("✅ SPEAKER ANALYSIS COMPLETE")
    print("=" * 60)


@app.local_entrypoint()
def main():
    speaker_analysis.remote()
