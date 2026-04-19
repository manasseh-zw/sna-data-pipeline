"""
Modal script — TTS subset selection analysis + ear test sample pull.

Runs against /data/sna_annotated on the volume.

Does two things in one run:
  1. ANALYSIS — for the top 20 speakers by effective speech hours, produces
     a combined filter matrix showing hours/clips retained under every
     combination of quality floor x duration ceiling. Printed to console
     and saved as JSON.

  2. SAMPLE PULL — for each top-20 speaker, pulls 10 clips from their
     top-quality tier and 10 clips from their bottom-quality tier as
     disjoint sets (duration <= 20s filter). WAVs + per-speaker metadata
     CSV are zipped for download.

Outputs:
  /data/reports/tts_subset_analysis.json   — full analysis results
  /data/tts_samples.zip                    — ear test samples

Download after running:
  modal volume get sna-data-vol /data/reports/tts_subset_analysis.json .
  modal volume get sna-data-vol /data/tts_samples.zip .
  unzip tts_samples.zip -d tts_samples/
"""

import modal

app = modal.App("sna-tts-subset-analysis")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1")
    .uv_pip_install(
        "datasets[audio]",
        "numpy",
        "pandas",
        "soundfile",
        "tqdm",
    )
)

# ── Filter combinations to evaluate ──────────────────────────────────────────
# Quality floors are derived from dataset percentiles at runtime.
QUALITY_PERCENTILES = [40, 50, 60, 70, 80]
DURATION_CEILINGS = [10, 12, 15, 18, 20, 25]  # seconds
DURATION_FLOOR = 5.0  # always applied

# ── Sample pull config ────────────────────────────────────────────────────────
SAMPLE_DURATION_CEILING = 20.0  # only sample from clips under this duration
SAMPLES_PER_TIER = 10  # 10 from top quality + 10 from bottom quality

TOP_N_SPEAKERS = 20


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/data": data_vol},
)
def analyse_and_sample():
    import io
    import json
    import os
    import zipfile
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from datasets import Audio, concatenate_datasets, load_from_disk
    from tqdm import tqdm

    print("=" * 72)
    print("SNA — TTS SUBSET SELECTION ANALYSIS + EAR TEST SAMPLE PULL")
    print("=" * 72)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("\n📂 Loading /data/sna_annotated ...")
    dataset_dict = load_from_disk("/data/sna_annotated")
    splits = list(dataset_dict.keys())
    print(f"   Splits: {splits}")

    # Concatenate all splits — we analyse across the full dataset
    ds_full = concatenate_datasets([dataset_dict[s] for s in splits])
    print(f"   Total clips: {len(ds_full):,}")

    # ── 2. Build metadata dataframe (no audio decoding yet) ───────────────────
    print("\n📊 Building metadata dataframe ...")
    meta_cols = [
        "speaker_id",
        "gender",
        "duration",
        "quality_score",
        "source_id",
        "transcription",
        "speaker_clip_count",
    ]
    df = ds_full.select_columns(meta_cols).to_pandas()
    df["duration"] = df["duration"].astype(float)
    df["quality_score"] = df["quality_score"].astype(float)
    df["ds_index"] = np.arange(len(df))  # preserve index into ds_full

    # Apply universal duration floor
    df = df[df["duration"] >= DURATION_FLOOR].copy()
    print(f"   After {DURATION_FLOOR}s floor: {len(df):,} clips")

    # Build quality floors from dataset percentiles
    quality_floor_map = {
        f"p{p}": float(np.percentile(df["quality_score"].values, p))
        for p in QUALITY_PERCENTILES
    }
    quality_floor_defs = [("none", None, 0.0)]
    quality_floor_defs.extend(
        (f"p{p}", p, quality_floor_map[f"p{p}"]) for p in QUALITY_PERCENTILES
    )

    print("\n📈 Quality floors from distribution:")
    for label, percentile, floor in quality_floor_defs:
        if label == "none":
            print("   none: no quality filter")
        else:
            print(f"   {label} (>= {percentile}th percentile): {floor:.2f}")

    # ── 3. Rank top 20 speakers by total effective speech hours ───────────────
    print(f"\n🏆 Ranking top {TOP_N_SPEAKERS} speakers by total speech hours ...")
    speaker_stats = (
        df.groupby(["speaker_id", "gender"])
        .agg(
            total_clips=("duration", "count"),
            total_hours=("duration", lambda x: round(x.sum() / 3600.0, 2)),
            mean_duration=("duration", lambda x: round(x.mean(), 2)),
            mean_quality=("quality_score", lambda x: round(x.mean(), 2)),
            median_quality=("quality_score", lambda x: round(x.median(), 2)),
            p10_quality=("quality_score", lambda x: round(np.percentile(x, 10), 2)),
            p90_quality=("quality_score", lambda x: round(np.percentile(x, 90), 2)),
        )
        .reset_index()
        .sort_values("total_hours", ascending=False)
        .head(TOP_N_SPEAKERS)
        .reset_index(drop=True)
    )
    speaker_stats.insert(0, "rank", range(1, len(speaker_stats) + 1))

    top_speaker_ids = speaker_stats["speaker_id"].tolist()

    print(
        f"\n  {'Rank':<5} {'Spk':>4} {'Gender':<8} "
        f"{'Hours':>6} {'Clips':>6} {'AvgQ':>6} {'P10Q':>6} {'P90Q':>6}"
    )
    print("  " + "-" * 55)
    for _, row in speaker_stats.iterrows():
        print(
            f"  {int(row['rank']):<5} {int(row['speaker_id']):>4} "
            f"{row['gender']:<8} {row['total_hours']:>6.2f} "
            f"{int(row['total_clips']):>6} {row['mean_quality']:>6.1f} "
            f"{row['p10_quality']:>6.1f} {row['p90_quality']:>6.1f}"
        )

    # ── 4. Combined filter matrix per speaker ─────────────────────────────────
    print("\n📋 Building combined filter matrix ...")

    matrix_rows = []
    df_top = df[df["speaker_id"].isin(top_speaker_ids)].copy()

    for spk_id in top_speaker_ids:
        spk_df = df_top[df_top["speaker_id"] == spk_id]
        gender = speaker_stats.loc[
            speaker_stats["speaker_id"] == spk_id, "gender"
        ].values[0]

        for q_label, q_percentile, q_floor in quality_floor_defs:
            for d_ceil in DURATION_CEILINGS:
                mask = spk_df["duration"] <= d_ceil
                if q_label != "none":
                    mask = mask & (spk_df["quality_score"] >= q_floor)

                filtered = spk_df[mask]
                clips = int(len(filtered))
                hours = round(float(filtered["duration"].sum()) / 3600.0, 2)
                pct_retained = round(
                    clips / len(spk_df) * 100 if len(spk_df) > 0 else 0, 1
                )

                matrix_rows.append(
                    {
                        "speaker_id": int(spk_id),
                        "gender": gender,
                        "quality_floor_label": q_label,
                        "quality_floor_percentile": q_percentile,
                        "quality_floor_value": round(float(q_floor), 2),
                        "duration_ceil": d_ceil,
                        "clips": clips,
                        "hours": hours,
                        "pct_retained": pct_retained,
                    }
                )

    matrix_df = pd.DataFrame(matrix_rows)

    # Print focused views for percentile-driven floors
    focus_labels = [label for label in ["p50", "p60"] if label in quality_floor_map]
    if not focus_labels:
        focus_labels = [d[0] for d in quality_floor_defs if d[0] != "none"][:2]

    print(
        f"\n  Filter matrix for top {TOP_N_SPEAKERS} speakers "
        f"(focus floors: {', '.join(focus_labels)})\n"
    )

    for q_label in focus_labels:
        q_floor = quality_floor_map[q_label]
        print(f"  ── Quality >= {q_floor:.2f} ({q_label}) ───────────────────────────")
        sub = matrix_df[matrix_df["quality_floor_label"] == q_label]
        pivot = sub.pivot_table(
            index=["speaker_id", "gender"],
            columns="duration_ceil",
            values="hours",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None
        pivot = pivot.rename(columns={c: f"<={c}s_h" for c in DURATION_CEILINGS})

        # Merge rank and total hours for context
        pivot = pivot.merge(
            speaker_stats[["speaker_id", "rank", "total_hours"]], on="speaker_id"
        ).sort_values("rank")

        print(
            f"\n  {'rank':<5} {'spk':>4} {'gender':<8} {'total_h':>8}  "
            + "  ".join(f"{c:>6}" for c in DURATION_CEILINGS)
        )
        print("  " + "-" * 68)
        for _, row in pivot.iterrows():
            vals = "  ".join(
                f"{row.get(f'<={c}s_h', 0):>6.2f}" for c in DURATION_CEILINGS
            )
            print(
                f"  {int(row['rank']):<5} {int(row['speaker_id']):>4} "
                f"{row['gender']:<8} {row['total_hours']:>8.2f}  {vals}"
            )
        print()

    # ── 5. Overall duration distribution ─────────────────────────────────────
    print("\n📊 Duration distribution (all clips, all speakers)")
    print(
        f"\n  {'Bucket':<12} {'Clips':>7} {'%Clips':>7} "
        f"{'Hours':>7} {'%Hours':>7} {'Cumul_h':>8}"
    )
    print("  " + "-" * 56)

    boundaries = list(range(int(DURATION_FLOOR), 26)) + [float("inf")]
    total_h = df["duration"].sum() / 3600.0
    cumul_h = 0.0
    dur_buckets = []

    for lo, hi in zip([DURATION_FLOOR] + boundaries[:-1], boundaries):
        label = f"{int(lo)}-{int(hi)}s" if hi != float("inf") else f"{int(lo)}s+"
        mask = (df["duration"] >= lo) & (df["duration"] < hi)
        clips = int(mask.sum())
        hrs = float(df.loc[mask, "duration"].sum()) / 3600.0
        cumul_h += hrs
        print(
            f"  {label:<12} {clips:>7,} {clips / len(df) * 100:>6.1f}%  "
            f"{hrs:>6.2f}h {hrs / total_h * 100:>6.1f}%  {cumul_h:>7.2f}h"
        )
        dur_buckets.append(
            {
                "bucket": label,
                "lo_s": lo,
                "hi_s": hi if hi != float("inf") else None,
                "clips": clips,
                "hours": round(hrs, 3),
                "pct_clips": round(clips / len(df) * 100, 1),
                "cumulative_hours": round(cumul_h, 3),
            }
        )

    # Cumulative hours retained at each ceiling
    print(f"\n  Cumulative retention at duration ceilings:")
    print(f"  {'Ceiling':<10} {'Clips':>7} {'%Clips':>7} {'Hours':>7} {'%Hours':>7}")
    print("  " + "-" * 44)
    ceiling_rows = []
    for ceil in DURATION_CEILINGS:
        mask = df["duration"] <= ceil
        clips = int(mask.sum())
        hrs = float(df.loc[mask, "duration"].sum()) / 3600.0
        print(
            f"  <={ceil}s      {clips:>7,} {clips / len(df) * 100:>6.1f}%  "
            f"{hrs:>6.2f}h {hrs / total_h * 100:>6.1f}%"
        )
        ceiling_rows.append(
            {
                "ceiling_s": ceil,
                "clips": clips,
                "hours": round(hrs, 3),
                "pct_clips": round(clips / len(df) * 100, 1),
                "pct_hours": round(hrs / total_h * 100, 1),
            }
        )

    # ── 6. Quality distribution ───────────────────────────────────────────────
    print("\n📊 Quality score distribution (all clips)")
    percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]
    q_vals = np.percentile(df["quality_score"].values, percentiles)
    print(f"\n  {'Percentile':<12} {'quality_score':>14}")
    print("  " + "-" * 28)
    q_percentiles = {}
    for p, v in zip(percentiles, q_vals):
        print(f"  p{p:<10} {v:>14.2f}")
        q_percentiles[f"p{p}"] = round(float(v), 2)

    print("\n  Hours retained at percentile-derived quality floors:")
    print(
        f"  {'Floor':<10} {'Value':>8} {'Clips':>7} {'%Clips':>7} {'Hours':>7} {'%Hours':>7}"
    )
    print("  " + "-" * 44)
    quality_floor_rows = []
    for q_label, q_percentile, q_floor in quality_floor_defs:
        if q_label == "none":
            continue
        mask = df["quality_score"] >= q_floor
        clips = int(mask.sum())
        hrs = float(df.loc[mask, "duration"].sum()) / 3600.0
        print(
            f"  {q_label:<10} {q_floor:>8.2f} {clips:>7,} {clips / len(df) * 100:>6.1f}%  "
            f"{hrs:>6.2f}h {hrs / total_h * 100:>6.1f}%"
        )
        quality_floor_rows.append(
            {
                "label": q_label,
                "percentile": q_percentile,
                "quality_floor_value": round(float(q_floor), 2),
                "clips": clips,
                "hours": round(hrs, 3),
                "pct_clips": round(clips / len(df) * 100, 1),
                "pct_hours": round(hrs / total_h * 100, 1),
            }
        )

    # ── 7. Sample pull for ear test ───────────────────────────────────────────
    print(f"\n🎧 Pulling ear test samples (duration<={SAMPLE_DURATION_CEILING}s) ...")
    print(
        f"   {SAMPLES_PER_TIER} top-quality + {SAMPLES_PER_TIER} bottom-quality "
        "per speaker (disjoint sets)"
    )

    # Cast audio column for decoding
    ds_audio = ds_full.cast_column("audio", Audio(sampling_rate=24000, decode=False))

    os.makedirs("/data/tts_samples", exist_ok=True)
    zip_path = "/data/tts_samples.zip"
    sample_manifest = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for spk_id in tqdm(top_speaker_ids, desc="Speakers", unit="spk"):
            spk_df = df_top[
                (df_top["speaker_id"] == spk_id)
                & (df_top["duration"] <= SAMPLE_DURATION_CEILING)
            ].copy()

            if len(spk_df) == 0:
                print(f"   ⚠️  Speaker {spk_id}: no clips pass sample filters")
                continue

            # Build disjoint best/worst pools.
            spk_desc = spk_df.sort_values("quality_score", ascending=False)

            n_top = min(SAMPLES_PER_TIER, len(spk_desc))
            top_clips = spk_desc.head(n_top)

            remaining = spk_desc.drop(index=top_clips.index)
            spk_asc = remaining.sort_values("quality_score", ascending=True)
            n_bot = min(SAMPLES_PER_TIER, len(spk_asc))
            bot_clips = spk_asc.head(n_bot)

            gender = speaker_stats.loc[
                speaker_stats["speaker_id"] == spk_id, "gender"
            ].values[0]
            rank = int(
                speaker_stats.loc[speaker_stats["speaker_id"] == spk_id, "rank"].values[
                    0
                ]
            )

            folder = f"speaker_{rank:02d}_id{spk_id}_{gender}"
            meta_rows = []

            for tier, clips_subset in [("top", top_clips), ("bot", bot_clips)]:
                for _, clip_row in clips_subset.iterrows():
                    ds_idx = int(clip_row["ds_index"])
                    source_id = clip_row["source_id"]

                    try:
                        item = ds_audio[ds_idx]
                        audio_bytes = item["audio"]["bytes"]
                        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                        if arr.ndim > 1:
                            arr = arr.mean(axis=1)

                        wav_name = (
                            f"{tier}_q{clip_row['quality_score']:.0f}_{source_id}.wav"
                        )
                        buf = io.BytesIO()
                        sf.write(buf, arr, sr, format="WAV", subtype="FLOAT")
                        zf.writestr(f"{folder}/{wav_name}", buf.getvalue())

                        meta_rows.append(
                            {
                                "tier": tier,
                                "filename": wav_name,
                                "source_id": source_id,
                                "quality_score": round(
                                    float(clip_row["quality_score"]), 2
                                ),
                                "duration_s": round(float(clip_row["duration"]), 2),
                                "transcription": clip_row["transcription"],
                            }
                        )
                        sample_manifest.append(
                            {
                                "speaker_id": int(spk_id),
                                "rank": rank,
                                "gender": gender,
                                **meta_rows[-1],
                            }
                        )

                    except Exception as e:
                        print(f"   ⚠️  {source_id}: {e}")

            # Per-speaker metadata CSV
            if meta_rows:
                meta_df = pd.DataFrame(meta_rows)
                csv_buf = io.StringIO()
                meta_df.to_csv(csv_buf, index=False)
                zf.writestr(f"{folder}/metadata.csv", csv_buf.getvalue())

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n   Zip written: {zip_path} ({zip_mb:.1f} MB)")
    print(f"   Total samples pulled: {len(sample_manifest)}")

    # ── 8. Save analysis JSON ─────────────────────────────────────────────────
    os.makedirs("/data/reports", exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "source": "/data/sna_annotated",
        "total_clips_analysed": int(len(df)),
        "total_hours": round(float(total_h), 3),
        "top_n_speakers": TOP_N_SPEAKERS,
        "quality_floor_strategy": {
            "type": "percentile_based",
            "percentiles": QUALITY_PERCENTILES,
            "floors": {k: round(v, 2) for k, v in quality_floor_map.items()},
        },
        "speaker_ranking": speaker_stats.to_dict(orient="records"),
        "filter_matrix": matrix_df.to_dict(orient="records"),
        "duration_buckets": dur_buckets,
        "duration_ceilings": ceiling_rows,
        "quality_percentiles": q_percentiles,
        "quality_floors": quality_floor_rows,
        "sample_pull": {
            "duration_ceiling": SAMPLE_DURATION_CEILING,
            "samples_per_tier": SAMPLES_PER_TIER,
            "disjoint_tiers": True,
            "total_samples": len(sample_manifest),
            "manifest": sample_manifest,
        },
    }

    report_path = "/data/reports/tts_subset_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    data_vol.commit()

    print("\n" + "=" * 72)
    print("✅ ANALYSIS COMPLETE")
    print(f"   Report  → {report_path}")
    print(f"   Samples → {zip_path}")
    print("\nDownload with:")
    print("  modal volume get sna-data-vol /data/reports/tts_subset_analysis.json .")
    print("  modal volume get sna-data-vol /data/tts_samples.zip .")
    print("=" * 72)


@app.local_entrypoint()
def main():
    analyse_and_sample.remote()
