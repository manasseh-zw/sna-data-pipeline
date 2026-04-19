"""
Modal script - audit expanded TTS subset options.

Goal:
  Compare scenario rubrics for a larger Phase-1 dataset (phonetic coverage)
  while quantifying trade-offs in quality and speaker concentration.

Outputs:
  /data/reports/tts_expansion_audit.json
  /data/reports/tts_expansion_audit_matrix.csv
"""

import modal

app = modal.App("sna-audit-tts-expansion-options")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "datasets[audio]",
    "numpy",
    "pandas",
)

ANNOTATED_PATH = "/data/sna_annotated"
REPORT_JSON_PATH = "/data/reports/tts_expansion_audit.json"
REPORT_CSV_PATH = "/data/reports/tts_expansion_audit_matrix.csv"

DURATION_FLOOR = 5.0
TARGET_HOURS = 40.0

# From current curation decisions (integer speaker_id in /data/sna_annotated)
PRIMARY_SPEAKERS = [0, 29, 1, 2, 31, 3, 7, 11]
CONDITIONAL_SPEAKERS = [28, 32, 30, 4, 6, 8, 35]

# Scenario grids
TOP_N_GRID = [20, 30, 40]
QUALITY_LABELS = ["p40", "p50", "p60", "p70"]
DURATION_CEILINGS = [20.0, 25.0]


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/data": data_vol},
)
def audit_tts_expansion_options():
    import json
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from datasets import concatenate_datasets, load_from_disk

    print("=" * 72)
    print("SNA - AUDIT TTS EXPANSION OPTIONS")
    print("=" * 72)

    print("\nLoading /data/sna_annotated ...")
    ds_dict = load_from_disk(ANNOTATED_PATH)
    ds_full = concatenate_datasets([ds_dict[s] for s in ds_dict.keys()])

    cols = ["speaker_id", "duration", "quality_score", "transcription"]
    missing = [c for c in cols if c not in ds_full.column_names]
    if missing:
        raise RuntimeError(f"Missing expected columns in annotated dataset: {missing}")

    df = ds_full.select_columns(cols).to_pandas()
    df["speaker_id"] = df["speaker_id"].astype("int32")
    df["duration"] = df["duration"].astype("float32")
    df["quality_score"] = df["quality_score"].astype("float32")

    # Baseline safety floor
    df = df[df["duration"] >= DURATION_FLOOR].copy()
    total_hours = float(df["duration"].sum()) / 3600.0

    print(f"   Clips analysed: {len(df):,}")
    print(f"   Hours analysed: {total_hours:.3f}")
    print(f"   Unique speakers: {df['speaker_id'].nunique()}")

    # Global quality percentiles
    percentile_map = {
        "p40": float(np.percentile(df["quality_score"].values, 40)),
        "p50": float(np.percentile(df["quality_score"].values, 50)),
        "p60": float(np.percentile(df["quality_score"].values, 60)),
        "p70": float(np.percentile(df["quality_score"].values, 70)),
    }

    print("\nGlobal quality floors:")
    for label in QUALITY_LABELS:
        print(f"   {label}: {percentile_map[label]:.2f}")

    # Speaker ranking by total hours
    speaker_hours = (
        df.groupby("speaker_id", as_index=False)["duration"]
        .sum()
        .sort_values("duration", ascending=False)
    )
    speaker_hours["hours"] = speaker_hours["duration"] / 3600.0
    ranked_speakers = speaker_hours["speaker_id"].tolist()

    top_n_speaker_sets = {
        f"top_{n}": ranked_speakers[:n] for n in TOP_N_GRID if n <= len(ranked_speakers)
    }

    fixed_speaker_sets = {
        "primary_only": PRIMARY_SPEAKERS,
        "primary_plus_conditional": sorted(
            list(set(PRIMARY_SPEAKERS + CONDITIONAL_SPEAKERS))
        ),
    }

    scenario_defs = []

    # Same speaker pool, varied quality/duration (tests lower thresholds idea)
    for set_name, spk_ids in fixed_speaker_sets.items():
        for q_label in QUALITY_LABELS:
            for d_ceil in DURATION_CEILINGS:
                scenario_defs.append(
                    {
                        "scenario_type": "fixed_speaker_pool",
                        "speaker_pool": set_name,
                        "speaker_count": len(spk_ids),
                        "speaker_ids": spk_ids,
                        "quality_label": q_label,
                        "quality_floor": percentile_map[q_label],
                        "duration_ceiling": d_ceil,
                    }
                )

    # Expanded speaker pool by top-N, standard floors
    for set_name, spk_ids in top_n_speaker_sets.items():
        for q_label in QUALITY_LABELS:
            for d_ceil in DURATION_CEILINGS:
                scenario_defs.append(
                    {
                        "scenario_type": "top_n_expansion",
                        "speaker_pool": set_name,
                        "speaker_count": len(spk_ids),
                        "speaker_ids": spk_ids,
                        "quality_label": q_label,
                        "quality_floor": percentile_map[q_label],
                        "duration_ceiling": d_ceil,
                    }
                )

    def add_word_metrics(sub_df):
        # Lightweight lexical coverage proxy
        texts = sub_df["transcription"].astype(str).str.lower().tolist()
        token_count = 0
        vocab = set()
        for t in texts:
            toks = t.split()
            token_count += len(toks)
            vocab.update(toks)
        return token_count, len(vocab)

    scenario_rows = []
    for s in scenario_defs:
        spk_set = set(s["speaker_ids"])
        sub = df[
            (df["speaker_id"].isin(spk_set))
            & (df["quality_score"] >= s["quality_floor"])
            & (df["duration"] <= s["duration_ceiling"])
        ].copy()

        if sub.empty:
            scenario_rows.append(
                {
                    **{k: v for k, v in s.items() if k != "speaker_ids"},
                    "clips": 0,
                    "hours": 0.0,
                    "pct_total_hours": 0.0,
                    "selected_speakers": 0,
                    "mean_quality": 0.0,
                    "p10_quality": 0.0,
                    "mean_duration": 0.0,
                    "top1_hours_share": 0.0,
                    "top5_hours_share": 0.0,
                    "hours_per_speaker_mean": 0.0,
                    "hours_gap_to_target": round(abs(TARGET_HOURS - 0.0), 3),
                    "word_tokens": 0,
                    "word_types": 0,
                }
            )
            continue

        hours = float(sub["duration"].sum()) / 3600.0
        clips = int(len(sub))
        selected_speakers = int(sub["speaker_id"].nunique())
        mean_quality = float(sub["quality_score"].mean())
        p10_quality = float(np.percentile(sub["quality_score"].values, 10))
        mean_duration = float(sub["duration"].mean())

        hours_by_speaker = (
            sub.groupby("speaker_id")["duration"].sum().sort_values(ascending=False)
            / 3600.0
        )
        top1_hours_share = float(hours_by_speaker.head(1).sum() / hours)
        top5_hours_share = float(hours_by_speaker.head(5).sum() / hours)
        hours_per_speaker_mean = float(hours_by_speaker.mean())

        word_tokens, word_types = add_word_metrics(sub)

        scenario_rows.append(
            {
                **{k: v for k, v in s.items() if k != "speaker_ids"},
                "clips": clips,
                "hours": round(hours, 3),
                "pct_total_hours": round((hours / total_hours) * 100.0, 1),
                "selected_speakers": selected_speakers,
                "mean_quality": round(mean_quality, 2),
                "p10_quality": round(p10_quality, 2),
                "mean_duration": round(mean_duration, 2),
                "top1_hours_share": round(top1_hours_share, 3),
                "top5_hours_share": round(top5_hours_share, 3),
                "hours_per_speaker_mean": round(hours_per_speaker_mean, 3),
                "hours_gap_to_target": round(abs(TARGET_HOURS - hours), 3),
                "word_tokens": int(word_tokens),
                "word_types": int(word_types),
            }
        )

    matrix_df = pd.DataFrame(scenario_rows).sort_values(
        ["hours_gap_to_target", "mean_quality", "selected_speakers"],
        ascending=[True, False, False],
    )

    # Practical recommendation shortlist
    # Near 40h, with non-trivial quality and diversity.
    shortlist = matrix_df[
        (matrix_df["hours"] >= 30.0)
        & (matrix_df["hours"] <= 45.0)
        & (matrix_df["mean_quality"] >= 66.0)
        & (matrix_df["selected_speakers"] >= 12)
    ].copy()
    shortlist = shortlist.head(10)

    print("\nTop 10 scenarios nearest 40h:")
    print(
        "  "
        + " | ".join(
            [
                "type",
                "pool",
                "q",
                "d<=",
                "hours",
                "spk",
                "mean_q",
                "top1%",
                "top5%",
            ]
        )
    )
    print("  " + "-" * 96)
    for _, r in matrix_df.head(10).iterrows():
        print(
            "  "
            + f"{r['scenario_type']:<16} | {r['speaker_pool']:<20} | "
            + f"{r['quality_label']:<4} | {r['duration_ceiling']:<4.0f} | "
            + f"{r['hours']:<6.2f} | {int(r['selected_speakers']):<3d} | "
            + f"{r['mean_quality']:<6.2f} | {r['top1_hours_share'] * 100:>5.1f} | "
            + f"{r['top5_hours_share'] * 100:>5.1f}"
        )

    os.makedirs("/data/reports", exist_ok=True)
    matrix_df.to_csv(REPORT_CSV_PATH, index=False)

    report = {
        "timestamp": datetime.now().isoformat(),
        "source": ANNOTATED_PATH,
        "dataset": {
            "clips": int(len(df)),
            "hours": round(total_hours, 3),
            "unique_speakers": int(df["speaker_id"].nunique()),
            "duration_floor_s": DURATION_FLOOR,
        },
        "target_hours": TARGET_HOURS,
        "quality_percentiles": {k: round(v, 2) for k, v in percentile_map.items()},
        "speaker_sets": {
            "primary_only": PRIMARY_SPEAKERS,
            "primary_plus_conditional": sorted(
                list(set(PRIMARY_SPEAKERS + CONDITIONAL_SPEAKERS))
            ),
            "top_n": {k: v for k, v in top_n_speaker_sets.items()},
        },
        "scenario_count": int(len(matrix_df)),
        "top_nearest_to_target": matrix_df.head(20).to_dict(orient="records"),
        "shortlist_30_to_45h": shortlist.to_dict(orient="records"),
        "paths": {
            "matrix_csv": REPORT_CSV_PATH,
            "report_json": REPORT_JSON_PATH,
        },
    }

    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    data_vol.commit()

    print("\nAudit complete")
    print(f"  Matrix CSV: {REPORT_CSV_PATH}")
    print(f"  Report JSON:{REPORT_JSON_PATH}")


@app.local_entrypoint()
def main():
    audit_tts_expansion_options.remote()
