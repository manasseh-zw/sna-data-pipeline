import modal

app = modal.App("sna-rebuild-annotated")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "datasets[audio]",
        "soundfile",
        "numpy",
        "pandas",
        "pyloudnorm",
        "tqdm",
    )
)

TARGET_LUFS = -23.0
SKIP_TOLERANCE_LU = 1.0
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1

SOURCE_DATASET_PATH = "/data/refined"
MAPPING_CSV_PATH = "/data/relabel/relabel_mapping.csv"

WAV_CACHE_DIR = "/data/wav_cache"
WAV_NORMALIZED_DIR = "/data/wav_normalised"

TMP_DATASET_PATH = "/data/sna_annotated_tmp"
FINAL_DATASET_PATH = "/data/sna_annotated"
PREV_DATASET_PATH = "/data/sna_annotated_prev"

REPORT_PATH = "/data/reports/rebuild_annotated_audit.json"


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=5400,
    volumes={"/data": data_vol},
)
def rebuild_annotated():
    import json
    import os
    import random
    import shutil
    from collections import defaultdict
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import pyloudnorm as pyln
    import soundfile as sf
    from datasets import Audio as AudioFeature
    from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
    from tqdm import tqdm

    print("=" * 72)
    print("SNA DATA PIPELINE - REBUILD ANNOTATED")
    print("=" * 72)

    print("\nStep 0 - Cleanup old directory")
    if os.path.exists("/data/final"):
        shutil.rmtree("/data/final")
        print("   Deleted /data/final")
    else:
        print("   /data/final not present, nothing to delete")

    print("\nStep 1 - Load relabel mapping and drop noise")
    mapping_df = pd.read_csv(MAPPING_CSV_PATH)
    mapping_df["source_id"] = mapping_df["source_id"].astype(str)
    input_mapping_rows = len(mapping_df)
    mapping_df = mapping_df[mapping_df["cluster_id"] != -1].copy()
    noise_dropped = input_mapping_rows - len(mapping_df)
    print(f"   Mapping rows: {input_mapping_rows}")
    print(f"   Noise rows dropped (cluster_id == -1): {noise_dropped}")

    print("\nStep 2 - Load source dataset")
    ds = load_from_disk(SOURCE_DATASET_PATH)
    input_clips = len(ds)
    print(f"   Loaded {input_clips} clips from {SOURCE_DATASET_PATH}")

    keep_source_cols = [
        "source_id",
        "transcription",
        "language",
        "has_punctuation",
        "snr_db",
        "speech_ratio",
        "quality_score",
        "duration",
    ]
    missing_cols = [c for c in keep_source_cols if c not in ds.column_names]
    if missing_cols:
        raise RuntimeError(f"Missing expected source columns: {missing_cols}")

    source_df = ds.select_columns(keep_source_cols).to_pandas()
    source_df["source_id"] = source_df["source_id"].astype(str)

    print("\nStep 3 - Join mapping and keep cleaned schema")
    clean_df = source_df.merge(mapping_df, on="source_id", how="inner")

    clean_df = clean_df.rename(
        columns={
            "cluster_id": "speaker_id",
            "cluster_gender": "gender",
        }
    )

    for col in ["gender_predicted", "gender_confidence", "noise_rescued", "flag"]:
        if col in clean_df.columns:
            clean_df = clean_df.drop(columns=[col])

    clean_df = clean_df[
        [
            "source_id",
            "transcription",
            "language",
            "has_punctuation",
            "snr_db",
            "speech_ratio",
            "quality_score",
            "duration",
            "speaker_id",
            "gender",
            "cluster_confidence",
        ]
    ].copy()

    clean_df = clean_df.rename(
        columns={"cluster_confidence": "speaker_assignment_confidence"}
    )

    print(f"   Rows after join/noise drop: {len(clean_df)}")

    print("\nStep 4 - Recompute speaker_clip_count")
    speaker_counts = clean_df["speaker_id"].value_counts().to_dict()
    clean_df["speaker_clip_count"] = (
        clean_df["speaker_id"].map(speaker_counts).astype("int32")
    )

    clean_df["speaker_id"] = clean_df["speaker_id"].astype("int32")
    clean_df["speaker_assignment_confidence"] = clean_df[
        "speaker_assignment_confidence"
    ].astype("float32")
    clean_df["snr_db"] = clean_df["snr_db"].astype("float32")
    clean_df["speech_ratio"] = clean_df["speech_ratio"].astype("float32")
    clean_df["quality_score"] = clean_df["quality_score"].astype("float32")
    clean_df["duration"] = clean_df["duration"].astype("float32")

    print("\nStep 5 - Loudness normalization to -23 LUFS")
    os.makedirs(WAV_NORMALIZED_DIR, exist_ok=True)
    os.makedirs("/data/reports", exist_ok=True)

    meter_cache = {}
    n_normalized = 0
    n_skipped = 0
    n_failed = 0
    n_output_peaks_at_zero = 0
    n_wav_written = 0

    input_lufs_values = []
    output_lufs_values = []
    gain_db_values = []

    def integrated_loudness(meter, arr):
        try:
            v = float(meter.integrated_loudness(arr))
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    for i, row in enumerate(
        tqdm(
            clean_df.itertuples(index=False), total=len(clean_df), desc="Normalize WAV"
        ),
        start=1,
    ):
        source_id = row.source_id
        in_wav = os.path.join(WAV_CACHE_DIR, f"{source_id}.wav")
        out_wav = os.path.join(WAV_NORMALIZED_DIR, f"{source_id}.wav")

        if not os.path.isfile(in_wav):
            raise FileNotFoundError(f"Missing wav_cache file: {in_wav}")

        if os.path.isfile(out_wav):
            arr_out, sr_out = sf.read(out_wav, dtype="float32")
            if arr_out.ndim > 1:
                arr_out = np.mean(arr_out, axis=1)

            arr_in, sr_in = sf.read(in_wav, dtype="float32")
            if arr_in.ndim > 1:
                arr_in = np.mean(arr_in, axis=1)

            meter_in = meter_cache.setdefault(sr_in, pyln.Meter(sr_in))
            meter_out = meter_cache.setdefault(sr_out, pyln.Meter(sr_out))
            in_lufs = integrated_loudness(meter_in, arr_in)
            out_lufs = integrated_loudness(meter_out, arr_out)

            if in_lufs is not None:
                input_lufs_values.append(in_lufs)
            if out_lufs is not None:
                output_lufs_values.append(out_lufs)

            if in_lufs is None or out_lufs is None:
                n_failed += 1
            else:
                gain_db = TARGET_LUFS - in_lufs
                gain_db_values.append(gain_db)
                if abs(gain_db) <= SKIP_TOLERANCE_LU:
                    n_skipped += 1
                else:
                    n_normalized += 1

            if float(np.max(np.abs(arr_out))) >= 1.0:
                n_output_peaks_at_zero += 1

        else:
            arr, sr = sf.read(in_wav, dtype="float32")
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            arr = np.nan_to_num(arr, copy=False)

            meter = meter_cache.setdefault(sr, pyln.Meter(sr))
            in_lufs = integrated_loudness(meter, arr)
            out_arr = arr

            if in_lufs is None:
                n_failed += 1
                out_lufs = None
            else:
                input_lufs_values.append(in_lufs)
                gain_db = TARGET_LUFS - in_lufs
                gain_db_values.append(gain_db)

                if abs(gain_db) <= SKIP_TOLERANCE_LU:
                    n_skipped += 1
                    out_arr = arr
                else:
                    n_normalized += 1
                    out_arr = pyln.normalize.loudness(arr, in_lufs, TARGET_LUFS)
                    out_arr = np.clip(out_arr, -1.0, 1.0).astype(np.float32)

                out_lufs = integrated_loudness(meter, out_arr)
                if out_lufs is not None:
                    output_lufs_values.append(out_lufs)

            if float(np.max(np.abs(out_arr))) >= 1.0:
                n_output_peaks_at_zero += 1

            sf.write(out_wav, out_arr, sr, subtype="FLOAT")
            n_wav_written += 1

            if i % 500 == 0:
                data_vol.commit()

    data_vol.commit()
    print(f"   WAVs newly written this run: {n_wav_written}")
    print(f"   clips_normalized: {n_normalized}")
    print(f"   clips_skipped_loudness: {n_skipped}")
    print(f"   clips_failed_loudness: {n_failed}")

    print("\nStep 6 - Build dataset object")
    clean_df["audio"] = clean_df["source_id"].map(
        lambda x: os.path.join(WAV_NORMALIZED_DIR, f"{x}.wav")
    )

    final_col_order = [
        "audio",
        "transcription",
        "source_id",
        "speaker_id",
        "speaker_clip_count",
        "language",
        "gender",
        "has_punctuation",
        "snr_db",
        "speech_ratio",
        "quality_score",
        "duration",
        "speaker_assignment_confidence",
    ]

    clean_df = clean_df[final_col_order]

    features = Features(
        {
            "audio": AudioFeature(sampling_rate=24000),
            "transcription": Value("string"),
            "source_id": Value("string"),
            "speaker_id": Value("int32"),
            "speaker_clip_count": Value("int32"),
            "language": Value("string"),
            "gender": Value("string"),
            "has_punctuation": Value("bool"),
            "snr_db": Value("float32"),
            "speech_ratio": Value("float32"),
            "quality_score": Value("float32"),
            "duration": Value("float32"),
            "speaker_assignment_confidence": Value("float32"),
        }
    )

    dataset = Dataset.from_pandas(clean_df, preserve_index=False)
    dataset = dataset.cast(features)

    print("\nStep 7 - Speaker-stratified split (80/10/10)")
    speaker_to_indices = defaultdict(list)
    for idx, spk in enumerate(dataset["speaker_id"]):
        speaker_to_indices[int(spk)].append(idx)

    rng = random.Random(SPLIT_SEED)
    train_idx, valid_idx, test_idx = [], [], []
    for _, idxs in speaker_to_indices.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(np.floor(n * TRAIN_RATIO))
        n_valid = int(np.floor(n * VALID_RATIO))
        if n_train == 0 and n > 0:
            n_train = 1
        train_idx.extend(idxs[:n_train])
        valid_idx.extend(idxs[n_train : n_train + n_valid])
        test_idx.extend(idxs[n_train + n_valid :])

    dataset_dict = DatasetDict(
        {
            "train": dataset.select(sorted(train_idx)),
            "validation": dataset.select(sorted(valid_idx)),
            "test": dataset.select(sorted(test_idx)),
        }
    )

    print("\nStep 8 - Save dataset to volume")
    for path in (TMP_DATASET_PATH, PREV_DATASET_PATH):
        if os.path.exists(path):
            shutil.rmtree(path)

    dataset_dict.save_to_disk(TMP_DATASET_PATH)
    if os.path.exists(FINAL_DATASET_PATH):
        os.replace(FINAL_DATASET_PATH, PREV_DATASET_PATH)
    os.replace(TMP_DATASET_PATH, FINAL_DATASET_PATH)
    if os.path.exists(PREV_DATASET_PATH):
        shutil.rmtree(PREV_DATASET_PATH)

    data_vol.commit()

    print("\nStep 9 - Write audit")
    total_hours = float(clean_df["duration"].sum()) / 3600.0

    def _stats(values):
        if not values:
            return {"mean": None, "min": None, "max": None, "std": None}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": round(float(arr.mean()), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
            "std": round(float(arr.std()), 3),
        }

    gender_dist = clean_df["gender"].value_counts().to_dict()

    audit = {
        "phase": "rebuild_annotated",
        "timestamp": datetime.now().isoformat(),
        "source_path": SOURCE_DATASET_PATH,
        "input_clips": int(input_clips),
        "noise_dropped": int(noise_dropped),
        "final_clips": int(len(clean_df)),
        "unique_speakers": int(clean_df["speaker_id"].nunique()),
        "total_hours": round(float(total_hours), 3),
        "loudness_target_lufs": TARGET_LUFS,
        "loudness_skip_tolerance_lu": SKIP_TOLERANCE_LU,
        "clips_normalised": int(n_normalized),
        "clips_skipped_loudness": int(n_skipped),
        "clips_failed_loudness": int(n_failed),
        "clips_peak_hit_0dbfs": int(n_output_peaks_at_zero),
        "loudness_input_lufs": _stats(input_lufs_values),
        "loudness_output_lufs": _stats(output_lufs_values),
        "loudness_gain_db": _stats(gain_db_values),
        "splits": {
            "train": int(len(dataset_dict["train"])),
            "validation": int(len(dataset_dict["validation"])),
            "test": int(len(dataset_dict["test"])),
        },
        "gender_distribution": {
            "Female": int(gender_dist.get("Female", 0)),
            "Male": int(gender_dist.get("Male", 0)),
            "Unknown": int(gender_dist.get("Unknown", 0)),
        },
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(audit, f, indent=2)

    data_vol.commit()

    print("=" * 72)
    print("REBUILD ANNOTATED COMPLETE")
    print(f"Dataset path: {FINAL_DATASET_PATH}")
    print(f"Audit: {REPORT_PATH}")
    print("=" * 72)


@app.local_entrypoint()
def main():
    rebuild_annotated.remote()
