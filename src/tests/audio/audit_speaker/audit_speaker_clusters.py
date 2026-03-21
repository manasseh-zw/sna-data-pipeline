"""
Local speaker clustering audit — runs ECAPA-TDNN + HDBSCAN on the
speaker_samples/ folder.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

SAMPLES_DIR = Path("src/tests/audio/speaker_samples")
AUDIT_DIR = SAMPLES_DIR / "audit_clusters"

HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5
MIN_CLUSTER_CLIPS_LOCAL = 3
ECAPA_SR = 16_000
TIER_SUFFIX_RE = re.compile(r"__tier-(low|mid|high)$")


def load_wav_16k(wav_path: Path) -> np.ndarray:
    arr, sr = sf.read(str(wav_path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != ECAPA_SR:
        t = torch.from_numpy(arr).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=ECAPA_SR)
        arr = t.squeeze(0).numpy()
    return arr


def main():
    import hdbscan as hdbscan_lib
    from speechbrain.inference.speaker import EncoderClassifier

    if not SAMPLES_DIR.exists():
        print(f"\n❌ Missing input folder: {SAMPLES_DIR}")
        return

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SNA — LOCAL SPEAKER CLUSTERING AUDIT (ISOLATED)")
    print("=" * 60)
    print(f"Input:  {SAMPLES_DIR}")
    print(f"Output: {AUDIT_DIR}")

    wav_records = []
    speaker_dirs = [
        p for p in sorted(SAMPLES_DIR.iterdir())
        if p.is_dir() and not p.name.startswith("audit")
    ]
    print(f"\n🔎 Scanning speaker folders ({len(speaker_dirs)} total) ...")

    for spk_dir in tqdm(speaker_dirs, desc="Discover WAVs", unit="spk"):
        folder_name = spk_dir.name
        source_speaker_id = folder_name

        parts = folder_name.split("_", 1)
        prefix = parts[0] + "_" + parts[1][:8] if len(parts) > 1 else folder_name[:10]

        meta_by_source_id = {}
        meta_path = spk_dir / "metadata.csv"
        if meta_path.exists():
            for _, row in pd.read_csv(meta_path).iterrows():
                meta_by_source_id[str(row["source_id"])] = row.to_dict()

        for wav_path in sorted(spk_dir.glob("sna_*.wav")):
            stem = wav_path.stem
            source_id = stem[4:] if stem.startswith("sna_") else stem
            base_source_id = TIER_SUFFIX_RE.sub("", source_id)
            meta = meta_by_source_id.get(base_source_id, {})

            wav_records.append({
                "path": wav_path,
                "source_speaker_id": source_speaker_id,
                "folder_prefix": prefix,
                "source_id": source_id,
                "base_source_id": base_source_id,
                "original_filename": wav_path.name,
                "quality_score": float(meta.get("quality_score", 0.0)),
                "snr_db": float(meta.get("snr_db", 0.0)),
                "gender": str(meta.get("gender", "")),
                "duration_s": float(meta.get("duration_s", 0.0)),
            })

    n_source_speakers = len({r["source_speaker_id"] for r in wav_records})
    print(f"\n   {len(wav_records)} WAVs across {n_source_speakers} source speaker folders")
    if not wav_records:
        print(f"\n❌ No WAVs found under {SAMPLES_DIR}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🧠 Loading ECAPA-TDNN on {device} ...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
        savedir="/tmp/ecapa_cache_local",
    )
    encoder.eval()
    print("   Loaded ✅")

    print(f"\n🎤 Extracting embeddings from {len(wav_records)} WAVs ...")
    embeddings = []
    valid_records = []
    n_failed = 0

    for rec in tqdm(wav_records, desc="Embedding", unit="clip"):
        try:
            arr = load_wav_16k(rec["path"])
            wav_t = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder.encode_batch(wav_t)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_records.append(rec)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"   ⚠️ {rec['path'].name}: {e}")

    print(f"   {len(embeddings)} embeddings ({n_failed} failed)")
    if not embeddings:
        print("\n❌ No embeddings extracted. Aborting clustering.")
        return

    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_norm = emb_matrix / (norms + 1e-10)

    print(
        f"\n🔵 HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
        f"min_samples={HDBSCAN_MIN_SAMPLES}) ..."
    )
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(emb_norm)

    n_raw = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"   Raw clusters: {n_raw}")
    print(f"   Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    label_counts = defaultdict(int)
    for lbl in labels:
        if lbl != -1:
            label_counts[lbl] += 1
    sorted_labels = sorted(label_counts.keys(), key=lambda l: label_counts[l], reverse=True)
    raw_to_rank = {raw: rank + 1 for rank, raw in enumerate(sorted_labels)}

    print(f"\n📁 Writing cluster folders to {AUDIT_DIR} ...")
    for d in AUDIT_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

    for rec, lbl in tqdm(zip(valid_records, labels), total=len(valid_records), desc="Copy to clusters", unit="clip"):
        if lbl == -1:
            cluster_folder = AUDIT_DIR / "cluster_NOISE"
        else:
            rank = raw_to_rank[lbl]
            cluster_folder = AUDIT_DIR / f"cluster_{rank:02d}"

        cluster_folder.mkdir(parents=True, exist_ok=True)
        new_name = f"{rec['folder_prefix']}_{rec['original_filename']}"
        shutil.copy2(rec["path"], cluster_folder / new_name)

    print("   Done ✅")

    print("\n🧾 Building cluster_report.csv ...")
    cluster_rows = []
    cluster_to_recs = defaultdict(list)
    for rec, lbl in zip(valid_records, labels):
        cluster_to_recs[lbl].append(rec)

    for lbl in (sorted_labels + ([-1] if -1 in labels else [])):
        recs = cluster_to_recs[lbl]
        unique_spk = sorted({r["source_speaker_id"] for r in recs})
        genders = sorted({r["gender"] for r in recs if r["gender"]})
        avg_q = float(np.mean([r["quality_score"] for r in recs]))
        rank = raw_to_rank.get(lbl, -1)
        folder_name = f"cluster_{rank:02d}" if lbl != -1 else "cluster_NOISE"

        flags = []
        if len(unique_spk) > 3:
            flags.append("MANY_SOURCES")
        if len(genders) > 1:
            flags.append("MIXED_GENDER")

        cluster_rows.append({
            "cluster_folder": folder_name,
            "hdbscan_label": lbl,
            "clip_count": len(recs),
            "unique_source_spk": len(unique_spk),
            "source_speaker_ids": "|".join(unique_spk),
            "genders": "|".join(genders),
            "avg_quality_score": round(avg_q, 2),
            "status": "NOISE" if lbl == -1 else ("SMALL" if len(recs) < MIN_CLUSTER_CLIPS_LOCAL else "OK"),
            "flags": " ".join(flags),
        })

    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(AUDIT_DIR / "cluster_report.csv", index=False)

    print("🧾 Building clip_assignments.csv ...")
    clip_rows = []
    for rec, lbl in tqdm(zip(valid_records, labels), total=len(valid_records), desc="Assignments", unit="clip"):
        rank = raw_to_rank.get(lbl, -1)
        folder_name = f"cluster_{rank:02d}" if lbl != -1 else "cluster_NOISE"
        clip_rows.append({
            "cluster_folder": folder_name,
            "hdbscan_label": lbl,
            "source_speaker_id": rec["source_speaker_id"],
            "source_id": rec["source_id"],
            "base_source_id": rec["base_source_id"],
            "original_filename": rec["original_filename"],
            "quality_score": rec["quality_score"],
            "snr_db": rec["snr_db"],
            "gender": rec["gender"],
            "duration_s": rec["duration_s"],
        })
    pd.DataFrame(clip_rows).to_csv(AUDIT_DIR / "clip_assignments.csv", index=False)

    ok = cluster_df[cluster_df["status"] == "OK"]
    small = cluster_df[cluster_df["status"] == "SMALL"]
    flagged = cluster_df[cluster_df["flags"] != ""]

    lines = [
        "=" * 60,
        "AUDIT SUMMARY",
        "=" * 60,
        f"  Input WAVs:           {len(wav_records)}",
        f"  Source speaker IDs:   {n_source_speakers}",
        f"  Embeddings extracted: {len(embeddings)}",
        "",
        "  HDBSCAN results:",
        f"    Raw clusters:       {n_raw}",
        f"    Noise points:       {n_noise} ({n_noise/len(labels)*100:.1f}%)",
        "",
        f"  After MIN_CLUSTER_CLIPS={MIN_CLUSTER_CLIPS_LOCAL}:",
        f"    OK clusters:        {len(ok)}",
        f"    Small/flagged:      {len(small)}",
        f"    Mixed gender flags: {len(flagged)}",
        "",
        "TOP CLUSTERS:",
        f"  {'Folder':<14} {'Clips':>5}  {'SrcSpks':>7}  {'Genders':<14}  {'AvgQ':>5}  Flags",
        "  " + "-" * 60,
    ]
    for _, row in cluster_df[cluster_df["status"] != "NOISE"].head(25).iterrows():
        lines.append(
            f"  {row['cluster_folder']:<14} "
            f"{int(row['clip_count']):>5}  "
            f"{int(row['unique_source_spk']):>7}  "
            f"{str(row['genders']):<14}  "
            f"{row['avg_quality_score']:>5.1f}  "
            f"{row['flags']}"
        )
    lines += ["", f"Output: {AUDIT_DIR}", "=" * 60]

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(AUDIT_DIR / "cluster_summary.txt", "w") as f:
        f.write(summary)

    with open(AUDIT_DIR / "params_used.json", "w") as f:
        json.dump(
            {
                "HDBSCAN_MIN_CLUSTER_SIZE": HDBSCAN_MIN_CLUSTER_SIZE,
                "HDBSCAN_MIN_SAMPLES": HDBSCAN_MIN_SAMPLES,
                "MIN_CLUSTER_CLIPS_LOCAL": MIN_CLUSTER_CLIPS_LOCAL,
                "n_wav_files": len(wav_records),
                "n_source_speakers": n_source_speakers,
            },
            f,
            indent=2,
        )

    print(f"\n✅ All outputs -> {AUDIT_DIR}/")


if __name__ == "__main__":
    main()
