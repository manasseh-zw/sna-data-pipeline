"""
Probe the trained logistic regression gender classifier over all speaker sample WAVs.

Outputs:
  src/tests/audio/audit_speaker/gender_probe/
    female/          — copies of clips predicted Female (named: {spk_prefix}_{filename})
    male/            — copies of clips predicted Male
    unknown/         — low-confidence clips (confidence < 0.65)
    probe_clips.csv  — per-clip predictions + confidence scores
    probe_speakers.csv — per-source-speaker summary
    probe_report.txt — human-readable report with flagged speakers

Also cross-references clip_assignments.csv from the v2 audit run if present, to
show whether previously-Mixed clusters are now resolved correctly.

Run from repo root:
  python src/tests/audio/audit_speaker/probe_gender_classifier.py
"""

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

SCRIPT_DIR   = Path(__file__).resolve().parent
SAMPLES_DIR  = Path("src/tests/audio/speaker_samples")
PROBE_DIR    = SCRIPT_DIR / "gender_probe"
GENDER_PKL   = SCRIPT_DIR / "gender_classifier_ecapa.pkl"

V2_CLIPS_CSV = Path("src/tests/audio/speaker_samples/audit_clusters_v2/clip_assignments.csv")

ECAPA_SR        = 16_000
ECAPA_HUB_ID    = "speechbrain/spkrec-ecapa-voxceleb"
CONF_THRESHOLD  = 0.65

TIER_SUFFIX_RE = re.compile(r"__tier-(low|mid|high)$")


def _load_dotenv_from_repo() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        env_file = parent / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            return


def _discover_ecapa_local() -> Path | None:
    roots = [Path.cwd()]
    here = Path(__file__).resolve()
    for p in here.parents:
        roots.append(p)
    seen: set[str] = set()
    for root in roots:
        base = root / "models" / "spkrec-ecapa-voxceleb"
        key = str(base.resolve())
        if key in seen:
            continue
        seen.add(key)
        if base.is_dir() and (base / "hyperparams.yaml").is_file():
            return base.resolve()
    return None


def load_wav_16k(wav_path: Path) -> np.ndarray:
    arr, sr = sf.read(str(wav_path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != ECAPA_SR:
        t = torch.from_numpy(arr).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=ECAPA_SR)
        arr = t.squeeze(0).numpy()
    return arr


def main() -> None:
    _load_dotenv_from_repo()

    import pickle
    if not GENDER_PKL.is_file():
        print(f"❌ No classifier found at {GENDER_PKL}")
        print("   Run train_gender_classifier.py first.")
        return

    with open(GENDER_PKL, "rb") as f:
        payload = pickle.load(f)
    clf = payload["model"]
    meta = payload.get("metadata", {})

    print("=" * 60)
    print("GENDER CLASSIFIER PROBE")
    print("=" * 60)
    print(f"  Classifier:    logistic regression on ECAPA embeddings")
    print(f"  Training data: female={meta.get('n_female_clips','?')}  male={meta.get('n_male_clips','?')}")
    cv_acc = meta.get("cv_accuracy")
    if cv_acc is not None:
        print(f"  CV accuracy:   {cv_acc:.1%}")
    print(f"  Conf threshold: {CONF_THRESHOLD}")
    print(f"  Input:  {SAMPLES_DIR}")
    print(f"  Output: {PROBE_DIR}")

    if not SAMPLES_DIR.exists():
        print(f"\n❌ Missing input folder: {SAMPLES_DIR}")
        return

    # ── Collect WAVs ──────────────────────────────────────────────────────────
    speaker_dirs = [
        p for p in sorted(SAMPLES_DIR.iterdir())
        if p.is_dir() and not p.name.startswith("audit") and not p.name.startswith("gender")
    ]
    print(f"\n🔎 Scanning {len(speaker_dirs)} speaker folders ...")

    wav_records = []
    for spk_dir in speaker_dirs:
        folder_name = spk_dir.name
        parts  = folder_name.split("_", 1)
        prefix = (parts[0] + "_" + parts[1][:8]) if len(parts) > 1 else folder_name[:10]

        meta_by_sid: dict[str, dict] = {}
        meta_path = spk_dir / "metadata.csv"
        if meta_path.exists():
            for _, row in pd.read_csv(meta_path).iterrows():
                meta_by_sid[str(row["source_id"])] = row.to_dict()

        for wav_path in sorted(spk_dir.glob("sna_*.wav")):
            stem          = wav_path.stem
            source_id     = stem[4:] if stem.startswith("sna_") else stem
            base_sid      = TIER_SUFFIX_RE.sub("", source_id)
            row_meta      = meta_by_sid.get(base_sid, {})
            wav_records.append({
                "path":              wav_path,
                "source_speaker_id": folder_name,
                "folder_prefix":     prefix,
                "source_id":         source_id,
                "original_filename": wav_path.name,
                "gender_original":   str(row_meta.get("gender", "")),
            })

    print(f"   {len(wav_records)} WAVs across {len(speaker_dirs)} speakers")

    # ── Load ECAPA ────────────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🧠 Loading ECAPA-TDNN on {device} ...")

    from speechbrain.inference.speaker import EncoderClassifier

    ecapa_local = _discover_ecapa_local()
    if ecapa_local:
        print(f"   Local: {ecapa_local}")
        encoder = EncoderClassifier.from_hparams(
            source=str(ecapa_local),
            run_opts={"device": str(device)},
            savedir=str(ecapa_local),
        )
    else:
        print(f"   Hub: {ECAPA_HUB_ID}")
        encoder = EncoderClassifier.from_hparams(
            source=ECAPA_HUB_ID,
            run_opts={"device": str(device)},
            savedir="/tmp/ecapa_cache_probe",
        )
    encoder.eval()
    print("   ECAPA-TDNN ✅")

    # ── Prepare output folders ────────────────────────────────────────────────
    for sub in ("female", "male", "unknown"):
        d = PROBE_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # ── Classify + copy ───────────────────────────────────────────────────────
    print(f"\n🔬 Classifying {len(wav_records)} clips ...")

    clip_rows  = []
    n_failed   = 0

    for rec in tqdm(wav_records, desc="Probe", unit="clip"):
        try:
            arr   = load_wav_16k(rec["path"])
            wav_t = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder.encode_batch(wav_t)
            vec = emb.squeeze().cpu().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-10)

            proba    = clf.predict_proba(vec.reshape(1, -1))[0]
            female_p = float(proba[0])
            male_p   = float(proba[1])
            conf     = max(female_p, male_p)

            if conf < CONF_THRESHOLD:
                gender_pred = "Unknown"
                bucket      = "unknown"
            elif female_p > male_p:
                gender_pred = "Female"
                bucket      = "female"
            else:
                gender_pred = "Male"
                bucket      = "male"

            dest_name = f"{rec['folder_prefix']}_{rec['original_filename']}"
            shutil.copy2(rec["path"], PROBE_DIR / bucket / dest_name)

            clip_rows.append({
                "source_speaker_id": rec["source_speaker_id"],
                "source_id":         rec["source_id"],
                "original_filename": rec["original_filename"],
                "gender_predicted":  gender_pred,
                "confidence":        round(conf, 4),
                "female_prob":       round(female_p, 4),
                "male_prob":         round(male_p, 4),
                "gender_original":   rec["gender_original"],
            })

        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"   ⚠️  {rec['path'].name}: {e}")
            clip_rows.append({
                "source_speaker_id": rec["source_speaker_id"],
                "source_id":         rec["source_id"],
                "original_filename": rec["original_filename"],
                "gender_predicted":  "ERROR",
                "confidence":        0.0,
                "female_prob":       0.0,
                "male_prob":         0.0,
                "gender_original":   rec["gender_original"],
            })

    clips_df = pd.DataFrame(clip_rows)
    clips_df.to_csv(PROBE_DIR / "probe_clips.csv", index=False)

    # ── Per-speaker summary ───────────────────────────────────────────────────
    spk_rows = []
    for spk_id, grp in clips_df.groupby("source_speaker_id"):
        total   = len(grp)
        n_f     = int((grp["gender_predicted"] == "Female").sum())
        n_m     = int((grp["gender_predicted"] == "Male").sum())
        n_u     = int((grp["gender_predicted"] == "Unknown").sum())
        n_err   = int((grp["gender_predicted"] == "ERROR").sum())
        dominant = "Female" if n_f >= n_m else "Male"
        pct_dom  = max(n_f, n_m) / max(total - n_u - n_err, 1)
        avg_conf = grp[grp["gender_predicted"] != "ERROR"]["confidence"].mean()
        orig_g   = grp["gender_original"].dropna().unique().tolist()

        flags = []
        if n_u / total > 0.15:
            flags.append("HIGH_UNKNOWN")
        if n_f > 0 and n_m > 0 and min(n_f, n_m) / total > 0.20:
            flags.append("SPLIT")

        spk_rows.append({
            "source_speaker_id": spk_id,
            "total_clips":       total,
            "female":            n_f,
            "male":              n_m,
            "unknown":           n_u,
            "dominant_gender":   dominant,
            "pct_dominant":      round(pct_dom, 3),
            "avg_confidence":    round(avg_conf, 3),
            "gender_original":   "|".join(str(g) for g in orig_g if g),
            "flags":             " ".join(flags),
        })

    spk_df = pd.DataFrame(spk_rows).sort_values("pct_dominant")
    spk_df.to_csv(PROBE_DIR / "probe_speakers.csv", index=False)

    # ── Cross-reference v2 Mixed clusters ─────────────────────────────────────
    mixed_section = ""
    if V2_CLIPS_CSV.is_file():
        v2_df   = pd.read_csv(V2_CLIPS_CSV)
        mixed   = v2_df[v2_df["cluster_folder"].str.contains("Mixed")]
        if not mixed.empty:
            probe_map = {
                row["original_filename"]: row
                for _, row in clips_df.iterrows()
            }
            lines = [
                "",
                "PREVIOUSLY MIXED CLUSTERS (v2) — NEW PREDICTIONS:",
                f"  {'Cluster':<22} {'File':<45} {'v2 pred':>8}  {'New pred':>8}  {'Conf':>6}",
                "  " + "-" * 90,
            ]
            for _, row in mixed.iterrows():
                new = probe_map.get(row["original_filename"])
                if new is not None:
                    lines.append(
                        f"  {str(row['cluster_folder']):<22} "
                        f"{str(row['original_filename']):<45} "
                        f"{str(row['gender_predicted']):>8}  "
                        f"{str(new['gender_predicted']):>8}  "
                        f"{new['confidence']:>6.3f}"
                    )
            mixed_section = "\n".join(lines)

    # ── Report ────────────────────────────────────────────────────────────────
    n_female  = int((clips_df["gender_predicted"] == "Female").sum())
    n_male    = int((clips_df["gender_predicted"] == "Male").sum())
    n_unknown = int((clips_df["gender_predicted"] == "Unknown").sum())
    flagged   = spk_df[spk_df["flags"] != ""]
    split_spk = spk_df[spk_df["flags"].str.contains("SPLIT", na=False)]

    lines = [
        "=" * 60,
        "GENDER PROBE REPORT",
        "=" * 60,
        f"  Total clips:     {len(clips_df)}",
        f"  Female:          {n_female} ({n_female/len(clips_df)*100:.1f}%)",
        f"  Male:            {n_male} ({n_male/len(clips_df)*100:.1f}%)",
        f"  Unknown (<{CONF_THRESHOLD:.0%} conf): {n_unknown} ({n_unknown/len(clips_df)*100:.1f}%)",
        f"  Failed:          {n_failed}",
        "",
        f"  Speakers total:  {len(spk_df)}",
        f"  Flagged SPLIT:   {len(split_spk)}  (>20% clips disagree with dominant gender)",
        f"  Flagged HIGH_UNK:{len(spk_df[spk_df['flags'].str.contains('HIGH_UNKNOWN', na=False)])}  (>15% unknown clips)",
        "",
        "SPEAKERS WITH SPLIT PREDICTIONS (inspect by ear):",
        f"  {'Speaker ID':<45} {'F':>4} {'M':>4} {'U':>4}  {'Dom%':>6}  {'AvgConf':>8}  Orig",
        "  " + "-" * 80,
    ]

    for _, row in split_spk.sort_values("pct_dominant").iterrows():
        lines.append(
            f"  {str(row['source_speaker_id']):<45} "
            f"{int(row['female']):>4} "
            f"{int(row['male']):>4} "
            f"{int(row['unknown']):>4}  "
            f"{row['pct_dominant']:>6.1%}  "
            f"{row['avg_confidence']:>8.3f}  "
            f"{row['gender_original']}"
        )

    if not split_spk.empty:
        lines.append("")

    lines += [
        "ALL SPEAKERS (sorted by confidence, lowest first):",
        f"  {'Speaker ID':<45} {'F':>4} {'M':>4} {'U':>4}  {'Dom':>6}  {'Dom%':>6}  {'AvgConf':>8}  Flags",
        "  " + "-" * 90,
    ]
    for _, row in spk_df.iterrows():
        lines.append(
            f"  {str(row['source_speaker_id']):<45} "
            f"{int(row['female']):>4} "
            f"{int(row['male']):>4} "
            f"{int(row['unknown']):>4}  "
            f"{str(row['dominant_gender']):>6}  "
            f"{row['pct_dominant']:>6.1%}  "
            f"{row['avg_confidence']:>8.3f}  "
            f"{row['flags']}"
        )

    if mixed_section:
        lines.append(mixed_section)

    lines += [
        "",
        f"Output: {PROBE_DIR}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(PROBE_DIR / "probe_report.txt", "w") as f:
        f.write(report)

    print(f"\n✅ All outputs → {PROBE_DIR}/")


if __name__ == "__main__":
    main()
