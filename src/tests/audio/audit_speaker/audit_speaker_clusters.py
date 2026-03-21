"""
Local speaker clustering audit v2 — ECAPA-TDNN + HDBSCAN + noise rescue + gender classification.

Expects folder structure:
  speaker_samples/
    01_7erMyYrd9qflZJjN1wOPrEgigQu2/
      sna_sna_22490__tier-high.wav
      metadata.csv
    02_T33w3KIs.../
      ...

Output structure:
  speaker_samples/audit_clusters/
    cluster_01_Female/
      01_7erMyYrd_sna_sna_22490__tier-high.wav
    cluster_02_Male/
      ...
    cluster_NOISE/
      ...
    cluster_report.csv
    clip_assignments.csv
    cluster_summary.txt
    params_used.json

Install dependencies (from repo root):
  python -m venv .venv-audit-speaker
  source .venv-audit-speaker/bin/activate
  pip install -r src/tests/audio/audit_speaker/requirements.txt

Run from repo root (HF_TOKEN optional but recommended for Hub rate limits):
  export HF_TOKEN=...   # or put HF_TOKEN=... in a .env file at the repo root (loaded automatically)
  python src/tests/audio/audit_speaker/audit_speaker_clusters.py

Offline / manual downloads (skip slow in-script Hub pulls):
  Download folders once (e.g. with huggingface-cli or your download manager):
    hf download speechbrain/spkrec-ecapa-voxceleb --local-dir ./models/spkrec-ecapa-voxceleb
    hf download prithivMLmods/Common-Voice-Gender-Detection --local-dir ./models/Common-Voice-Gender-Detection
  If those paths exist under ./models/… (cwd or repo root), they are used automatically — no env vars required.
  Optional overrides in .env:
    SNA_AUDIT_ECAPA_DIR=...
    SNA_AUDIT_GENDER_DIR=...
  ECAPA needs hyperparams.yaml; gender needs config.json (and weights).
"""

import json
import os
import pickle
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

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path("src/tests/audio/speaker_samples")
AUDIT_DIR   = SAMPLES_DIR / "audit_clusters_v2"

HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES      = 5
MIN_CLUSTER_CLIPS_LOCAL  = 3

# Noise rescue — cosine similarity threshold for reassigning noise points
# to nearest cluster centroid. 0.75 is conservative and safe.
# Lower to 0.70 to rescue more points at slightly higher merge risk.
NOISE_RESCUE_THRESHOLD = 0.75

ECAPA_SR   = 16_000   # ECAPA-TDNN expects 16kHz
GENDER_SR  = 16_000   # gender model also expects 16kHz

TIER_SUFFIX_RE = re.compile(r"__tier-(low|mid|high)$")

GENDER_PKL = Path(__file__).resolve().parent / "gender_classifier_ecapa.pkl"


def _load_gender_pkl() -> tuple[object, dict] | None:
    """Load the logistic regression gender classifier if it exists. Returns (model, metadata) or None."""
    if not GENDER_PKL.is_file():
        return None
    with open(GENDER_PKL, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload.get("metadata", {})


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


def _env_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _env_model_dir(key: str) -> Path | None:
    raw = os.environ.get(key)
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return p if p.is_dir() else None


GENDER_MODEL_HUB_ID = "prithivMLmods/Common-Voice-Gender-Detection"
ECAPA_HUB_ID = "speechbrain/spkrec-ecapa-voxceleb"


def _discover_models_under_models_dir() -> tuple[Path | None, Path | None]:
    """Use <cwd-or-ancestor>/models/spkrec-ecapa-voxceleb and .../Common-Voice-Gender-Detection if present."""
    ecapa_name = "spkrec-ecapa-voxceleb"
    gender_name = "Common-Voice-Gender-Detection"
    roots: list[Path] = [Path.cwd()]
    here = Path(__file__).resolve()
    for p in here.parents:
        roots.append(p)

    ecapa: Path | None = None
    gender: Path | None = None
    seen_bases: set[str] = set()
    for root in roots:
        base = root / "models"
        if not base.is_dir():
            continue
        key = str(base.resolve())
        if key in seen_bases:
            continue
        seen_bases.add(key)
        e = base / ecapa_name
        g = base / gender_name
        if ecapa is None and e.is_dir() and (e / "hyperparams.yaml").is_file():
            ecapa = e.resolve()
        if gender is None and g.is_dir() and (g / "config.json").is_file():
            gender = g.resolve()
        if ecapa is not None and gender is not None:
            break
    return ecapa, gender


def _patch_speechbrain_hf_hub():
    import huggingface_hub as _h
    from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError

    _orig = _h.hf_hub_download
    _not_found = (RemoteEntryNotFoundError, EntryNotFoundError)

    def _wrapped(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("use_auth_token"))
        if kwargs.get("token") is False and _env_hf_token():
            kwargs["token"] = True
        hub_filename = kwargs.get("filename") if "filename" in kwargs else (args[1] if len(args) >= 2 else None)
        try:
            return _orig(*args, **kwargs)
        except _not_found as e:
            # SpeechBrain treats missing optional custom.py as ValueError; transformers expects Hub
            # EntryNotFoundError for other files (e.g. optional processor_config.json).
            if hub_filename == "custom.py":
                raise ValueError("File not found on HF hub") from e
            raise

    _h.hf_hub_download = _wrapped


def load_wav_16k(wav_path: Path) -> np.ndarray:
    arr, sr = sf.read(str(wav_path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != ECAPA_SR:
        t   = torch.from_numpy(arr).unsqueeze(0)
        t   = torchaudio.functional.resample(t, orig_freq=sr, new_freq=ECAPA_SR)
        arr = t.squeeze(0).numpy()
    return arr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    _load_dotenv_from_repo()
    _patch_speechbrain_hf_hub()
    if _env_hf_token():
        print("   Hugging Face Hub: HF_TOKEN / HUGGINGFACE_HUB_TOKEN detected (authenticated downloads)")
    import hdbscan as hdbscan_lib
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    if not SAMPLES_DIR.exists():
        print(f"\n❌ Missing input folder: {SAMPLES_DIR}")
        return

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SNA — LOCAL SPEAKER CLUSTERING AUDIT v2")
    print("  + noise rescue")
    print("  + gender classification")
    print("=" * 60)
    print(f"Input:  {SAMPLES_DIR}")
    print(f"Output: {AUDIT_DIR}")

    # ── 1. Discover WAVs ──────────────────────────────────────────────────────
    wav_records  = []
    speaker_dirs = [
        p for p in sorted(SAMPLES_DIR.iterdir())
        if p.is_dir() and not p.name.startswith("audit")
    ]
    print(f"\n🔎 Scanning {len(speaker_dirs)} speaker folders ...")

    for spk_dir in tqdm(speaker_dirs, desc="Discover WAVs", unit="spk"):
        folder_name       = spk_dir.name
        source_speaker_id = folder_name
        parts  = folder_name.split("_", 1)
        prefix = (parts[0] + "_" + parts[1][:8]) if len(parts) > 1 else folder_name[:10]

        meta_by_source_id = {}
        meta_path = spk_dir / "metadata.csv"
        if meta_path.exists():
            for _, row in pd.read_csv(meta_path).iterrows():
                meta_by_source_id[str(row["source_id"])] = row.to_dict()

        for wav_path in sorted(spk_dir.glob("sna_*.wav")):
            stem           = wav_path.stem
            source_id      = stem[4:] if stem.startswith("sna_") else stem
            base_source_id = TIER_SUFFIX_RE.sub("", source_id)
            meta           = meta_by_source_id.get(base_source_id, {})

            wav_records.append({
                "path":              wav_path,
                "source_speaker_id": source_speaker_id,
                "folder_prefix":     prefix,
                "source_id":         source_id,
                "base_source_id":    base_source_id,
                "original_filename": wav_path.name,
                "quality_score":     float(meta.get("quality_score", 0.0)),
                "snr_db":            float(meta.get("snr_db", 0.0)),
                "gender_original":   str(meta.get("gender", "")),
                "duration_s":        float(meta.get("duration_s", 0.0)),
            })

    n_source_speakers = len({r["source_speaker_id"] for r in wav_records})
    print(f"\n   {len(wav_records)} WAVs across {n_source_speakers} source speaker folders")

    if not wav_records:
        print(f"\n❌ No WAVs found under {SAMPLES_DIR}")
        return

    # ── 2. Load models ────────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🧠 Loading models on {device} ...")

    # Check for pre-trained logistic regression gender classifier
    gender_clf_result = _load_gender_pkl()
    if gender_clf_result is not None:
        gender_lr_clf, gender_lr_meta = gender_clf_result
        n_f = gender_lr_meta.get("n_female_clips", "?")
        n_m = gender_lr_meta.get("n_male_clips", "?")
        cv_acc = gender_lr_meta.get("cv_accuracy")
        acc_str = f", CV acc={cv_acc:.1%}" if cv_acc is not None else ""
        print(f"   Gender: logistic regression (ECAPA embeddings, female={n_f} male={n_m}{acc_str}) ✅")
        print(f"           Skipping Wav2Vec2 gender model.")
        gender_source = "ecapa_logreg"
        gender_processor = None
        gender_model = None
    else:
        gender_lr_clf = None
        gender_lr_meta = {}
        gender_processor = None
        gender_model = None

    ecapa_env = _env_model_dir("SNA_AUDIT_ECAPA_DIR")
    gender_env = _env_model_dir("SNA_AUDIT_GENDER_DIR")
    ecapa_auto, gender_auto = _discover_models_under_models_dir()
    if os.environ.get("SNA_AUDIT_ECAPA_DIR") and ecapa_env is None:
        print(f"\n❌ SNA_AUDIT_ECAPA_DIR is not a directory: {os.environ.get('SNA_AUDIT_ECAPA_DIR')!r}")
        return
    ecapa_local = ecapa_env if os.environ.get("SNA_AUDIT_ECAPA_DIR") else ecapa_auto

    if ecapa_local is not None and not (ecapa_local / "hyperparams.yaml").is_file():
        print(f"\n❌ Local ECAPA folder must contain hyperparams.yaml (SpeechBrain layout): {ecapa_local}")
        return

    # Speaker embedding model
    if ecapa_local is not None:
        _ec_src = "SNA_AUDIT_ECAPA_DIR" if ecapa_env else "models/spkrec-ecapa-voxceleb (auto)"
        print(f"   ECAPA: local ({_ec_src}, no Hub download) → {ecapa_local}")
        speaker_encoder = EncoderClassifier.from_hparams(
            source=str(ecapa_local),
            run_opts={"device": str(device)},
            savedir=str(ecapa_local),
        )
    else:
        print(f"   ECAPA: Hugging Face → {ECAPA_HUB_ID}")
        speaker_encoder = EncoderClassifier.from_hparams(
            source=ECAPA_HUB_ID,
            run_opts={"device": str(device)},
            savedir="/tmp/ecapa_cache_local",
        )
    speaker_encoder.eval()
    print("   ECAPA-TDNN speaker encoder ✅")

    # Wav2Vec2 gender model — only loaded if no .pkl classifier is available
    if gender_lr_clf is None:
        if os.environ.get("SNA_AUDIT_GENDER_DIR") and gender_env is None:
            print(f"\n❌ SNA_AUDIT_GENDER_DIR is not a directory: {os.environ.get('SNA_AUDIT_GENDER_DIR')!r}")
            return
        gender_local = gender_env if os.environ.get("SNA_AUDIT_GENDER_DIR") else gender_auto
        if gender_local is not None and not (gender_local / "config.json").is_file():
            print(f"\n❌ Local gender folder must contain config.json: {gender_local}")
            return
        if gender_local is not None:
            _g_src = "SNA_AUDIT_GENDER_DIR" if gender_env else "models/Common-Voice-Gender-Detection (auto)"
            print(f"   Gender: local ({_g_src}, no Hub download) → {gender_local}")
            gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(gender_local), local_files_only=True
            )
            gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                str(gender_local), local_files_only=True
            ).to(device)
        else:
            print(f"   Gender: Hugging Face → {GENDER_MODEL_HUB_ID}")
            gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(GENDER_MODEL_HUB_ID)
            gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                GENDER_MODEL_HUB_ID
            ).to(device)
        gender_model.eval()
        print("   Gender classifier (Common-Voice-Gender-Detection) ✅")
        gender_source = "local" if gender_local else "hub"
    else:
        gender_local = None

    ecapa_source = "local" if ecapa_local else "hub"

    # ── 3. Extract embeddings + predict gender per clip ───────────────────────
    print(f"\n🎤 Extracting embeddings + gender for {len(wav_records)} clips ...")

    embeddings    = []
    gender_preds  = []
    valid_records = []
    n_failed      = 0

    for rec in tqdm(wav_records, desc="Embed + Gender", unit="clip"):
        try:
            arr = load_wav_16k(rec["path"])

            # Speaker embedding
            wav_t = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = speaker_encoder.encode_batch(wav_t)   # [1, 1, 192]
            vec = emb.squeeze().cpu().numpy()
            embeddings.append(vec)

            # Gender prediction — logistic regression on ECAPA embedding if available
            if gender_lr_clf is not None:
                vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
                proba = gender_lr_clf.predict_proba(vec_norm.reshape(1, -1))[0]
                female_p, male_p = proba[0], proba[1]
                confidence = max(female_p, male_p)
                if confidence < 0.65:
                    gender_pred = "Unknown"
                elif female_p > male_p:
                    gender_pred = "Female"
                else:
                    gender_pred = "Male"
            else:
                inputs = gender_processor(
                    arr,
                    sampling_rate=GENDER_SR,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs["input_values"].to(device)
                with torch.no_grad():
                    logits = gender_model(input_values).logits   # [1, 2]
                probs      = torch.softmax(logits, dim=-1)[0]
                female_p   = probs[0].item()
                male_p     = probs[1].item()
                confidence = max(female_p, male_p)
                if confidence < 0.55:
                    gender_pred = "Unknown"
                elif female_p > male_p:
                    gender_pred = "Female"
                else:
                    gender_pred = "Male"

            gender_preds.append(gender_pred)
            valid_records.append(rec)

        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"   ⚠️  {rec['path'].name}: {e}")

    print(f"   {len(embeddings)} embeddings  ({n_failed} failed)")

    if not embeddings:
        print("\n❌ No embeddings extracted. Aborting.")
        return

    # L2-normalise → euclidean on normalised == cosine
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms       = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_norm    = emb_matrix / (norms + 1e-10)

    # ── 4. HDBSCAN clustering ─────────────────────────────────────────────────
    print(f"\n🔵 HDBSCAN "
          f"(min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={HDBSCAN_MIN_SAMPLES}) ...")

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels         = list(clusterer.fit_predict(emb_norm))
    n_raw          = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_before = int(sum(1 for l in labels if l == -1))
    print(f"   Raw clusters:        {n_raw}")
    print(f"   Noise before rescue: {n_noise_before} "
          f"({n_noise_before / len(labels) * 100:.1f}%)")

    # ── 5. Noise rescue ───────────────────────────────────────────────────────
    print(f"\n🔧 Noise rescue (threshold={NOISE_RESCUE_THRESHOLD}) ...")

    # Compute centroid for each cluster
    label_to_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl != -1:
            label_to_indices[lbl].append(i)

    cluster_centroids = {}
    for lbl, idxs in label_to_indices.items():
        centroid = emb_norm[idxs].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-10)
        cluster_centroids[lbl] = centroid

    n_rescued = 0
    for i, lbl in enumerate(labels):
        if lbl != -1:
            continue   # already assigned — skip

        best_lbl = -1
        best_sim = -1.0
        for clbl, centroid in cluster_centroids.items():
            sim = cosine_sim(emb_norm[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_lbl = clbl

        if best_sim >= NOISE_RESCUE_THRESHOLD:
            labels[i]  = best_lbl
            n_rescued += 1

    n_noise_after = int(sum(1 for l in labels if l == -1))
    print(f"   Rescued:             {n_rescued}")
    print(f"   Noise after rescue:  {n_noise_after} "
          f"({n_noise_after / len(labels) * 100:.1f}%)")

    # ── 6. Re-rank clusters by size descending ────────────────────────────────
    label_counts = defaultdict(int)
    for lbl in labels:
        if lbl != -1:
            label_counts[lbl] += 1

    sorted_labels = sorted(
        label_counts.keys(), key=lambda l: label_counts[l], reverse=True
    )
    raw_to_rank = {raw: rank + 1 for rank, raw in enumerate(sorted_labels)}

    # ── 7. Resolve gender per cluster by majority vote ────────────────────────
    cluster_gender_votes = defaultdict(lambda: defaultdict(int))
    for i, lbl in enumerate(labels):
        if lbl != -1:
            cluster_gender_votes[lbl][gender_preds[i]] += 1

    cluster_gender = {}
    for lbl, votes in cluster_gender_votes.items():
        total          = sum(votes.values())
        top_g, top_cnt = max(votes.items(), key=lambda x: x[1])
        confidence     = top_cnt / total
        if confidence >= 0.75:
            cluster_gender[lbl] = top_g
        elif "Male" in votes and "Female" in votes:
            cluster_gender[lbl] = "Mixed"
        else:
            cluster_gender[lbl] = "Unknown"

    # ── 8. Copy WAVs into labelled cluster folders ────────────────────────────
    print(f"\n📁 Writing cluster folders ...")

    # Clear previous run
    for d in AUDIT_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

    for rec, lbl in tqdm(
        zip(valid_records, labels),
        total=len(valid_records),
        desc="Copy WAVs",
        unit="clip",
    ):
        if lbl == -1:
            folder = AUDIT_DIR / "cluster_NOISE"
        else:
            rank   = raw_to_rank[lbl]
            gender = cluster_gender.get(lbl, "Unknown")
            folder = AUDIT_DIR / f"cluster_{rank:02d}_{gender}"

        folder.mkdir(parents=True, exist_ok=True)
        new_name = f"{rec['folder_prefix']}_{rec['original_filename']}"
        shutil.copy2(rec["path"], folder / new_name)

    print("   Done ✅")

    # ── 9. cluster_report.csv ─────────────────────────────────────────────────
    cluster_to_recs = defaultdict(list)
    for rec, lbl in zip(valid_records, labels):
        cluster_to_recs[lbl].append(rec)

    cluster_rows = []
    for lbl in sorted_labels + ([-1] if -1 in labels else []):
        recs            = cluster_to_recs[lbl]
        unique_spk      = sorted({r["source_speaker_id"] for r in recs})
        orig_genders    = sorted({r["gender_original"] for r in recs
                                   if r["gender_original"]})
        resolved_gender = cluster_gender.get(lbl, "Unknown") if lbl != -1 else "N/A"
        avg_q           = float(np.mean([r["quality_score"] for r in recs]))
        rank            = raw_to_rank.get(lbl, -1)
        folder_name     = (
            f"cluster_{rank:02d}_{resolved_gender}"
            if lbl != -1 else "cluster_NOISE"
        )
        votes    = cluster_gender_votes.get(lbl, {})
        vote_str = " | ".join(f"{g}:{c}" for g, c in sorted(votes.items()))

        flags = []
        if len(unique_spk) > 3:
            flags.append("MANY_SOURCES")
        if resolved_gender == "Mixed":
            flags.append("MIXED_GENDER")
        if len(orig_genders) > 1:
            flags.append("ORIG_LABEL_MIXED")

        cluster_rows.append({
            "cluster_folder":         folder_name,
            "hdbscan_label":          lbl,
            "clip_count":             len(recs),
            "unique_source_spk":      len(unique_spk),
            "source_speaker_ids":     "|".join(unique_spk),
            "gender_resolved":        resolved_gender,
            "gender_votes":           vote_str,
            "gender_original_labels": "|".join(orig_genders),
            "avg_quality_score":      round(avg_q, 2),
            "status":                 "NOISE" if lbl == -1 else (
                                      "SMALL" if len(recs) < MIN_CLUSTER_CLIPS_LOCAL
                                      else "OK"),
            "flags":                  " ".join(flags),
        })

    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(AUDIT_DIR / "cluster_report.csv", index=False)

    # ── 10. clip_assignments.csv ──────────────────────────────────────────────
    clip_rows = []
    for i, (rec, lbl) in enumerate(zip(valid_records, labels)):
        rank            = raw_to_rank.get(lbl, -1)
        resolved_gender = cluster_gender.get(lbl, "Unknown") if lbl != -1 else "N/A"
        folder_name     = (
            f"cluster_{rank:02d}_{resolved_gender}"
            if lbl != -1 else "cluster_NOISE"
        )
        clip_rows.append({
            "cluster_folder":    folder_name,
            "hdbscan_label":     lbl,
            "source_speaker_id": rec["source_speaker_id"],
            "source_id":         rec["source_id"],
            "original_filename": rec["original_filename"],
            "gender_predicted":  gender_preds[i],
            "gender_original":   rec["gender_original"],
            "quality_score":     rec["quality_score"],
            "snr_db":            rec["snr_db"],
            "duration_s":        rec["duration_s"],
        })

    pd.DataFrame(clip_rows).to_csv(AUDIT_DIR / "clip_assignments.csv", index=False)

    # ── 11. Summary ───────────────────────────────────────────────────────────
    ok      = cluster_df[cluster_df["status"] == "OK"]
    small   = cluster_df[cluster_df["status"] == "SMALL"]
    flagged = cluster_df[cluster_df["flags"] != ""]

    gender_summary = defaultdict(int)
    for lbl in sorted_labels:
        gender_summary[cluster_gender.get(lbl, "Unknown")] += 1

    lines = [
        "=" * 60,
        "AUDIT SUMMARY v2 (noise rescue + gender classification)",
        "=" * 60,
        f"  Input WAVs:             {len(wav_records)}",
        f"  Source speaker IDs:     {n_source_speakers}",
        f"  Embeddings extracted:   {len(embeddings)}",
        "",
        "  HDBSCAN results:",
        f"    Raw clusters:         {n_raw}",
        f"    Noise before rescue:  {n_noise_before} "
        f"({n_noise_before / len(labels) * 100:.1f}%)",
        f"    Rescued:              {n_rescued}",
        f"    Noise after rescue:   {n_noise_after} "
        f"({n_noise_after / len(labels) * 100:.1f}%)",
        "",
        f"  After MIN_CLUSTER_CLIPS={MIN_CLUSTER_CLIPS_LOCAL}:",
        f"    OK clusters:          {len(ok)}",
        f"    Small/dropped:        {len(small)}",
        f"    Flagged:              {len(flagged)}",
        "",
        "  Gender resolution (majority vote per cluster):",
    ]
    for g, cnt in sorted(gender_summary.items()):
        lines.append(f"    {g:<10} {cnt} clusters")

    lines += [
        "",
        "TOP CLUSTERS:",
        f"  {'Folder':<30} {'Clips':>5}  {'SrcSpks':>7}  "
        f"{'Gender':>8}  {'AvgQ':>5}  Flags",
        "  " + "-" * 70,
    ]

    for _, row in cluster_df[cluster_df["status"] != "NOISE"].head(25).iterrows():
        lines.append(
            f"  {str(row['cluster_folder']):<30} "
            f"{int(row['clip_count']):>5}  "
            f"{int(row['unique_source_spk']):>7}  "
            f"{str(row['gender_resolved']):>8}  "
            f"{row['avg_quality_score']:>5.1f}  "
            f"{row['flags']}"
        )

    lines += [
        "",
        "EXTRAPOLATION TO FULL 16,980 CLIPS:",
        f"  Est. noise after rescue: ~{int(16980 * n_noise_after / max(len(labels), 1))}",
        f"  Est. raw clusters:       ~{int(n_raw * 16980 / max(len(embeddings), 1))}",
        "",
        "TUNING GUIDE:",
        "  Noise% after rescue > 10% → lower HDBSCAN_MIN_CLUSTER_SIZE to 10",
        "  Same speaker split        → lower HDBSCAN_MIN_CLUSTER_SIZE",
        "  Different spks merged     → raise HDBSCAN_MIN_SAMPLES to 8",
        "  MIXED_GENDER on cluster   → inspect that folder by ear",
        "  ORIG_LABEL_MIXED          → original dataset had gender errors ✅",
        "",
        f"Output: {AUDIT_DIR}",
        "=" * 60,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(AUDIT_DIR / "cluster_summary.txt", "w") as f:
        f.write(summary)

    with open(AUDIT_DIR / "params_used.json", "w") as f:
        json.dump(
            {
                "HDBSCAN_MIN_CLUSTER_SIZE": HDBSCAN_MIN_CLUSTER_SIZE,
                "HDBSCAN_MIN_SAMPLES":      HDBSCAN_MIN_SAMPLES,
                "MIN_CLUSTER_CLIPS_LOCAL":  MIN_CLUSTER_CLIPS_LOCAL,
                "NOISE_RESCUE_THRESHOLD":   NOISE_RESCUE_THRESHOLD,
                "n_wav_files":              len(wav_records),
                "n_source_speakers":        n_source_speakers,
                "ecapa_source":             ecapa_source,
                "ecapa_hub_id":             None if ecapa_local else ECAPA_HUB_ID,
                "ecapa_local_dir":          str(ecapa_local) if ecapa_local else None,
                "ecapa_local_via":          (("env" if ecapa_env else "auto") if ecapa_local else None),
                "gender_source":            gender_source,
                "gender_hub_id":            None if (gender_local or gender_lr_clf) else GENDER_MODEL_HUB_ID,
                "gender_local_dir":         str(gender_local) if gender_local else None,
                "gender_local_via":         (("env" if gender_env else "auto") if gender_local else None),
                "gender_logreg_pkl":        str(GENDER_PKL) if gender_lr_clf else None,
                "gender_logreg_meta":       gender_lr_meta if gender_lr_clf else None,
            },
            f,
            indent=2,
        )

    print(f"\n✅ All outputs → {AUDIT_DIR}/")


if __name__ == "__main__":
    main()