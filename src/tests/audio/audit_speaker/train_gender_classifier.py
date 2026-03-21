"""
Train a logistic regression gender classifier on top of ECAPA-TDNN embeddings.

Place labeled WAV clips in:
  src/tests/audio/audit_speaker/gender_training_data/female/   ← any female Shona clips
  src/tests/audio/audit_speaker/gender_training_data/male/     ← any male Shona clips

Good sources for training data:
  - Copy clips from clean audit clusters (e.g. cluster_08_Male, cluster_09_Female)
    that had unanimous gender votes.
  - Use clips from known-gender speakers listed in .docs/context.md.

Output:
  src/tests/audio/audit_speaker/gender_classifier_ecapa.pkl

The main audit_speaker_clusters.py script loads this file automatically if present,
replacing the Wav2Vec2 gender model entirely.

Run from repo root:
  python src/tests/audio/audit_speaker/train_gender_classifier.py
"""

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DATA_DIR = SCRIPT_DIR / "gender_training_data"
OUTPUT_PKL = SCRIPT_DIR / "gender_classifier_ecapa.pkl"

ECAPA_SR = 16_000
ECAPA_HUB_ID = "speechbrain/spkrec-ecapa-voxceleb"

LABEL_MAP = {"female": 0, "male": 1}
LABEL_NAMES = {0: "Female", 1: "Male"}


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


def extract_embeddings(wav_paths: list[Path], encoder, device: torch.device) -> np.ndarray:
    embeddings = []
    for path in tqdm(wav_paths, desc="Extracting embeddings", unit="clip"):
        arr = load_wav_16k(path)
        wav_t = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder.encode_batch(wav_t)
        vec = emb.squeeze().cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        embeddings.append(vec)
    return np.array(embeddings, dtype=np.float32)


def main() -> None:
    _load_dotenv_from_repo()

    female_dir = TRAINING_DATA_DIR / "female"
    male_dir = TRAINING_DATA_DIR / "male"

    female_wavs = sorted(female_dir.glob("*.wav")) if female_dir.is_dir() else []
    male_wavs = sorted(male_dir.glob("*.wav")) if male_dir.is_dir() else []

    print(f"Training data:")
    print(f"  Female clips: {len(female_wavs)}")
    print(f"  Male clips:   {len(male_wavs)}")

    if len(female_wavs) < 5 or len(male_wavs) < 5:
        print(
            f"\n❌ Need at least 5 clips per gender. "
            f"Add WAVs to:\n"
            f"   {female_dir}\n"
            f"   {male_dir}"
        )
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🧠 Loading ECAPA-TDNN on {device} ...")

    from speechbrain.inference.speaker import EncoderClassifier

    ecapa_local = _discover_ecapa_local()
    if ecapa_local is not None:
        print(f"   Local model: {ecapa_local}")
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
            savedir="/tmp/ecapa_cache_gender_train",
        )
    encoder.eval()
    print("   ECAPA-TDNN ✅")

    print("\n🎤 Extracting embeddings ...")
    female_embs = extract_embeddings(female_wavs, encoder, device)
    male_embs = extract_embeddings(male_wavs, encoder, device)

    X = np.concatenate([female_embs, male_embs], axis=0)
    y = np.array(
        [LABEL_MAP["female"]] * len(female_embs) + [LABEL_MAP["male"]] * len(male_embs)
    )

    print(f"\n   Total embeddings: {len(X)}  (female={len(female_embs)}, male={len(male_embs)})")

    print("\n📊 Cross-validation (5-fold stratified) ...")
    if len(X) >= 10:
        n_splits = min(5, len(female_embs), len(male_embs))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X, y):
            clf_cv = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf_cv.fit(X[train_idx], y[train_idx])
            cv_scores.append(clf_cv.score(X[val_idx], y[val_idx]))
        mean_acc = float(np.mean(cv_scores))
        print(f"   CV accuracy: {mean_acc:.3f}  (folds: {[f'{s:.3f}' for s in cv_scores]})")
    else:
        mean_acc = None
        print("   Too few clips for CV — skipping.")

    print("\n🔧 Fitting final model on all data ...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"   Training accuracy: {train_acc:.3f}")
    print("\n" + classification_report(y, clf.predict(X), target_names=["Female", "Male"]))

    payload = {
        "model": clf,
        "metadata": {
            "n_female_clips": len(female_embs),
            "n_male_clips": len(male_embs),
            "train_accuracy": round(train_acc, 4),
            "cv_accuracy": round(mean_acc, 4) if mean_acc is not None else None,
            "ecapa_hub_id": ECAPA_HUB_ID,
            "ecapa_local": str(ecapa_local) if ecapa_local else None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n✅ Saved → {OUTPUT_PKL}")
    print(f"   Female clips used: {len(female_embs)}")
    print(f"   Male clips used:   {len(male_embs)}")
    if mean_acc is not None:
        print(f"   CV accuracy:       {mean_acc:.1%}")


if __name__ == "__main__":
    main()
