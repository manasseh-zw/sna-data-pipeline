import modal

app = modal.App("sna-classify-speakers")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "datasets[audio]",
    "numpy",
    "pandas",
    "soundfile",
    "torch",
    "torchaudio",
    "speechbrain",
    "hdbscan",
    "scikit-learn",
)


WAV_CACHE_DIR = "/data/wav_cache"
MODEL_PKL_PATH = "/data/models/gender_classifier_ecapa.pkl"
REFINED_DATASET_PATH = "/data/refined"
RELABEL_DIR = "/data/relabel"
REPORTS_DIR = "/data/reports"
CHECKPOINT_DIR = "/data/relabel/checkpoints"
EMBED_CHECKPOINT_NPZ = "/data/relabel/checkpoints/embed_gender_checkpoint.npz"
EMBED_CHECKPOINT_META = "/data/relabel/checkpoints/embed_gender_checkpoint_meta.json"

AUDIO_SR_CACHE = 24_000
ECAPA_SR = 16_000
EMBED_BATCH_SIZE = 32
MIN_EMBED_MICROBATCH_SIZE = 2
EMBED_LENGTH_BUCKET_MULTIPLIER = 4
GENDER_UNKNOWN_THRESHOLD = 0.65
CHECKPOINT_EVERY_BATCHES = 20
PROGRESS_EVERY_WAV = 1000
PROGRESS_EVERY_BATCHES = 5
PROGRESS_EVERY_RESCUE = 2000

EMBED_DIM = 192

HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

NOISE_RESCUE_THRESHOLD = 0.75
MANY_SOURCES_THRESHOLD = 3


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=14400,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_dotenv()],
)
def classify_speakers():
    import hashlib
    import json
    import os
    import pickle
    import time
    from collections import defaultdict
    from datetime import datetime

    import hdbscan
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import torchaudio
    from datasets import Audio, DatasetDict, concatenate_datasets, load_from_disk
    from torch.nn.utils.rnn import pad_sequence

    GENDER_LABEL_TO_CODE = {"Female": 0, "Male": 1, "Unknown": -1}
    GENDER_CODE_TO_LABEL = {-1: "Unknown", 0: "Female", 1: "Male"}

    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _progress_line(prefix: str, done: int, total: int, started_at: float) -> str:
        elapsed = max(time.time() - started_at, 1e-6)
        rate = done / elapsed
        remaining = max(total - done, 0)
        eta = remaining / max(rate, 1e-6)
        pct = (done / max(total, 1)) * 100.0
        return (
            f"{prefix}: {done}/{total} ({pct:.1f}%) | "
            f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
        )

    def _patch_speechbrain_hf_hub() -> None:
        import huggingface_hub as hf_hub
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError

        original_download = hf_hub.hf_hub_download
        not_found_errors = (RemoteEntryNotFoundError, EntryNotFoundError)

        def wrapped_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs.setdefault("token", kwargs.pop("use_auth_token"))

            if kwargs.get("token") is False:
                token = os.environ.get("HF_TOKEN") or os.environ.get(
                    "HUGGINGFACE_HUB_TOKEN"
                )
                if token:
                    kwargs["token"] = True

            filename = (
                kwargs.get("filename")
                if "filename" in kwargs
                else (args[1] if len(args) >= 2 else None)
            )

            try:
                return original_download(*args, **kwargs)
            except not_found_errors as err:
                if filename == "custom.py":
                    raise ValueError("File not found on HF hub") from err
                raise

        hf_hub.hf_hub_download = wrapped_download

    def _patch_torchaudio_compat() -> None:
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None

    def _flatten_dataset(ds_obj):
        if isinstance(ds_obj, DatasetDict):
            split_order = ["train", "validation", "test"]
            ordered = [ds_obj[s] for s in split_order if s in ds_obj]
            ordered.extend(ds_obj[s] for s in ds_obj.keys() if s not in split_order)
            if not ordered:
                raise RuntimeError("DatasetDict is empty.")
            if len(ordered) == 1:
                return ordered[0]
            return concatenate_datasets(ordered)
        return ds_obj

    def _load_wav_as_mono_16k(path: str) -> np.ndarray:
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != ECAPA_SR:
            wav_t = torch.from_numpy(arr).unsqueeze(0)
            wav_t = torchaudio.functional.resample(
                wav_t, orig_freq=sr, new_freq=ECAPA_SR
            )
            arr = wav_t.squeeze(0).numpy()
        return arr

    def _fit_hdbscan_partition(
        partition_indices, emb_by_index, cluster_id_start, partition_name
    ):
        if len(partition_indices) < HDBSCAN_MIN_CLUSTER_SIZE:
            return (
                cluster_id_start,
                {},
                {},
                {
                    "partition": partition_name,
                    "n_points": int(len(partition_indices)),
                    "n_clusters": 0,
                    "n_noise": int(len(partition_indices)),
                },
            )

        matrix = np.stack([emb_by_index[i] for i in partition_indices], axis=0)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric=HDBSCAN_METRIC,
            cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
            core_dist_n_jobs=-1,
        )

        raw_labels = clusterer.fit_predict(matrix)
        raw_probs = clusterer.probabilities_

        counts = defaultdict(int)
        for raw_label in raw_labels:
            if raw_label != -1:
                counts[int(raw_label)] += 1

        sorted_raw_labels = sorted(counts.keys(), key=lambda k: (-counts[k], k))
        raw_to_cluster = {
            raw_label: cluster_id_start + i
            for i, raw_label in enumerate(sorted_raw_labels)
        }

        labels_by_index = {}
        confidence_by_index = {}
        for idx, raw_label, raw_prob in zip(partition_indices, raw_labels, raw_probs):
            raw_label = int(raw_label)
            if raw_label == -1:
                labels_by_index[idx] = -1
                confidence_by_index[idx] = float(raw_prob)
            else:
                labels_by_index[idx] = int(raw_to_cluster[raw_label])
                confidence_by_index[idx] = float(raw_prob)

        next_cluster_id = cluster_id_start + len(sorted_raw_labels)
        summary = {
            "partition": partition_name,
            "n_points": int(len(partition_indices)),
            "n_clusters": int(len(sorted_raw_labels)),
            "n_noise": int(sum(1 for v in labels_by_index.values() if v == -1)),
        }
        return next_cluster_id, labels_by_index, confidence_by_index, summary

    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _encode_embeddings_adaptive(wav_tensors, lengths, preferred_batch_size):
        emb_chunks = []
        cursor = 0
        microbatch_size = min(preferred_batch_size, len(wav_tensors))

        while cursor < len(wav_tensors):
            current = min(microbatch_size, len(wav_tensors) - cursor)
            tensors_chunk = wav_tensors[cursor : cursor + current]
            lengths_chunk = lengths[cursor : cursor + current]

            max_len = max(lengths_chunk)
            wav_lens = torch.tensor(
                [length / max_len for length in lengths_chunk], dtype=torch.float32
            )

            padded = pad_sequence(tensors_chunk, batch_first=True)
            padded = padded.to(device)
            wav_lens = wav_lens.to(device)

            try:
                with torch.no_grad():
                    emb = speaker_encoder.encode_batch(padded, wav_lens)
                emb_chunks.append(emb.squeeze(1).cpu().numpy())
                cursor += current
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if current <= MIN_EMBED_MICROBATCH_SIZE:
                    raise

                microbatch_size = max(MIN_EMBED_MICROBATCH_SIZE, current // 2)
                print(
                    f"OOM at clip idx {cursor}; reducing ECAPA microbatch to {microbatch_size} and retrying"
                )
                continue

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(emb_chunks, axis=0)

    def _save_embed_checkpoint(
        *,
        embeddings_matrix: np.ndarray,
        embedding_ok: np.ndarray,
        gender_code: np.ndarray,
        gender_confidence: np.ndarray,
        source_signature: str,
        n_total: int,
        last_completed_idx: int,
        n_embed_failed: int,
    ) -> None:
        np.savez_compressed(
            EMBED_CHECKPOINT_NPZ,
            embeddings_matrix=embeddings_matrix,
            embedding_ok=embedding_ok,
            gender_code=gender_code,
            gender_confidence=gender_confidence,
        )
        with open(EMBED_CHECKPOINT_META, "w") as f:
            json.dump(
                {
                    "checkpoint_version": 2,
                    "source_signature": source_signature,
                    "n_total": int(n_total),
                    "embed_dim": EMBED_DIM,
                    "last_completed_idx": int(last_completed_idx),
                    "n_embed_failed": int(n_embed_failed),
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint_npz": EMBED_CHECKPOINT_NPZ,
                },
                f,
                indent=2,
            )
        data_vol.commit()

    def _load_embed_checkpoint(source_signature: str, n_total: int):
        if not os.path.isfile(EMBED_CHECKPOINT_NPZ) or not os.path.isfile(
            EMBED_CHECKPOINT_META
        ):
            return None

        with open(EMBED_CHECKPOINT_META) as f:
            meta = json.load(f)

        if meta.get("source_signature") != source_signature:
            print("Checkpoint found but source signature changed; starting fresh.")
            return None
        if int(meta.get("n_total", -1)) != int(n_total):
            print("Checkpoint found but dataset size changed; starting fresh.")
            return None
        if int(meta.get("embed_dim", -1)) != EMBED_DIM:
            print("Checkpoint found but embedding dimension changed; starting fresh.")
            return None
        if int(meta.get("checkpoint_version", 1)) != 2:
            print("Checkpoint format changed; starting fresh.")
            return None

        payload = np.load(EMBED_CHECKPOINT_NPZ)
        embeddings_matrix = payload["embeddings_matrix"].astype(np.float32)
        embedding_ok = payload["embedding_ok"].astype(bool)
        gender_code = payload["gender_code"].astype(np.int8)
        gender_confidence = payload["gender_confidence"].astype(np.float32)

        if embeddings_matrix.shape != (n_total, EMBED_DIM):
            print(
                "Checkpoint found but embedding matrix shape mismatch; starting fresh."
            )
            return None

        return {
            "embeddings_matrix": embeddings_matrix,
            "embedding_ok": embedding_ok,
            "gender_code": gender_code,
            "gender_confidence": gender_confidence,
            "last_completed_idx": int(meta.get("last_completed_idx", -1)),
            "n_embed_failed": int(meta.get("n_embed_failed", 0)),
        }

    start_time = time.time()
    _patch_torchaudio_compat()
    _patch_speechbrain_hf_hub()

    from speechbrain.inference.speaker import EncoderClassifier

    print("=" * 72)
    print("SNA DATA PIPELINE - SPEAKER CLASSIFICATION AND RELABELING")
    print("=" * 72)

    print("Loading dataset from /data/refined ...")
    ds_obj = load_from_disk(REFINED_DATASET_PATH)
    ds = _flatten_dataset(ds_obj)

    required_columns = ["source_id", "source_speaker_id", "audio"]
    missing_columns = [c for c in required_columns if c not in ds.column_names]
    if missing_columns:
        raise RuntimeError(f"Missing required columns in dataset: {missing_columns}")

    total_clips = len(ds)
    print(f"Loaded {total_clips} clips.")

    os.makedirs(WAV_CACHE_DIR, exist_ok=True)
    os.makedirs(RELABEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Ensuring wav cache exists for all clips ...")
    n_wav_existing = 0
    n_wav_written = 0
    n_wav_failed = 0

    source_ids = [str(v) for v in ds.select_columns(["source_id"])["source_id"]]

    wav_scan_started_at = time.time()
    missing_source_ids = []
    for i, source_id in enumerate(source_ids):
        wav_path = os.path.join(WAV_CACHE_DIR, f"{source_id}.wav")
        if os.path.exists(wav_path):
            n_wav_existing += 1
        else:
            missing_source_ids.append(source_id)

        if (i + 1) % PROGRESS_EVERY_WAV == 0 or (i + 1) == total_clips:
            print(
                _progress_line(
                    "WAV cache scan", i + 1, total_clips, wav_scan_started_at
                )
            )

    if missing_source_ids:
        print(
            f"Missing WAVs detected: {len(missing_source_ids)}. "
            "Decoding audio only for missing entries ..."
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SR_CACHE, decode=True))
        source_id_to_idx = {source_id: i for i, source_id in enumerate(source_ids)}

        wav_write_started_at = time.time()
        for i, source_id in enumerate(missing_source_ids):
            wav_path = os.path.join(WAV_CACHE_DIR, f"{source_id}.wav")
            idx = source_id_to_idx[source_id]
            try:
                example = ds[idx]
                arr = np.asarray(example["audio"]["array"], dtype=np.float32)
                sf.write(wav_path, arr, AUDIO_SR_CACHE, subtype="FLOAT")
                n_wav_written += 1
            except Exception:
                n_wav_failed += 1

            if (i + 1) % PROGRESS_EVERY_WAV == 0 or (i + 1) == len(missing_source_ids):
                print(
                    _progress_line(
                        "WAV cache write",
                        i + 1,
                        len(missing_source_ids),
                        wav_write_started_at,
                    )
                )
    else:
        print("WAV cache already complete; no audio decoding needed.")

    data_vol.commit()
    print(
        f"WAV cache check complete: existing={n_wav_existing}, "
        f"written={n_wav_written}, failed={n_wav_failed}"
    )

    if not os.path.isfile(MODEL_PKL_PATH):
        raise FileNotFoundError(
            f"Gender classifier artifact not found: {MODEL_PKL_PATH}. "
            "Upload it with modal volume put before running this phase."
        )

    with open(MODEL_PKL_PATH, "rb") as f:
        payload = pickle.load(f)

    gender_clf = payload.get("model")
    gender_meta = payload.get("metadata", {})
    if gender_clf is None:
        raise RuntimeError("Loaded gender classifier payload does not contain a model.")

    if not hasattr(gender_clf, "multi_class"):
        gender_clf.multi_class = "auto"

    print("Loading ECAPA speaker encoder ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir="/tmp/ecapa_cache",
    )
    speaker_encoder.eval()
    print(f"ECAPA ready on device={device}.")

    meta_df = ds.select_columns(["source_id", "source_speaker_id"]).to_pandas()
    meta_df["source_id"] = meta_df["source_id"].astype(str)
    meta_df["source_speaker_id"] = meta_df["source_speaker_id"].astype(str)

    n_total = len(meta_df)
    source_signature = hashlib.sha1(
        "\n".join(meta_df["source_id"].tolist()).encode("utf-8")
    ).hexdigest()

    embeddings_matrix = np.zeros((n_total, EMBED_DIM), dtype=np.float32)
    embedding_ok = np.zeros(n_total, dtype=bool)
    gender_code = np.full(n_total, -1, dtype=np.int8)
    gender_confidence = np.zeros(n_total, dtype=np.float32)

    cluster_id = np.full(n_total, -1, dtype=np.int32)
    cluster_confidence = np.zeros(n_total, dtype=np.float32)
    noise_rescued = np.zeros(n_total, dtype=bool)

    resume_idx = 0
    n_embed_failed = 0
    checkpoint_state = _load_embed_checkpoint(source_signature, n_total)
    if checkpoint_state is not None:
        embeddings_matrix = checkpoint_state["embeddings_matrix"]
        embedding_ok = checkpoint_state["embedding_ok"]
        gender_code = checkpoint_state["gender_code"]
        gender_confidence = checkpoint_state["gender_confidence"]
        n_embed_failed = int(checkpoint_state["n_embed_failed"])
        resume_idx = max(0, int(checkpoint_state["last_completed_idx"]) + 1)
        print(
            "Loaded embedding/gender checkpoint: "
            f"resume_idx={resume_idx}, embedded={int(embedding_ok.sum())}, failed={n_embed_failed}"
        )

    print(
        f"Running batched ECAPA inference and gender classification (batch={EMBED_BATCH_SIZE}) ..."
    )
    batch_number = 0
    embed_started_at = time.time()
    bucket_window_size = EMBED_BATCH_SIZE * EMBED_LENGTH_BUCKET_MULTIPLIER
    if bucket_window_size < EMBED_BATCH_SIZE:
        bucket_window_size = EMBED_BATCH_SIZE

    female_col_idx = 0
    male_col_idx = 1
    classes_validated = False

    for batch_start in range(resume_idx, n_total, bucket_window_size):
        batch_number += 1
        batch_end = min(batch_start + bucket_window_size, n_total)
        batch_indices = list(range(batch_start, batch_end))

        batch_records = []

        for idx in batch_indices:
            source_id = meta_df.iloc[idx]["source_id"]
            wav_path = os.path.join(WAV_CACHE_DIR, f"{source_id}.wav")
            try:
                arr = _load_wav_as_mono_16k(wav_path)
                if arr.size == 0:
                    raise RuntimeError("empty wav")
                t = torch.from_numpy(arr)
                batch_records.append((idx, t, int(t.shape[0])))
            except Exception:
                n_embed_failed += 1

        if not batch_records:
            continue

        batch_records.sort(key=lambda rec: rec[2])
        valid_indices = [rec[0] for rec in batch_records]
        wav_tensors = [rec[1] for rec in batch_records]
        lengths = [rec[2] for rec in batch_records]

        emb_batch = _encode_embeddings_adaptive(
            wav_tensors,
            lengths,
            preferred_batch_size=EMBED_BATCH_SIZE,
        )

        emb_norm_batch = emb_batch / (
            np.linalg.norm(emb_batch, axis=1, keepdims=True) + 1e-10
        )
        try:
            proba_batch = gender_clf.predict_proba(emb_norm_batch)
        except AttributeError:
            if not hasattr(gender_clf, "coef_") or not hasattr(
                gender_clf, "intercept_"
            ):
                raise

            logits = emb_norm_batch @ gender_clf.coef_.T + gender_clf.intercept_
            if logits.ndim == 1 or logits.shape[1] == 1:
                z = logits.reshape(-1)
                p1 = 1.0 / (1.0 + np.exp(-z))
                proba_batch = np.stack([1.0 - p1, p1], axis=1).astype(np.float32)
            else:
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                proba_batch = (
                    exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                ).astype(np.float32)

        if not classes_validated:
            classes = list(getattr(gender_clf, "classes_", []))
            if len(classes) != 2:
                raise RuntimeError(f"Expected 2 gender classes, got: {classes}")

            classes_lower = [str(c).lower() for c in classes]
            if classes == [0, 1]:
                female_col_idx, male_col_idx = 0, 1
            elif classes == [1, 0]:
                female_col_idx, male_col_idx = 1, 0
            elif classes_lower == ["female", "male"]:
                female_col_idx, male_col_idx = 0, 1
            elif classes_lower == ["male", "female"]:
                female_col_idx, male_col_idx = 1, 0
            else:
                raise RuntimeError(
                    f"Unsupported gender classifier class order: {classes}. "
                    "Expected [0,1], [1,0], ['female','male'], or ['male','female']."
                )
            classes_validated = True

        for i, idx in enumerate(valid_indices):
            vec = emb_norm_batch[i].astype(np.float32)
            embeddings_matrix[idx] = vec
            embedding_ok[idx] = True

            female_p = float(proba_batch[i][female_col_idx])
            male_p = float(proba_batch[i][male_col_idx])
            conf = max(female_p, male_p)

            if conf < GENDER_UNKNOWN_THRESHOLD:
                pred_code = GENDER_LABEL_TO_CODE["Unknown"]
            elif female_p >= male_p:
                pred_code = GENDER_LABEL_TO_CODE["Female"]
            else:
                pred_code = GENDER_LABEL_TO_CODE["Male"]

            gender_code[idx] = pred_code
            gender_confidence[idx] = conf

        done_batches = batch_number
        if (
            done_batches % PROGRESS_EVERY_BATCHES == 0
            or batch_end == n_total
            or batch_number == 1
        ):
            print(_progress_line("Embed+Gender", batch_end, n_total, embed_started_at))

        if done_batches % CHECKPOINT_EVERY_BATCHES == 0 or batch_end == n_total:
            _save_embed_checkpoint(
                embeddings_matrix=embeddings_matrix,
                embedding_ok=embedding_ok,
                gender_code=gender_code,
                gender_confidence=gender_confidence,
                source_signature=source_signature,
                n_total=n_total,
                last_completed_idx=batch_end - 1,
                n_embed_failed=n_embed_failed,
            )
            print(
                f"Checkpoint saved at index {batch_end - 1} "
                f"(embedded={int(embedding_ok.sum())}, failed={n_embed_failed})"
            )

    n_valid_embeddings = int(embedding_ok.sum())
    print(f"Embeddings complete: valid={n_valid_embeddings}, failed={n_embed_failed}")

    gender_predicted = np.array(
        [GENDER_CODE_TO_LABEL[int(code)] for code in gender_code], dtype=object
    )

    female_indices = [
        i
        for i in range(n_total)
        if embedding_ok[i] and gender_code[i] == GENDER_LABEL_TO_CODE["Female"]
    ]
    male_indices = [
        i
        for i in range(n_total)
        if embedding_ok[i] and gender_code[i] == GENDER_LABEL_TO_CODE["Male"]
    ]
    unknown_indices = [
        i
        for i in range(n_total)
        if embedding_ok[i] and gender_code[i] == GENDER_LABEL_TO_CODE["Unknown"]
    ]

    print(
        "Partition sizes: "
        f"female={len(female_indices)}, male={len(male_indices)}, unknown={len(unknown_indices)}"
    )

    cluster_partition = {}
    hdbscan_summaries = []

    next_cluster_id = 0
    next_cluster_id, female_labels, female_probs, female_summary = (
        _fit_hdbscan_partition(
            female_indices,
            embeddings_matrix,
            next_cluster_id,
            "Female",
        )
    )
    hdbscan_summaries.append(female_summary)
    for idx, label in female_labels.items():
        cluster_id[idx] = label
        cluster_confidence[idx] = float(female_probs[idx])
        if label != -1:
            cluster_partition[label] = "Female"

    next_cluster_id, male_labels, male_probs, male_summary = _fit_hdbscan_partition(
        male_indices,
        embeddings_matrix,
        next_cluster_id,
        "Male",
    )
    hdbscan_summaries.append(male_summary)
    for idx, label in male_labels.items():
        cluster_id[idx] = label
        cluster_confidence[idx] = float(male_probs[idx])
        if label != -1:
            cluster_partition[label] = "Male"

    n_noise_before_rescue = int(np.sum((cluster_id == -1) & embedding_ok))
    print(f"Noise before rescue: {n_noise_before_rescue}")

    centroid_by_cluster = {}
    cluster_member_indices = defaultdict(list)
    for idx in range(n_total):
        cid = int(cluster_id[idx])
        if cid != -1 and embedding_ok[idx]:
            cluster_member_indices[cid].append(idx)

    for cid, members in cluster_member_indices.items():
        matrix = np.stack([embeddings_matrix[i] for i in members], axis=0)
        centroid = matrix.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroid_by_cluster[cid] = centroid.astype(np.float32)

    female_cluster_ids = [cid for cid, p in cluster_partition.items() if p == "Female"]
    male_cluster_ids = [cid for cid, p in cluster_partition.items() if p == "Male"]
    all_cluster_ids = sorted(centroid_by_cluster.keys())

    rescue_started_at = time.time()
    n_rescued = 0
    for idx in range(n_total):
        if not embedding_ok[idx] or cluster_id[idx] != -1:
            continue

        g = gender_predicted[idx]
        if g == "Female":
            candidates = female_cluster_ids
        elif g == "Male":
            candidates = male_cluster_ids
        else:
            candidates = all_cluster_ids

        if not candidates:
            continue

        vec = embeddings_matrix[idx]
        best_cluster = -1
        best_sim = -1.0
        for cid in candidates:
            sim = _cosine_similarity(vec, centroid_by_cluster[cid])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid

        if best_sim >= NOISE_RESCUE_THRESHOLD:
            cluster_id[idx] = int(best_cluster)
            cluster_confidence[idx] = float(best_sim)
            noise_rescued[idx] = True
            n_rescued += 1

        if (idx + 1) % PROGRESS_EVERY_RESCUE == 0 or (idx + 1) == n_total:
            print(_progress_line("Noise rescue", idx + 1, n_total, rescue_started_at))

    n_noise_after_rescue = int(np.sum((cluster_id == -1) & embedding_ok))
    print(
        f"Noise rescue complete: rescued={n_rescued}, noise_after={n_noise_after_rescue}"
    )

    cluster_gender_votes = defaultdict(lambda: defaultdict(int))
    for idx in range(n_total):
        cid = int(cluster_id[idx])
        if cid == -1:
            continue
        cluster_gender_votes[cid][str(gender_predicted[idx])] += 1

    cluster_gender = {}
    cluster_gender_confidence = {}
    for cid, votes in cluster_gender_votes.items():
        total = int(sum(votes.values()))
        top_gender, top_count = max(votes.items(), key=lambda kv: kv[1])
        confidence = float(top_count / max(total, 1))
        cluster_gender_confidence[cid] = confidence
        if confidence >= 0.75:
            cluster_gender[cid] = top_gender
        elif votes.get("Male", 0) > 0 and votes.get("Female", 0) > 0:
            cluster_gender[cid] = "Mixed"
        else:
            cluster_gender[cid] = "Unknown"

    cluster_sizes = defaultdict(int)
    cluster_speakers = defaultdict(set)
    for idx in range(n_total):
        cid = int(cluster_id[idx])
        if cid == -1:
            continue
        cluster_sizes[cid] += 1
        cluster_speakers[cid].add(meta_df.iloc[idx]["source_speaker_id"])

    rows = []
    for idx in range(n_total):
        cid = int(cluster_id[idx])
        if cid == -1:
            c_gender = "Unknown"
            c_size = 0
            c_unique_sources = 0
        else:
            c_gender = cluster_gender.get(cid, "Unknown")
            c_size = int(cluster_sizes.get(cid, 0))
            c_unique_sources = int(len(cluster_speakers.get(cid, set())))

        flags = []
        if cid == -1:
            flags.append("NOISE")
        if bool(noise_rescued[idx]):
            flags.append("RESCUED")
        if cid != -1 and c_gender == "Mixed":
            flags.append("MIXED_GENDER")
        if cid != -1 and c_unique_sources > MANY_SOURCES_THRESHOLD:
            flags.append("MANY_SOURCES")
        if not embedding_ok[idx]:
            flags.append("EMBEDDING_FAILED")

        rows.append(
            {
                "source_id": meta_df.iloc[idx]["source_id"],
                "source_speaker_id": meta_df.iloc[idx]["source_speaker_id"],
                "cluster_id": cid,
                "cluster_gender": c_gender,
                "gender_predicted": str(gender_predicted[idx]),
                "gender_confidence": round(float(gender_confidence[idx]), 6),
                "cluster_confidence": round(float(cluster_confidence[idx]), 6),
                "noise_rescued": bool(noise_rescued[idx]),
                "cluster_size": c_size,
                "unique_source_speakers": c_unique_sources,
                "flag": "|".join(flags),
            }
        )

    mapping_df = pd.DataFrame(rows)

    cluster_rows = []
    unique_cluster_ids = sorted([cid for cid in cluster_sizes.keys()])
    for cid in unique_cluster_ids:
        member_mask = mapping_df["cluster_id"] == cid
        cluster_df = mapping_df[member_mask]
        votes = cluster_gender_votes.get(cid, {})

        c_flags = []
        if cluster_gender.get(cid, "Unknown") == "Mixed":
            c_flags.append("MIXED_GENDER")
        if len(cluster_speakers.get(cid, set())) > MANY_SOURCES_THRESHOLD:
            c_flags.append("MANY_SOURCES")

        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": int(cluster_sizes[cid]),
                "unique_source_speakers": int(len(cluster_speakers[cid])),
                "cluster_gender": cluster_gender.get(cid, "Unknown"),
                "cluster_gender_confidence": round(
                    float(cluster_gender_confidence.get(cid, 0.0)), 6
                ),
                "female_clips": int(votes.get("Female", 0)),
                "male_clips": int(votes.get("Male", 0)),
                "unknown_clips": int(votes.get("Unknown", 0)),
                "mean_gender_confidence": round(
                    float(cluster_df["gender_confidence"].mean()), 6
                ),
                "mean_cluster_confidence": round(
                    float(cluster_df["cluster_confidence"].mean()), 6
                ),
                "source_speaker_ids": "|".join(sorted(cluster_speakers[cid])),
                "flag": "|".join(c_flags),
            }
        )

    cluster_report_df = pd.DataFrame(cluster_rows).sort_values(
        by=["cluster_size", "cluster_id"], ascending=[False, True]
    )

    mapping_path = os.path.join(RELABEL_DIR, "relabel_mapping.csv")
    cluster_report_path = os.path.join(RELABEL_DIR, "cluster_report.csv")
    params_path = os.path.join(RELABEL_DIR, "params_used.json")
    report_path = os.path.join(REPORTS_DIR, "speaker_relabel_audit.json")

    mapping_df.to_csv(mapping_path, index=False)
    cluster_report_df.to_csv(cluster_report_path, index=False)

    run_seconds = time.time() - start_time
    n_clusters_final = int(len(unique_cluster_ids))
    mixed_clusters = int(
        sum(1 for c in unique_cluster_ids if cluster_gender.get(c) == "Mixed")
    )

    gender_distribution_pred = {
        str(k): int(v)
        for k, v in mapping_df["gender_predicted"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    cluster_gender_distribution = {
        str(k): int(v)
        for k, v in cluster_report_df["cluster_gender"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }

    params = {
        "timestamp": datetime.now().isoformat(),
        "model_pkl_path": MODEL_PKL_PATH,
        "gender_model_metadata": gender_meta,
        "audio_sr_cache": AUDIO_SR_CACHE,
        "ecapa_sr": ECAPA_SR,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "gender_unknown_threshold": GENDER_UNKNOWN_THRESHOLD,
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC,
            "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD,
        },
        "noise_rescue_threshold": NOISE_RESCUE_THRESHOLD,
        "many_sources_threshold": MANY_SOURCES_THRESHOLD,
        "partition_summaries": hdbscan_summaries,
        "output": {
            "mapping_csv": mapping_path,
            "cluster_report_csv": cluster_report_path,
            "audit_json": report_path,
        },
    }

    audit = {
        "phase": "classify_speakers",
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(float(run_seconds), 3),
        "source_path": REFINED_DATASET_PATH,
        "total_clips": int(n_total),
        "clips_with_embedding": int(n_valid_embeddings),
        "clips_with_embedding_failure": int(n_embed_failed),
        "wav_cache": {
            "path": WAV_CACHE_DIR,
            "existing_files": int(n_wav_existing),
            "new_files_written": int(n_wav_written),
            "write_failures": int(n_wav_failed),
        },
        "clusters": {
            "count": n_clusters_final,
            "mixed_gender_clusters": mixed_clusters,
            "noise_before_rescue": int(n_noise_before_rescue),
            "noise_after_rescue": int(n_noise_after_rescue),
            "noise_rate_after_rescue": round(
                float(n_noise_after_rescue / max(n_valid_embeddings, 1)), 6
            ),
            "rescued_clips": int(n_rescued),
        },
        "gender_distribution_predicted": gender_distribution_pred,
        "cluster_gender_distribution": cluster_gender_distribution,
        "hdbscan_parameters": params["hdbscan"],
        "gender_classifier": {
            "path": MODEL_PKL_PATH,
            "metadata": gender_meta,
        },
        "output": {
            "mapping_csv": mapping_path,
            "cluster_report_csv": cluster_report_path,
            "params_json": params_path,
            "audit_json": report_path,
        },
    }

    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)

    data_vol.commit()

    print("=" * 72)
    print("SPEAKER CLASSIFICATION COMPLETE")
    print(f"Total clips: {n_total}")
    print(f"Embeddings valid/failed: {n_valid_embeddings}/{n_embed_failed}")
    print(f"Clusters: {n_clusters_final}")
    print(f"Noise before/after rescue: {n_noise_before_rescue}/{n_noise_after_rescue}")
    print(f"Mixed-gender clusters: {mixed_clusters}")
    print(f"Mapping CSV: {mapping_path}")
    print(f"Cluster report: {cluster_report_path}")
    print(f"Params JSON: {params_path}")
    print(f"Audit JSON: {report_path}")
    print("=" * 72)


@app.local_entrypoint()
def main():
    classify_speakers.remote()
