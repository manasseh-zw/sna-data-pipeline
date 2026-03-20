import modal

app = modal.App("sna-normalize-text")

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
def normalize_text():
    import json
    import os
    import re
    from collections import Counter
    from datetime import datetime
    from datasets import load_from_disk
    import pandas as pd

    print("=" * 60)
    print("SNA DATA PIPELINE — PHASE 2: NORMALIZE TEXT")
    print("=" * 60)

    print("\n📂 Loading dataset from /data/raw/...")
    dataset = load_from_disk("/data/raw/")
    print(f"   {len(dataset)} rows loaded")

    # --- Character frequency audit (before) ---
    def char_freq(texts):
        counter = Counter()
        for t in texts:
            counter.update(t)
        return dict(counter.most_common())

    transcriptions_before = dataset["transcription"]
    freq_before = char_freq(transcriptions_before)

    unexpected_before = {
        c: n for c, n in freq_before.items()
        if not re.match(r"[A-Za-z0-9.,?!' \-]", c)
    }
    print(f"\n📊 Unexpected characters before normalisation: {len(unexpected_before)}")
    if unexpected_before:
        for ch, cnt in sorted(unexpected_before.items(), key=lambda x: -x[1])[:20]:
            print(f"   {repr(ch):10} {cnt:>6}x")

    # --- Normalisation ---
    def normalize(example):
        import unicodedata

        text = example["transcription"]

        # strip diacritics — Shona standard orthography has none (e.g. ú → u)
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = unicodedata.normalize("NFC", text)

        # smart/curly apostrophes and quotes → ASCII apostrophe
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", "'").replace("\u201d", "'")
        text = text.replace("`", "'")

        # escaped double-quote artifacts e.g. ""word"" → "word"
        text = re.sub(r'"{2,}', '"', text)

        # em-dash and en-dash → space (but leave ASCII hyphens intact)
        text = text.replace("\u2014", " ").replace("\u2013", " ")

        # spaced hyphen → hyphen e.g. "word - word" → "word-word"
        text = re.sub(r"\s+-\s+", "-", text)

        # ensure space after sentence-ending period followed by a capital
        # e.g. "wakarukwa.Wakarukwa" → "wakarukwa. Wakarukwa"
        text = re.sub(r"([a-zA-Z])\.([A-Z])", r"\1. \2", text)

        # fix spacing artifacts around punctuation (e.g. " ," → ",")
        text = re.sub(r"\s+([,.])", r"\1", text)

        # strip characters outside allowed set
        # allowed: letters (upper+lower), digits, . , ? ! ' " space hyphen
        text = re.sub(r'[^A-Za-z0-9.,?!\'" \-]', " ", text)

        # collapse multiple spaces and strip edges
        text = re.sub(r"\s+", " ", text).strip()

        example["transcription"] = text
        example["has_punctuation"] = bool(re.search(r"[.,?!]", text))
        return example

    map_workers = max(1, min(4, (os.cpu_count() or 1)))
    dataset = dataset.map(
        normalize,
        num_proc=map_workers,
        desc=f"Normalising text ({map_workers} workers)",
    )

    # --- Character frequency audit (after) ---
    transcriptions_after = dataset["transcription"]
    freq_after = char_freq(transcriptions_after)

    unexpected_after = {
        c: n for c, n in freq_after.items()
        if not re.match(r"[A-Za-z0-9.,?!' \-]", c)
    }
    print(f"\n📊 Unexpected characters after normalisation: {len(unexpected_after)}")
    if unexpected_after:
        for ch, cnt in sorted(unexpected_after.items(), key=lambda x: -x[1])[:20]:
            print(f"   {repr(ch):10} {cnt:>6}x")
    else:
        print("   ✅ None — character set is clean")

    # --- has_punctuation summary ---
    has_punct = pd.Series(dataset["has_punctuation"])
    print(f"\n📝 TRANSCRIPTION STATS")
    print(f"   has_punctuation=True:  {has_punct.sum():>6} ({has_punct.mean()*100:.1f}%)")
    print(f"   has_punctuation=False: {(~has_punct).sum():>6} ({(~has_punct).mean()*100:.1f}%)")

    lengths = pd.Series([len(t) for t in transcriptions_after])
    print(f"   Mean chars:   {lengths.mean():.1f}")
    print(f"   Median chars: {lengths.median():.1f}")
    print(f"   Min chars:    {lengths.min()}")
    print(f"   Max chars:    {lengths.max()}")
    empty = (lengths == 0).sum()
    print(f"   Empty after normalisation: {empty}")

    # --- Sample diff ---
    print("\n🔍 SAMPLE DIFF (first 3 rows)")
    print("-" * 60)
    for i in range(3):
        before = transcriptions_before[i][:80]
        after = transcriptions_after[i][:80]
        if before != after:
            print(f"   BEFORE: {before}")
            print(f"   AFTER:  {after}")
        else:
            print(f"   [{i}] unchanged: {after[:60]}...")
        print()

    # --- Audit report ---
    audit = {
        "phase": "normalize_text",
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(dataset),
        "columns": dataset.column_names,
        "char_audit": {
            "unexpected_chars_before": {
                repr(k): int(v) for k, v in list(unexpected_before.items())[:30]
            },
            "unexpected_chars_after": {
                repr(k): int(v) for k, v in list(unexpected_after.items())[:30]
            },
        },
        "has_punctuation": {
            "true": int(has_punct.sum()),
            "false": int((~has_punct).sum()),
            "pct_true": round(float(has_punct.mean()) * 100, 2),
        },
        "transcription_length": {
            "mean_chars": round(float(lengths.mean()), 1),
            "median_chars": round(float(lengths.median()), 1),
            "min_chars": int(lengths.min()),
            "max_chars": int(lengths.max()),
            "empty_count": int(empty),
        },
    }

    os.makedirs("/data/reports", exist_ok=True)
    report_path = "/data/reports/02_normalize_text_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"💾 Audit report saved → {report_path}")

    # --- Save to /data/refined/ ---
    print("\n💾 Saving to /data/refined/...")
    os.makedirs("/data/refined", exist_ok=True)
    dataset.save_to_disk("/data/refined/")
    data_vol.commit()

    print("\n" + "=" * 60)
    print("✅ NORMALIZE TEXT COMPLETE")
    print(f"   {len(dataset)} rows saved to /data/refined/")
    print(f"   columns: {dataset.column_names}")
    print("=" * 60)


@app.local_entrypoint()
def main():
    normalize_text.remote()
