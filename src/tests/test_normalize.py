"""
Local test for normalize_text.py logic.
Reads src/unnormalized.txt, applies normalization, writes src/normalized.txt,
and prints a line-by-line diff.
"""

import re
import unicodedata


def normalize(text: str) -> str:
    # decompose unicode and strip combining diacritical marks
    # e.g. ú → u, é → e (Shona has no diacritics in standard orthography)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = unicodedata.normalize("NFC", text)

    # smart/curly apostrophes and quotes → ASCII apostrophe
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", "'").replace("\u201d", "'")
    text = text.replace("`", "'")

    # escaped double-quote artifacts e.g. ""word"" → "word"
    text = re.sub(r'"{2,}', '"', text)

    # em-dash and en-dash → space
    text = text.replace("\u2014", " ").replace("\u2013", " ")

    # spaced hyphen (compound word artifact) → hyphen e.g. "word - word" → "word-word"
    text = re.sub(r"\s+-\s+", "-", text)

    # ensure space after sentence-ending period when immediately followed by a capital
    # e.g. "wakarukwa.Wakarukwa" → "wakarukwa. Wakarukwa"
    text = re.sub(r"([a-zA-Z])\.([A-Z])", r"\1. \2", text)

    # fix spacing artifacts around punctuation e.g. " ," → ","
    text = re.sub(r"\s+([,.])", r"\1", text)

    # strip characters outside allowed set
    # allowed: letters (upper+lower), digits, . , ? ! ' " space hyphen
    text = re.sub(r'[^A-Za-z0-9.,?!\'" \-]', " ", text)

    # collapse multiple spaces and strip edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


def has_punctuation(text: str) -> bool:
    return bool(re.search(r"[.,?!]", text))


if __name__ == "__main__":
    input_path = "src/tests/unnormalized.txt"
    output_path = "src/tests/normalized.txt"

    with open(input_path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    normalized_lines = [normalize(line) for line in lines]

    with open(output_path, "w", encoding="utf-8") as f:
        for line in normalized_lines:
            f.write(line + "\n")

    print("=" * 70)
    print("NORMALIZATION TEST — LINE-BY-LINE DIFF")
    print("=" * 70)

    changed = 0
    for i, (before, after) in enumerate(zip(lines, normalized_lines), start=1):
        if before != after:
            changed += 1
            print(f"\nLine {i}:")
            print(f"  BEFORE: {before}")
            print(f"  AFTER:  {after}")

    print("\n" + "=" * 70)
    if changed == 0:
        print("✅ No changes — all lines already clean.")
    else:
        print(f"⚠️  {changed} / {len(lines)} lines were modified.")
    print(f"\n📄 Normalized output written to {output_path}")
