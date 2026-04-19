"""
Local verification for the Resemble Enhance bakeoff helpers.

Checks:
  - deterministic stratified sampling
  - exact sample count
  - expected low/mid/high tier split for 50 samples
  - paired local listening filenames

Run from repo root:
  python src/tests/audio/test_resemble_enhance_bakeoff.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ENHANCE_SCRIPT = REPO_ROOT / "src" / "resemble_enhance_bakeoff.py"
PULL_SCRIPT = REPO_ROOT / "src" / "tests" / "audio" / "pull_resemble_enhance_bakeoff.py"


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_metadata(rows: int = 90) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        data.append(
            {
                "file_name": f"sna_{idx:05d}.wav",
                "transcription": f"clip {idx}",
                "quality_score": float(idx),
                "duration": 5.0 + (idx % 7),
                "source_id": f"sna_{idx:05d}",
            }
        )
    return pd.DataFrame(data)


def main():
    enhance = load_module(ENHANCE_SCRIPT, "resemble_enhance_bakeoff")
    puller = load_module(PULL_SCRIPT, "pull_resemble_enhance_bakeoff")

    metadata = make_metadata()
    sample_a = enhance.build_stratified_sample(metadata, num_samples=50, seed=42)
    sample_b = enhance.build_stratified_sample(metadata, num_samples=50, seed=42)

    assert len(sample_a) == 50, f"Expected 50 rows, got {len(sample_a)}"
    assert sample_a["file_name"].tolist() == sample_b["file_name"].tolist(), (
        "Sampling should be deterministic for the same seed"
    )

    tier_counts = sample_a["quality_tier"].value_counts().to_dict()
    expected_counts = {"low": 17, "mid": 16, "high": 17}
    assert tier_counts == expected_counts, (
        f"Unexpected tier counts: {tier_counts} != {expected_counts}"
    )

    assert puller.paired_name("sna_68669.wav", "original") == "sna_68669_original.wav"
    assert puller.paired_name("sna_68669.wav", "enhanced") == "sna_68669_enhanced.wav"
    assert enhance.mounted_to_volume_path("/data/speakers/1_Female_full") == "/speakers/1_Female_full"

    print("=" * 72)
    print("RESEMBLE ENHANCE BAKEOFF HELPER TESTS")
    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
