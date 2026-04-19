"""
Remove Resemble Enhance bakeoff artifacts from an exported speaker folder.

This deletes only the enhancement outputs and logs created by:
  - src/resemble_enhance_bakeoff.py
  - src/tests/audio/pull_resemble_enhance_bakeoff.py

Default target:
  /data/speakers/1_Female_full

Example:
  modal run src/cleanup_resemble_enhance_bakeoff.py \
      --speaker-dir /data/speakers/1_Female_full
"""

from __future__ import annotations

import modal

app = modal.App("sna-cleanup-resemble-enhance-bakeoff")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10")

DEFAULT_SPEAKER_DIR = "/data/speakers/1_Female_full"


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=1800,
    volumes={"/data": data_vol},
)
def cleanup_resemble_enhance_bakeoff(
    speaker_dir: str = DEFAULT_SPEAKER_DIR,
    remove_listen_pack: bool = True,
):
    import os
    import shutil
    from pathlib import Path

    speaker_root = Path(speaker_dir)
    targets = [
        speaker_root / "audio_enhanced",
        speaker_root / "resemble_enhance_manifest.csv",
        speaker_root / "resemble_enhance_failures.csv",
        speaker_root / "resemble_enhance_report.json",
    ]

    if remove_listen_pack:
        targets.extend(
            [
                speaker_root / "resemble_listen_pack",
                speaker_root / "resemble_listen_pack.zip",
            ]
        )

    print("=" * 72)
    print("SNA DATA PIPELINE - CLEANUP RESEMBLE ENHANCE BAKEOFF")
    print("=" * 72)
    print(f"  Speaker dir        : {speaker_root}")
    print(f"  Remove listen pack : {remove_listen_pack}")
    print()

    if not speaker_root.is_dir():
        raise FileNotFoundError(f"Speaker folder not found: {speaker_root}")

    removed = 0
    missing = 0

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed += 1
            print(f"removed dir  {target}")
        elif target.is_file():
            os.remove(target)
            removed += 1
            print(f"removed file {target}")
        else:
            missing += 1
            print(f"missing      {target}")

    data_vol.commit()

    print()
    print("=" * 72)
    print("CLEANUP COMPLETE")
    print("=" * 72)
    print(f"  Removed : {removed}")
    print(f"  Missing : {missing}")


@app.local_entrypoint()
def main(
    speaker_dir: str = DEFAULT_SPEAKER_DIR,
    remove_listen_pack: bool = True,
):
    cleanup_resemble_enhance_bakeoff.remote(
        speaker_dir=speaker_dir,
        remove_listen_pack=remove_listen_pack,
    )
