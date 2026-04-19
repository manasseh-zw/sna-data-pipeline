"""
Build a local listening pack for a completed Resemble Enhance bakeoff.

Reads the saved manifest from the speaker export folder, copies the original
and enhanced WAVs into one flat folder with paired filenames, writes a compact
CSV, then zips the pack for local download.

Example:
  modal run src/tests/audio/pull_resemble_enhance_bakeoff.py \
      --speaker-dir /data/speakers/1_Female_full
"""

from __future__ import annotations

import shutil
from pathlib import Path

import modal

app = modal.App("sna-pull-resemble-enhance-bakeoff")

data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "pandas",
)

DEFAULT_SPEAKER_DIR = "/data/speakers/1_Female_full"
DEFAULT_OUTPUT_SUBDIR = "audio_enhanced"
DEFAULT_PACK_DIRNAME = "resemble_listen_pack"
DEFAULT_PACK_ZIP_NAME = "resemble_listen_pack.zip"
MANIFEST_CSV_NAME = "resemble_enhance_manifest.csv"
PACK_METADATA_NAME = "listen_manifest.csv"
MOUNT_PREFIX = "/data"
SUCCESS_STATUSES = {"success", "reused_existing"}


def mounted_to_volume_path(path_str: str) -> str:
    path = Path(path_str)
    mount_root = Path(MOUNT_PREFIX)
    try:
        relative = path.relative_to(mount_root)
    except ValueError:
        return path_str
    return "/" + relative.as_posix()


def paired_name(file_name: str, variant: str) -> str:
    original = Path(file_name)
    return f"{original.stem}_{variant}{original.suffix}"


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/data": data_vol},
)
def pull_resemble_enhance_bakeoff(
    speaker_dir: str = DEFAULT_SPEAKER_DIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    manifest_name: str = MANIFEST_CSV_NAME,
    pack_dirname: str = DEFAULT_PACK_DIRNAME,
    pack_zip_name: str = DEFAULT_PACK_ZIP_NAME,
):
    import os
    import zipfile

    import pandas as pd

    speaker_root = Path(speaker_dir)
    audio_dir = speaker_root / "audio"
    enhanced_dir = speaker_root / output_subdir
    manifest_csv = speaker_root / manifest_name
    pack_dir = speaker_root / pack_dirname
    pack_zip = speaker_root / pack_zip_name
    pack_metadata = pack_dir / PACK_METADATA_NAME
    zip_tmp = pack_zip.with_suffix(".zip.tmp")

    print("=" * 72)
    print("SNA DATA PIPELINE - PULL RESEMBLE ENHANCE BAKEOFF")
    print("=" * 72)
    print(f"  Speaker dir    : {speaker_root}")
    print(f"  Manifest CSV   : {manifest_csv}")
    print(f"  Audio dir      : {audio_dir}")
    print(f"  Enhanced dir   : {enhanced_dir}")
    print(f"  Pack dir       : {pack_dir}")
    print(f"  Pack zip       : {pack_zip}")
    print()

    if not speaker_root.is_dir():
        raise FileNotFoundError(f"Speaker folder not found: {speaker_root}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Missing audio directory: {audio_dir}")
    if not enhanced_dir.is_dir():
        raise FileNotFoundError(f"Missing enhanced directory: {enhanced_dir}")
    if not manifest_csv.is_file():
        raise FileNotFoundError(
            f"Missing bakeoff manifest: {manifest_csv}\n"
            "Run the enhancement script first."
        )

    manifest = pd.read_csv(manifest_csv)
    if "status" not in manifest.columns or "file_name" not in manifest.columns:
        raise RuntimeError(f"Manifest is missing required columns: {manifest_csv}")

    selected = manifest[manifest["status"].isin(SUCCESS_STATUSES)].copy()
    if selected.empty:
        raise RuntimeError("No successful rows found in the bakeoff manifest.")

    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    pack_rows: list[dict[str, object]] = []

    print(f"Copying {len(selected)} successful pairs into flat listening pack ...")
    ordered = selected.sort_values(
        by=["file_name"],
        ascending=[True],
        kind="mergesort",
    ).reset_index(drop=True)

    for index, row in enumerate(ordered.itertuples(index=False), start=1):
        source_name = str(row.file_name)
        original_path = audio_dir / source_name
        enhanced_path = enhanced_dir / source_name

        if not original_path.is_file():
            raise FileNotFoundError(f"Missing original audio for pack: {original_path}")
        if not enhanced_path.is_file():
            raise FileNotFoundError(f"Missing enhanced audio for pack: {enhanced_path}")

        original_pack_name = paired_name(source_name, "original")
        enhanced_pack_name = paired_name(source_name, "enhanced")

        shutil.copy2(original_path, pack_dir / original_pack_name)
        shutil.copy2(enhanced_path, pack_dir / enhanced_pack_name)

        pack_rows.append(
            {
                "sample_id": getattr(row, "sample_id", index),
                "source_id": getattr(row, "source_id", ""),
                "file_name": source_name,
                "quality_tier": getattr(row, "quality_tier", ""),
                "quality_score": getattr(row, "quality_score", None),
                "duration_sec": getattr(row, "duration_sec", None),
                "transcription": getattr(row, "transcription", ""),
                "original_name": original_pack_name,
                "enhanced_name": enhanced_pack_name,
            }
        )

    pd.DataFrame(pack_rows).to_csv(pack_metadata, index=False)

    if zip_tmp.exists():
        zip_tmp.unlink()

    print(f"Creating zip archive at {pack_zip} ...")
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pack_dir.iterdir()):
            if path.is_file():
                archive.write(path, path.name)

    with zipfile.ZipFile(zip_tmp, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity check failed at member: {bad_member}")

    os.replace(zip_tmp, pack_zip)
    data_vol.commit()

    volume_zip_path = mounted_to_volume_path(str(pack_zip))
    print()
    print("=" * 72)
    print("LISTENING PACK READY")
    print("=" * 72)
    print(f"  Files packed       : {len(pack_rows) * 2}")
    print(f"  Pair count         : {len(pack_rows)}")
    print(f"  Pack dir           : {pack_dir}")
    print(f"  Pack metadata      : {pack_metadata}")
    print(f"  Zip path           : {pack_zip}")
    print(f"  Volume zip path    : {volume_zip_path}")
    print()
    print("Download with:")
    print(
        "  modal volume get sna-data-vol "
        f"{volume_zip_path} src/tests/audio/{pack_zip.name}"
    )


@app.local_entrypoint()
def main(
    speaker_dir: str = DEFAULT_SPEAKER_DIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    manifest_name: str = MANIFEST_CSV_NAME,
    pack_dirname: str = DEFAULT_PACK_DIRNAME,
    pack_zip_name: str = DEFAULT_PACK_ZIP_NAME,
):
    pull_resemble_enhance_bakeoff.remote(
        speaker_dir=speaker_dir,
        output_subdir=output_subdir,
        manifest_name=manifest_name,
        pack_dirname=pack_dirname,
        pack_zip_name=pack_zip_name,
    )
