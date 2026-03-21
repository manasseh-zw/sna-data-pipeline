"""
Re-zip existing speaker samples on the Modal volume with integrity checks.
"""

import modal

app = modal.App("sna-zip-speaker-samples")
data_vol = modal.Volume.from_name("sna-data-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={"/data": data_vol},
)
def cleanup_volume_artifacts():
    import os
    import shutil

    targets = [
        "/data/curate_test",
        "/data/curate_test.zip",
        "/data/speaker_audit",
        "/data/speaker_audit.zip",
        "/data/speaker_samples.zip",
        "/data/speaker_samples.zip.tmp",
    ]

    print("Cleaning volume artifacts...")
    for path in targets:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  removed dir  {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"  removed file {path}")
        else:
            print(f"  missing      {path}")

    data_vol.commit()
    print("Cleanup complete.")


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/data": data_vol},
)
def zip_existing_speaker_samples():
    import hashlib
    import os
    import zipfile

    source_root = "/data/speaker_samples"
    zip_path = "/data/speaker_samples.zip"
    tmp_zip_path = "/data/speaker_samples.zip.tmp"

    if not os.path.isdir(source_root):
        raise RuntimeError(f"Source folder missing: {source_root}")

    if os.path.exists(tmp_zip_path):
        os.remove(tmp_zip_path)

    print(f"Creating archive from {source_root} ...")
    written_files = 0
    with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_root):
            for name in sorted(files):
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, source_root)
                zf.write(abs_path, rel_path)
                written_files += 1

    with zipfile.ZipFile(tmp_zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity check failed at member: {bad_member}")

    os.replace(tmp_zip_path, zip_path)
    data_vol.commit()

    with open(zip_path, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Archive ready: {zip_path}")
    print(f"Files zipped: {written_files}")
    print(f"Zip size: {zip_mb:.1f} MB")
    print(f"SHA256: {sha256}")


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={"/data": data_vol},
)
def verify_volume_zip():
    import hashlib
    import os
    import zipfile

    zip_path = "/data/speaker_samples.zip"
    if not os.path.isfile(zip_path):
        raise RuntimeError(f"Missing file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = len(zf.infolist())
        bad_member = zf.testzip()

    with open(zip_path, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()

    size_bytes = os.path.getsize(zip_path)
    size_mib = size_bytes / (1024 * 1024)

    print(f"Zip path: {zip_path}")
    print(f"Members: {members}")
    print(f"Size bytes: {size_bytes}")
    print(f"Size MiB: {size_mib:.2f}")
    print(f"SHA256: {sha256}")
    print(f"Integrity: {'OK' if bad_member is None else f'BAD ({bad_member})'}")


@app.local_entrypoint()
def main(action: str = "zip"):
    if action == "cleanup":
        cleanup_volume_artifacts.remote()
    elif action == "zip":
        zip_existing_speaker_samples.remote()
    elif action == "verify":
        verify_volume_zip.remote()
    else:
        raise ValueError("action must be one of: cleanup, zip, verify")
