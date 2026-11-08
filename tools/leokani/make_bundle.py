#!/usr/bin/env python3
"""Build a Marmalade install archive from the official Leokani-7M int8 model.

Requires: pip install huggingface_hub
Run from the repository root: python tools/leokani/make_bundle.py
Outputs a tar.gz and its SHA256 / sizes; no runtime Python on Android.
"""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "milliwattlabs/leokani-tts-7M"
REVISION = "ba0beb24f7b124dc1eb297ffc28d01db892462bd"
OUTPUT = Path("leokani-tts-7m.tar.gz")
FILES = {
    "leokani-tts-int8.onnx": "model.onnx",
    "config.json": "config.json",
    "LICENSE": "LICENSE",
    "README.md": "MODEL_CARD.md",
}


def main():
    entries = []
    for upstream, target in FILES.items():
        data = Path(hf_hub_download(REPO, upstream, revision=REVISION)).read_bytes()
        entries.append((target, data))
    # Deterministic archive: normalize both TAR member timestamps and the
    # outer GZIP header (tarfile.open(w:gz) otherwise embeds wall clock time).
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for filename, data in entries:
                    info = tarfile.TarInfo("leokani-tts-7m/" + filename)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
    archive_data = OUTPUT.read_bytes()
    manifest = {
        "filename": OUTPUT.name,
        "archive_sha256": hashlib.sha256(archive_data).hexdigest(),
        "archive_size": len(archive_data),
        "installed_size": sum(len(data) for _, data in entries),
        "model_repo": REPO,
        "model_revision": REVISION,
        "license": "CC BY-NC 4.0",
    }
    Path("leokani-bundle-metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
