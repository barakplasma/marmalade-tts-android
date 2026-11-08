#!/usr/bin/env python3
"""Reproducibly bundle pinned, unmodified Paradee v1.2 int8 model data."""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

REPO = "sahilmahendrakar/Paradee-8M-v1.2"
REVISION = "ed2934d4df8a4fd891e29584c5e1104649409765"
GITHUB_LICENSE = "https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/LICENSE"
OUTPUT = Path("paradee-8m-v1_2.tar.gz")
FILES = {
    "onnx/paradee_int8.onnx": "model.onnx",
    "config.json": "config.json",
    "README.md": "MODEL_CARD.md",
}

def main():
    entries = [(dest, Path(hf_hub_download(REPO, src, revision=REVISION)).read_bytes())
               for src, dest in FILES.items()]
    entries.append(("LICENSE", urlopen(GITHUB_LICENSE, timeout=30).read()))
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for filename, data in entries:
                    info = tarfile.TarInfo("paradee-8m-v1_2/" + filename)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
    archive_bytes = OUTPUT.read_bytes()
    manifest = dict(
        filename=OUTPUT.name,
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        archive_size=len(archive_bytes),
        installed_size=sum(len(data) for _, data in entries),
        model_repo=REPO, model_revision=REVISION, license="Apache-2.0",
    )
    Path("paradee-bundle-metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
