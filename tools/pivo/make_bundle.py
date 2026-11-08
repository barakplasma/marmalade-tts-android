#!/usr/bin/env python3
"""Reproduce the optional PiVo 1.4M model archive from pinned upstream files."""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "empero-ai/PiVo-1.4M"
REVISION = "515d84fc3420da6ae138f1b81017fcbc24d622a8"
OUTPUT = Path("pivo-1m-v1.tar.gz")
FILES = {
    "onnx/encoder.onnx": "encoder.onnx",
    "onnx/decoder.onnx": "decoder.onnx",
    "onnx/conditioning.onnx": "conditioning.onnx",
    "onnx/vocoder_step.onnx": "vocoder_step.onnx",
    "onnx/metadata.json": "metadata.json",
    "pivo/resources/cmudict.dict": "cmudict.dict",
    "pivo/resources/CMUDICT_LICENSE": "CMUDICT_LICENSE",
    "LICENSE": "LICENSE",
    "README.md": "MODEL_CARD.md",
    "THIRD_PARTY.md": "THIRD_PARTY.md",
}
def main():
    entries = [(target, Path(hf_hub_download(REPO, src, revision=REVISION)).read_bytes()) for src,target in FILES.items()]
    with OUTPUT.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name,data in entries:
                    info=tarfile.TarInfo("pivo-1m-v1/"+name)
                    info.size=len(data);info.mtime=0;info.mode=0o644
                    archive.addfile(info,io.BytesIO(data))
    contents=OUTPUT.read_bytes()
    meta={"filename":OUTPUT.name,"archive_sha256":hashlib.sha256(contents).hexdigest(),
          "archive_size":len(contents),"installed_size":sum(len(v) for _,v in entries),
          "model_repo":REPO,"model_revision":REVISION,"license":"Apache-2.0"}
    Path("pivo-bundle-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
