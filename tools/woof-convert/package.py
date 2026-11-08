#!/usr/bin/env python3
"""Build and validate an optional Underdog Woof ONNX model archive.

This script runs on GitHub Actions, not on Android.
No original NumPy code or Python interpreter is copied into the APK.
Converted model weights retain the Underdog Noncommercial License.
"""
import gzip
import hashlib
import io
import json
import os
import runpy
import shutil
import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

REPO = "ConwayResearch/Underdog-Woof"
REVISION = "be702960539777fc3f698e54335276a8391b00f9"
ROOT = Path("woof-onnx-v1")
OUTPUT = Path("woof-onnx-v1.tar.gz")

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> None:
    ROOT.mkdir(exist_ok=True)
    source = Path(snapshot_download(
        REPO,
        revision=REVISION,
        allow_patterns=["woof_int8.npz", "underdog_woof/*.py",
                        "LICENSE", "NOTICE", "README.md"],
    ))
    import sys
    sys.path.insert(0, str(source))
    from underdog_woof.runtime import unflatten, Speech
    with np.load(source / "woof_int8.npz", allow_pickle=False) as z:
        st = unflatten(z)
    speech = Speech(st["speech"])

    for filename, arr in [
        ("text_emb.npy", speech.text_emb),
        ("bos.npy", speech.bos),
        ("lat_in_w.npy", speech.lat_in[0]),
        ("lat_in_b.npy", speech.lat_in[1]),
        ("voice_juno.npy", st["voices"]["juno"]),
        ("voice_mira.npy", st["voices"]["mira"]),
    ]:
        np.save(ROOT / filename, np.asarray(arr, np.float32), allow_pickle=False)
    for filename in ["LICENSE", "NOTICE"]:
        shutil.copyfile(source / filename, ROOT / filename)
    shutil.copyfile(source / "README.md", ROOT / "MODEL_CARD.md")

    scripts = [
        ("tools/woof-convert/flow.py", "woof_head.onnx", "flow_head.onnx"),
        ("tools/woof-convert/speech.py", "woof_speech_step.onnx", "speech_step.onnx"),
        ("tools/woof-convert/decoder.py", "woof_full_decoder_fixed_T8.onnx", "decoder.onnx"),
    ]
    for script, generated, target in scripts:
        print("EXPORT", script, flush=True)
        runpy.run_path(script, run_name="__main__")
        os.replace(generated, ROOT / target)
        print("PASS", target, (ROOT / target).stat().st_size, flush=True)

    files = {p.name: {"bytes":p.stat().st_size,
                      "sha256":hash_bytes(p.read_bytes())}
             for p in sorted(ROOT.iterdir()) if p.is_file()}
    manifest = {
        "format": "marmalade-underdog-woof-onnx-v1",
        "source_repo": REPO,
        "source_revision": REVISION,
        "license": "Underdog Noncommercial License",
        "language": "en-US",
        "sample_rate": 24000,
        "frames_per_second": 25,
        "voices": ["juno", "mira"],
        "files": files,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for file in sorted(ROOT.iterdir()):
                    info = tarfile.TarInfo(f"woof-onnx-v1/{file.name}")
                    info.mode = 0o644
                    info.mtime = 0
                    info.size = file.stat().st_size
                    with file.open("rb") as data:
                        tar.addfile(info, data)
    size = OUTPUT.stat().st_size
    summary = {
        "filename": OUTPUT.name,
        "sha256": hash_bytes(OUTPUT.read_bytes()),
        "archive_size": size,
        "installed_size": sum(p.stat().st_size for p in ROOT.iterdir()),
        "model_revision": REVISION,
        "license": "Underdog Noncommercial License",
    }
    Path("woof-bundle-metadata.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("BUNDLE_RESULT",json.dumps(summary),flush=True)

if __name__ == "__main__":
    main()
