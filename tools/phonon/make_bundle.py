#!/usr/bin/env python3
"""Pinned optional Phonon 31M English q8 model for Rust-based Android runtime."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
from urllib.request import urlopen
from huggingface_hub import hf_hub_download
REPO="gradium/phonon-31m-en"
REVISION="92741c5b07d085ff840e26fea8ce43110b87852f"
OUTPUT=Path("phonon-31m-en.tar.gz")
FILES={
 "config.json":"config.json",
 "tokenizer.json":"tokenizer.json",
 "model.q8.gguf":"model.q8.gguf",
 "default-voice.safetensors":"default-voice.safetensors",
 "README.md":"MODEL_CARD.md",
}
def main():
 entries=[(dest,Path(hf_hub_download(REPO,src,revision=REVISION)).read_bytes()) for src,dest in FILES.items()]
 url="https://raw.githubusercontent.com/gradium-ai/xn-ptts/e5110afd3ead725c940b6411eec4985fb8b111aa/"
 entries.extend([("LICENSE-APACHE",urlopen(url+"LICENSE-APACHE",timeout=30).read()),
                 ("LICENSE-MIT",urlopen(url+"LICENSE-MIT",timeout=30).read())])
 with OUTPUT.open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as compressed:
   with tarfile.open(fileobj=compressed,mode="w") as archive:
    for name,data in entries:
     info=tarfile.TarInfo("phonon-31m-en/"+name)
     info.size=len(data);info.mtime=0;info.mode=0o644
     archive.addfile(info,io.BytesIO(data))
 contents=OUTPUT.read_bytes()
 meta=dict(filename=OUTPUT.name,sha256=hashlib.sha256(contents).hexdigest(),
           archive_size=len(contents),installed_size=sum(len(x) for _,x in entries),
           model_revision=REVISION,native_revision="e5110afd3ead725c940b6411eec4985fb8b111aa")
 Path("phonon-bundle-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
 print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
