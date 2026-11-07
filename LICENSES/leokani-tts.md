# Leokani TTS (optional English model)

Model and weights: [Milliwatt Labs / leokani-tts-7M](https://huggingface.co/milliwattlabs/leokani-tts-7M)

The model weights are licensed under **CC BY-NC 4.0**. Model credit: Milliwatt Labs.
See the upstream model repository's LICENSE, README, and model card for full conditions.
This optional engine is intended for noncommercial use.

The source code for Marmalade's Leokani ONNX wrapper, pronunciation mapping,
and inverse STFT is independent Kotlin code. It does not embed Python or the
upstream inference runtime. Model distribution and licensing are distinct
from the Marmalade source license.

The upstream model's training samples used Kokoro-82M-generated audio.
