# Paradee 8M v1.2

Optional model download: https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2

The model is trained by Sahil Mahendrakar, distilled from Kokoro-82M by
hexgrad. Model and upstream eSpeak-to-Misaki pronunciation conversion
code are distributed under the Apache License, Version 2.0.

The Android fork implements a Kotlin ONNX inference backend, with
phoneme-conversion rules adapted from upstream `web/misaki.js`.
The original model card and Apache-2.0 license are included in the
optional model archive. No Paradee weights are embedded in the APK.

Hugging Face revision: ed2934d4df8a4fd891e29584c5e1104649409765
Bundle: https://github.com/barakplasma/marmalade-tts-android/releases/tag/paradee-model-v1_2
