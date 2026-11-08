# Leokani 7M experimental engine (fork)

This fork provides an optional, installable English Leokani TTS model in the
F-Droid-flavor experimental APK. No Python runtime is used on Android.

## Install and try it

1. Install the latest **Leokani experimental RC** from
   https://github.com/barakplasma/marmalade-tts-android/releases
   (the app's ID ends in `.rc`, so it can coexist with Marmalade).
2. Open **Engines**, find **Leokani 7M (experimental)**, and select **Install**.
   The app downloads the 6.3 MB bundle from this fork's GitHub release, checks
   its SHA256, extracts `model.onnx`, `config.json`, `LICENSE`, and
   `MODEL_CARD.md`, and verifies the layout.
3. Pick the **Leokani** English voice on the Speak screen or as a voice alias
   for system text-to-speech. Try short English sentences first.
4. Compare sound quality and speed with Kitten and Pocket on the same phone.

## Implementation

- ONNX Runtime already shipped by Marmalade, native eSpeak-ng JNI, Kotlin
  Kokoro-IPA normalization, pure Kotlin inverse STFT, 24 kHz PCM16.
- ONNX model inputs: `ids` int64 [1, tokens], `speed` float32 [1].
- Outputs: `mag` and `phase` float32 [1, 513, frames].
- User playback speed uses Marmalade's existing downstream time-stretcher.
- Model bytes are *not* embedded in the APK or committed to source control.

## Source and model attribution

- Original: https://huggingface.co/milliwattlabs/leokani-tts-7M
- Pinned revision: `ba0beb24f7b124dc1eb297ffc28d01db892462bd`.
- Fork model archive: https://github.com/barakplasma/marmalade-tts-android/releases/tag/leokani-model-v1
- Archive SHA256: `fbc1eb4cba47cbc7dfc532cd16515e488e414fb6bbe2d79894d642820e91d506`
- Reproduce: `python -m pip install huggingface_hub &&
  python tools/leokani/make_bundle.py`.
- License: CC BY-NC 4.0, noncommercial; both `LICENSE` and upstream
  `MODEL_CARD.md` accompany the downloaded model.

## Known limitations

The engine is experimental. The model and decoder have been tested at
the numerical level, and the APK CI build compiles, but phone playback,
phoneme fidelity, install/uninstall UX, long reads, and cold-start performance
must still be verified on real Android hardware. The preliminary BASIC/HEAVY
catalog grades are not measured performance claims.
