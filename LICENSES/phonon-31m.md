# Gradium Phonon 31M English

Model data: https://huggingface.co/gradium/phonon-31m-en
Revision: `92741c5b07d085ff840e26fea8ce43110b87852f`
Optional model bundle: https://github.com/barakplasma/marmalade-tts-android/releases/tag/phonon-model-v1

Rust neural inference runtime by Gradium AI:
https://github.com/gradium-ai/xn-ptts
Pinned FFI runtime revision: `e5110afd3ead725c940b6411eec4985fb8b111aa`
Precompiled ARM64 and x86_64 CPU libraries:
https://github.com/barakplasma/marmalade-tts-android/releases/tag/phonon-native-v1

The original Kotlin JNA wrapper was adapted with its upstream license
notices. The native model is downloadable; the APK includes only the
runtime. Gradium's source tree contains Apache-2.0 and MIT license texts.
Phonon includes no Python interpreter.
